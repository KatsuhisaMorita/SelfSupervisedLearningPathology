# -*- coding: utf-8 -*-
"""
# barlowtwins with recursive resuming

@author: Katsuhisa MORITA
"""
# path setting
PROJECT_PATH = '/work/gd43/a97001'

# packages installed in the current environment
import sys
import os
import gc
import datetime
import argparse
import time
import random

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from timm.scheduler import CosineLRScheduler

# original packages in src
sys.path.append(f"{PROJECT_PATH}/src/SelfSupervisedLearningPathology")
import sslmodel
from sslmodel import data_handler as dh
import sslmodel.sslutils as sslutils
import tggate.utils as utils

# argument
parser = argparse.ArgumentParser(description='CLI learning')
# base settings
parser.add_argument('--note', type=str, help='barlowtwins running')
parser.add_argument('--seed', type=int, default=0)
# data settings
parser.add_argument('--model_path', type=str, help='dir_model')
parser.add_argument('--dir_result', type=str, help='result')
parser.add_argument('--mouse_dataset', action='store_true')
parser.add_argument('--rat_dataset', action='store_true')
parser.add_argument('--eisai_dataset', action='store_true')
parser.add_argument('--shionogi_dataset', action='store_true')
# model/learning settings
parser.add_argument('--model_name', type=str, default='ResNet18') # model architecture name
parser.add_argument('--ssl_name', type=str, default='barlowtwins') # ssl architecture name
parser.add_argument('--num_epoch', type=int, default=50) # epoch
parser.add_argument('--batch_size', type=int, default=64) # batch size
parser.add_argument('--lr', type=float, default=0.01) # learning rate
parser.add_argument('--patience', type=int, default=3) # early stopping
parser.add_argument('--delta', type=float, default=0.002) # early stopping
# Transform (augmentation) settings
parser.add_argument('--color_plob', type=float, default=0.8)
parser.add_argument('--blur_plob', type=float, default=0.4)
parser.add_argument('--solar_plob', type=float, default=0.)
# scheduler
parser.add_argument('--lr_min', type=float, default=1e-5)
parser.add_argument('--warmup_t', type=int, default=15)
parser.add_argument('--warmup_lr_init', type=float, default=1e-5)

args = parser.parse_args()
sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

# prepare data
class Dataset_Batch(torch.utils.data.Dataset):
    """ load for each version """
    def __init__(self,
                transform=None,
                ):
        # set transform
        if type(transform)!=list:
            self._transform = [transform]
        else:
            self._transform = transform
        # load data
        if args.rat_dataset:
            filein="/work/gd43/share/Lab/Rat_DILI/batch/batch.npy"
        elif args.mouse_dataset:
            filein="/work/gd43/share/Lab/mouse_DILI/batch/batch.npy"
        elif args.eisai_dataset:
            filein="/work/gd43/share/pharm/eisai/batch/batch.npy"
        elif args.shionogi_dataset:
            filein="/work/gd43/share/pharm/shionogi/batch/batch.npy"
        with open(filein, 'rb') as f:
            self.data = np.load(f)
        self.datanum = len(self.data)
        gc.collect()

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = self.data[idx]
        out_data = Image.fromarray(out_data).convert("RGB")
        if self._transform:
            for t in self._transform:
                out_data = t(out_data)
        return out_data

def prepare_data(batch_size:int=32):
    """
    data preparation
    
    """
    # normalization
    train_transform = ssl_class.prepare_transform(
        color_plob=args.color_plob, blur_plob=args.blur_plob, solar_plob=args.solar_plob,
    )
    train_dataset = Dataset_Batch(transform=train_transform)
    # to loader
    train_loader = dh.prep_dataloader(train_dataset, batch_size)
    return train_loader

# prepare model
def prepare_model(
        model_name:str='ResNet18', patience:int=7, delta:float=0, lr:float=0.003, num_epoch:int=150):
    """
    preparation of models
    Parameters
    ----------
        model_name (str)
            model architecture name
        
        patience (int)
            How long to wait after last time validation loss improved.

        delta (float)
            Minimum change in the monitored quantity to qualify as an improvement.

    """
    # model building with indicated name
    try:
        encoder = utils.DICT_MODEL[model_name][0](weights=None)
        size=utils.DICT_MODEL[model_name][1]
    except:
        print("indicated model name is not implemented")
        ValueError
    if model_name=="DenseNet121":
        backbone = nn.Sequential(
            *list(encoder.children())[:-1],
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
            )
    else:
        backbone = nn.Sequential(
            *list(encoder.children())[:-1],
            )
    model, criterion = ssl_class.prepare_model(backbone, head_size=size)
    model.load_state_dict(torch.load(args.model_path))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=num_epoch, lr_min=args.lr_min,
        warmup_t=args.warmup_t, warmup_lr_init=args.warmup_lr_init, warmup_prefix=True)
    early_stopping = sslmodel.utils.EarlyStopping(patience=patience, delta=delta, path=f'{DIR_NAME}/checkpoint.pt')
    return model, criterion, optimizer, scheduler, early_stopping

# train epoch
def train_epoch(model, data_loader, criterion, optimizer, epoch):
    """
    train for epoch
    with minibatch
    """
    model.train() # training
    train_batch_loss = []
    # train
    for data in data_loader:
        loss = ssl_class.calc_loss(
            model, data, criterion,
        )
        train_batch_loss.append(loss.item())
        optimizer.zero_grad() # reset gradients
        loss.backward() # backpropagation
        optimizer.step() # update parameters
    return model, np.mean(train_batch_loss)

# train
def train(model, criterion, optimizer, scheduler, early_stopping, num_epoch:int=100, ):
    """ train ssl model """
    # settings
    start = time.time() # for time stamp
    # prep data
    train_loader=prepare_data(batch_size=args.batch_size)
    train_loss=list()
    for epoch in range(num_epoch):
        # train
        model, train_epoch_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
        scheduler.step(epoch)
        train_loss.append(train_epoch_loss)
        LOGGER.logger.info(
            f'Epoch: {epoch + 1}, train_loss: {train_epoch_loss:.4f}'
            )
        LOGGER.logger.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))
        # save model
        state = {
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "criterion":criterion,
            "early_stopping":early_stopping,
            "train_loss":train_loss
        }
        torch.save(state, f'{DIR_NAME}/state.pt')
        LOGGER.save_logger(fileout=file_log)
        # state check
        ## early stopping
        early_stopping(train_epoch_loss, model)
        if early_stopping.early_stop:
            LOGGER.logger.info(f'Early Stopping with Epoch: {epoch}')
            model.load_state_dict(torch.load(early_stopping.path))        
            return model, train_loss, True
    return model, train_loss, True

def main():
    # 1. Self-Supervised Learning
    model, criterion, optimizer, scheduler, early_stopping = prepare_model(
        model_name=args.model_name, patience=args.patience, delta=args.delta, lr=args.lr, num_epoch=args.num_epoch
    )
    model, train_loss, flag_finish = train(
        model, criterion, optimizer, scheduler, early_stopping, num_epoch=args.num_epoch, 
    )        
    if flag_finish:
        sslmodel.plot.plot_progress_train(train_loss, DIR_NAME)
        sslmodel.utils.summarize_model(
            model,
            None,
            DIR_NAME, lst_name=['summary_ssl.txt', 'model_ssl.pt']
        )
        # 2. save results & config
        LOGGER.to_logger(name='argument', obj=args)
        LOGGER.to_logger(name='loss', obj=criterion)
        LOGGER.to_logger(
            name='optimizer', obj=optimizer, skip_keys={'state', 'param_groups'}
        )
        LOGGER.to_logger(name='scheduler', obj=scheduler)
    else:
        LOGGER.logger.info('reached max epoch / train')
        
if __name__ == '__main__':
    filename = os.path.basename(__file__).split('.')[0]
    DIR_NAME = PROJECT_PATH + '/result/' +args.dir_result # for output
    file_log = f'{DIR_NAME}/logger.pkl'
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    now = datetime.datetime.now().strftime('%H%M%S')
    LOGGER = sslmodel.utils.logger_save()
    LOGGER.init_logger(filename, DIR_NAME, now, level_console='debug') 
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
    # Set SSL class
    ssl_class=utils.DICT_SSL[args.ssl_name](DEVICE=DEVICE)
    main()

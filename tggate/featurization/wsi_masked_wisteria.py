# -*- coding: utf-8 -*-
"""
# featurize module

@author: Katsuhisa MORITA
"""
import argparse
import time
import os
import re
import sys
import datetime
import random
from typing import List, Tuple, Union, Sequence

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import skimage

sys.path.append("/work/gd43/a97001/src/SelfSupervisedLearningPathology")
import sslmodel
import sslmodel.sslutils as sslutils
import tggate.utils as utils

# argument
parser = argparse.ArgumentParser(description='CLI inference')
parser.add_argument('--note', type=str, help='feature')
parser.add_argument('--seed', type=int, default=24771)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_pool_patch', type=int, default=None)
parser.add_argument('--num_patch', type=int, default=None)
parser.add_argument('--patch_size', type=int, default=224)
parser.add_argument('--model_name', type=str, default='ResNet18') # architecture name
parser.add_argument('--ssl_name', type=str, default='barlowtwins') # ssl architecture name
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--dir_result', type=str, default='')
parser.add_argument('--result_name', type=str, default='foldx_')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--tggate_all', action='store_true')
parser.add_argument('--eisai', action='store_true')
parser.add_argument('--shionogi', action='store_true')
parser.add_argument('--rat', action='store_true')

args = parser.parse_args()
sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

# DataLoader
class DatasetWSI(torch.utils.data.Dataset):
    def __init__(
        self,
        filein:str="",
        filemask:str="",
        transform=None,
        patch_size=224,
        num_patch=None, random_state=24771,
        ):
        # set transform
        if type(transform)!=list:
            self._transform = [transform]
        else:
            self._transform = transform
        # load data
        self.wsi = skimage.io.imread(filein)
        self.lst_location=pd.read_pickle(filemask)
        if num_patch:
            random.seed(random_state)
            self.lst_location=random.sample(self.lst_location, num_patch)
        self.datanum = len(self.lst_location)
        self.patch_size=patch_size

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        location=self.lst_location[idx]
        out_data=self.wsi[location[0]:location[0]+self.patch_size,location[1]:location[1]+self.patch_size,:]
        out_data = Image.fromarray(out_data).convert("RGB")
        if self._transform:
            for t in self._transform:
                out_data = t(out_data)
        return out_data

def prepare_dataset(filein:str="", filemask:str="", patch_size:int=224, batch_size:int=32, num_patch=None,):
    """
    data preparation
    
    """
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([
        transforms.Resize((224,224), antialias=True),
        transforms.ToTensor(),
        normalize
    ])
    # data
    dataset = DatasetWSI(
        filein=filein,
        filemask=filemask,
        transform=data_transform,
        num_patch=num_patch,
        patch_size=patch_size,
        )
    # to loader
    data_loader = sslmodel.data_handler.prep_dataloader(
        dataset, batch_size=batch_size, 
        shuffle=False,
        drop_last=False)
    return data_loader

def featurize_layer(
    model_name="", ssl_name="", model_path="", pretrained=False,
    lst_filein=list(), lst_filename=list(), lst_filemask=list(), dir_result="",
    num_pool_patch=None, num_patch=None,
    batch_size=128, patch_size=224,
    DEVICE="cpu", ):
    """featurize module"""
    # load model
    model=utils.prepare_model_eval(
        model_name=model_name, 
        ssl_name=ssl_name,
        model_path=model_path,
        pretrained=pretrained, 
        DEVICE=DEVICE
    )
    extract_class = utils.DICT_MODEL[model_name][2](DEVICE=DEVICE)
    # result dir
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    # featurize
    for filein, filename, filemask in tqdm(zip(lst_filein, lst_filename, lst_filemask)):
        data_loader=prepare_dataset(filein=filein, filemask=filemask, batch_size=batch_size, patch_size=patch_size, num_patch=num_patch)
        extract_class.featurize(model, data_loader)
        if num_pool_patch:
            extract_class.pooling(num_pool_patch=num_pool_patch)
            extract_class.save_outpool(folder=dir_result, name=filename)
        else:
            extract_class.save_outall(folder=dir_result, name=filename)
        
def main():
    # settings
    start = time.time() # for time stamp
    print(f"start: {start}")

    # 1. load file locations and names
    if args.tggate_all: 
        df_info=pd.read_csv("/work/gd43/a97001/experiments_liver/tggate_info_ext.csv")
        lst_filein=df_info["DIR_wisteria"].tolist()
        lst_filename=[f"{args.result_name}{i}" for i in list(range(df_info.shape[0]))]
        lst_filemask=[f"/work/gd43/share/tggates/liver/mask/{args.patch_size}/{i}_location.pickle" for i in list(range(df_info.shape[0]))]
            
    if args.resume:
        lst_fileout=[f"{args.dir_result}/{i}_layer5.npy" for i in lst_filename]
        lst_tf=[not os.path.isfile(i) for i in lst_fileout]
        lst_filein=[i for i, v in zip(lst_filein, lst_tf) if v]
        lst_filename=[i for i, v in zip(lst_filename, lst_tf) if v]
        lst_filemask=[i for i, v in zip(lst_filemask, lst_tf) if v]
    print(len(lst_filein))

    # 2. inference & save results
    featurize_layer(
        model_name=args.model_name, ssl_name=args.ssl_name, 
        model_path=args.model_path, pretrained=args.pretrained,
        lst_filein=lst_filein, lst_filename=lst_filename, lst_filemask=lst_filemask,
        dir_result=args.dir_result,
        num_pool_patch=args.num_pool_patch, num_patch=args.num_patch,
        batch_size=args.batch_size, patch_size=args.patch_size,
        DEVICE=DEVICE)
    print('elapsed_time: {:.2f} min'.format((time.time() - start)/60))        

if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
    main()
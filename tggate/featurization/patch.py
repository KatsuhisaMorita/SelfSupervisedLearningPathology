# -*- coding: utf-8 -*-
"""
# featurize / pooling with num_pool_patch

@author: Katsuhisa MORITA
"""
# path setting
PROJECT_PATH = '/workspace/pathology'

# packages installed in the current environment
import sys
import datetime
import argparse
import time
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import ImageOps, Image
import cv2
from openslide import OpenSlide

# original packages in src
sys.path.append(f"{PROJECT_PATH}/src/SelfSupervisedLearningPathology")
import settings
import sslmodel
import tggate.utils as utils

# argument
parser = argparse.ArgumentParser(description='CLI inference')
parser.add_argument('--note', type=str, help='feature')
parser.add_argument('--seed', type=int, default=24771)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_pool_patch', type=int, default=256)
parser.add_argument('--model_name', type=str, default='ResNet18') # architecture name
parser.add_argument('--ssl_name', type=str, default='barlowtwins') # ssl architecture name
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--result_name', type=str, default='')
parser.add_argument('--folder_name', type=str, default='')
parser.add_argument('--pretrained', action='store_true')

parser.add_argument('--tggate', action='store_true')
parser.add_argument('--tggate_all', action='store_true')
parser.add_argument('--eisai', action='store_true')
parser.add_argument('--shionogi', action='store_true')
parser.add_argument('--rat', action='store_true')

args = parser.parse_args()
sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

# DataLoader
class Dataset_Patch(torch.utils.data.Dataset):
    """ load for each version """
    def __init__(self,
                filein:str="",
                transform=None,
                ):
        # set transform
        if type(transform)!=list:
            self._transform = [transform]
        else:
            self._transform = transform
        # set
        self.data=np.load(filein)
        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = self.data[idx]
        out_data = Image.fromarray(out_data).convert("RGB")
        if self._transform:
            for t in self._transform:
                out_data = t(out_data)
        return out_data

def prepare_dataset_patch(filein:str="", batch_size:int=32, ):
    """
    data preparation
    
    """
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    # data
    dataset = Dataset_Patch(
        filein=filein,
        transform=data_transform,
        )
    # to loader
    data_loader = sslmodel.data_handler.prep_dataloader(
        dataset, batch_size, 
        shuffle=False,
        drop_last=False)
    return data_loader

def featurize_layer(
    model, model_name="", ssl_name="",
    batch_size=128, lst_filein=list(), 
    folder_name="", result_name="", 
    DEVICE="cpu", num_pool_patch=200,):
    """main module"""
    extract_class = utils.DICT_MODEL[model_name][2](DEVICE=DEVICE)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # featurize
    for filein in lst_filein:
        data_loader=prepare_dataset_patch(filein=filein, batch_size=batch_size)
        extract_class.featurize(model, data_loader)
        extract_class.pooling(num_pool_patch=num_pool_patch)
    extract_class.save_outpool(folder=folder_name, name=result_name)

def main():
    # settings
    start = time.time() # for time stamp
    print(f"start: {start}")
    # 1. model construction
    model = utils.prepare_model_eval(
        model_name=args.model_name, 
        ssl_name=args.ssl_name, 
        model_path=args.model_path,
        pretrained=args.pretrained,
        DEVICE=DEVICE
        )
    ## file names
    if args.tggate_all:
        lst_filein=[f"/workspace/HDD3/TGGATEs/batch_all/batch_{i}.npy" for i in range(64)]
    if args.tggate:
        df_info=pd.read_csv(settings.file_tggate)
        lst_filein=df_info["DIR_PATCH"].tolist()   
    if args.eisai:
        df_info=pd.read_csv(settings.file_eisai)
        df_info=df_info.sort_values(by=["INDEX"])
        lst_filein=df_info["DIR_PATCH"].tolist()        
    if args.shionogi:
        df_info=pd.read_csv(settings.file_shionogi)
        df_info=df_info.sort_values(by=["INDEX"])
        lst_filein=df_info["DIR_PATCH"].tolist()        
    if args.rat:
        df_info=pd.read_csv(settings.file_our)
        df_info=df_info.sort_values(by=["INDEX"])
        lst_filein=df_info["DIR_PATCH"].tolist()        

    # 2. inference & save results
    featurize_layer(
        model, model_name=args.model_name,
        batch_size=args.batch_size, lst_filein=lst_filein,
        folder_name=args.folder_name, result_name=args.result_name,
        DEVICE=DEVICE, num_pool_patch=args.num_pool_patch)
    print('elapsed_time: {:.2f} min'.format((time.time() - start)/60))        

if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
    main()
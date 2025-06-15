# -*- coding: utf-8 -*-
"""
# featurize module

@author: Katsuhisa MORITA
"""
import time
import os
import re
import sys
import datetime
import random
from typing import List, Tuple, Union, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import ImageOps, Image
import cv2
try:
    from openslide import OpenSlide
except:
    print("openslide is not available")

import sslmodel
import sslmodel.sslutils as sslutils

# Featurize Class
class Featurize:
    def __init__(self, DEVICE="cpu", lst_size=[], ):
        self.DEVICE=DEVICE
        self.lst_size=lst_size
        self.out_all=[[] for size in self.lst_size]
        self.out_all_pool=[[] for size in self.lst_size]

    def extraction():
        return None

    def featurize(self, model, data_loader, ):
        # featurize
        with torch.inference_mode():
            for data in data_loader:
                data = data.to(self.DEVICE)
                outs = self.extraction(model, data)
                for i, out in enumerate(outs):
                    self.out_all[i].append(out)

    def pooling(self, num_pool_patch:int=200):
        """max pooling"""
        for i, out in enumerate(self.out_all):
            self.out_all_pool[i].append(
                np.max(np.concatenate(out).reshape(-1, num_pool_patch, self.lst_size[i]),axis=1)
                )
        # reset output list
        self.out_all=[[] for size in self.lst_size]

    def save_outall(self, folder="", name=""):
        for i, out in enumerate(self.out_all):
            out=np.concatenate(out).astype(np.float32)
            np.save(f"{folder}/{name}_layer{i+1}.npy", out)
        self.out_all=[[] for size in self.lst_size]

    def save_outpool(self, folder="", name=""):
        for i, out in enumerate(self.out_all_pool):
            out=np.concatenate(out).astype(np.float32)
            np.save(f"{folder}/{name}_layer{i+1}.npy", out)

class ResNet18Featurize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[64,64,128,256,512],
            )
    def extraction(self, model, x):
        x = model[0](x)# conv1
        x = model[1](x)# bn
        x = model[2](x)# relu
        x = model[3](x)# maxpool
        x1 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,64)
        x = model[4](x)# layer1
        x2 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,64)
        x = model[5](x)# layer2
        x3 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,128)
        x = model[6](x)# layer3
        x4 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,256)
        x = model[7](x)# layer4
        x5 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,512)
        return x1, x2, x3, x4, x5

class DenseNet121Featurize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[64,128,256,512,1024],
            )
    def extraction(self, model, x):
        x = model[0][0](x)
        x = model[0][1](x)
        x = model[0][2](x)
        x = model[0][3](x)
        x1 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,64)
        x = model[0][4](x)
        x = model[0][5](x)
        x2 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,128)
        x = model[0][6](x)
        x = model[0][7](x)
        x3 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,256)
        x = model[0][8](x)
        x = model[0][9](x)
        x4 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,512)
        x = model[0][10](x)
        x = model[0][11](x)
        x5 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,1024)
        return x1, x2, x3, x4, x5

class EfficientNetB3Featurize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[24,32,48,136,1536],
            )
    def extraction(self, model, x):
        x = model[0][0](x)
        x = model[0][1](x)
        x1 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,24)
        x = model[0][2](x)
        x2 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,32)
        x = model[0][3](x)
        x3 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,48)
        x = model[0][4](x)
        x = model[0][5](x)
        x4 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,136)
        x = model[0][6](x)
        x = model[0][7](x)
        x = model[0][8](x)
        x5 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,1536)
        return x1, x2, x3, x4, x5

class ConvNextTinyFeaturize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[96,192,384,768],
            )
    def extraction(self, model, x):
        x = model[0][0](x)
        x = model[0][1](x)
        x1 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,96)
        x = model[0][2](x)
        x = model[0][3](x)
        x2 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,192)
        x = model[0][4](x)
        x = model[0][5](x)
        x3 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,384)
        x = model[0][6](x)
        x = model[0][7](x)
        x4 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,768)
        return x1, x2, x3, x4

class ConvNextTinyFeaturize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[96,192,384,768],
            )
    def extraction(self, model, x):
        x = model[0][0](x)
        x = model[0][1](x)
        x1 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,96)
        x = model[0][2](x)
        x = model[0][3](x)
        x2 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,192)
        x = model[0][4](x)
        x = model[0][5](x)
        x3 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,384)
        x = model[0][6](x)
        x = model[0][7](x)
        x4 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,768)
        return x1, x2, x3, x4

class RegNetY16gfFeaturize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[32,48,120,336,888],
            )
    def extraction(self, model, x):
        x = model[0](x)
        x1 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,32)
        x = model[1][0](x)
        x2 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,48)
        x = model[1][1](x)
        x3 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,120)
        x = model[1][2](x)
        x4 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,336)
        x = model[1][3](x)
        x5 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,888)
        return x1, x2, x3, x4, x5

## Featurize Methods
# name: [Model_Class, last_layer_size, Featurize_Class]
DICT_MODEL = {
    "EfficientNetB3": [torchvision.models.efficientnet_b3, 1536, EfficientNetB3Featurize],
    "ConvNextTiny": [torchvision.models.convnext_tiny, 768, ConvNextTinyFeaturize],
    "ResNet18": [torchvision.models.resnet18, 512, ResNet18Featurize],
    "RegNetY16gf": [torchvision.models.regnet_y_1_6gf, 888, RegNetY16gfFeaturize],
    "DenseNet121": [torchvision.models.densenet121, 1024, DenseNet121Featurize],
}
## Model architecture
DICT_SSL={
    "barlowtwins":sslutils.BarlowTwins,
    "swav":sslutils.SwaV,
    "byol":sslutils.Byol,
    "simsiam":sslutils.SimSiam,
    "wsl":sslutils.WSL,
}

def _load_state_dict_dense(model, weights):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls (pretrained-models). This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    for key in list(weights.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            weights[new_key] = weights[key]
            del weights[key]
    model.load_state_dict(weights)
    return model

def prepare_model_eval(
        model_name:str='ResNet18', ssl_name="barlowtwins",  
        model_path="", 
        pretrained=False, 
        DEVICE="cpu"):
    """
    preparation of models
    Parameters
    ----------
        modelname (str)
            model architecture name

    """
    # model building with indicated name
    if pretrained:
        if model_name=="DenseNet121":
            encoder = DICT_MODEL[model_name][0](weights=None)
            encoder = _load_state_dict_dense(encoder, torch.load(model_path))
            model = nn.Sequential(
                *list(encoder.children())[:-1],
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
                )
        else:
            encoder = DICT_MODEL[model_name][0](weights=None)
            encoder.load_state_dict(torch.load(model_path))
            model=nn.Sequential(*list(encoder.children())[:-1])
    else:
        encoder = DICT_MODEL[model_name][0](weights=None)
        size = DICT_MODEL[model_name][1]
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
        model = DICT_SSL[ssl_name](DEVICE=DEVICE).prepare_featurize_model(
            backbone, model_path=model_path,
            head_size=size,
        )
    model.to(DEVICE)
    model.eval()
    return model

def prepare_model_train(
        model_name:str='ResNet18', ssl_name="barlowtwins",
        patience:int=7, delta:float=0, 
        lr:float=0.003, num_epoch:int=150,
        DEVICE="cpu", DIR_NAME=""):
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
        encoder = DICT_MODEL[model_name][0](weights=None)
        size=DICT_MODEL[model_name][1]
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
    model, criterion = DICT_SSL[ssl_name](DEVICE=DEVICE).prepare_model(backbone, head_size=size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=0
        )
    early_stopping = sslmodel.utils.EarlyStopping(patience=patience, delta=delta, path=f'{DIR_NAME}/checkpoint.pt')
    return model, criterion, optimizer, scheduler, early_stopping

# prepare model
def load_model_train(
        model_name:str='ResNet18', ssl_name="barlowtwins",
        patience:int=7, delta:float=0, 
        lr:float=0.003, num_epoch:int=150,
        DIR_NAME="",
        DEVICE="cpu"):
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
    # load
    state = torch.load(f'{DIR_NAME}/state.pt')
    # model building with indicated name
    try:
        encoder = DICT_MODEL[model_name][0](weights=None)
        size=DICT_MODEL[model_name][1]
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
    model, criterion = DICT_SSL[ssl_name](DEVICE=DEVICE).prepare_model(backbone, head_size=size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=0
        )
    epoch=state['epoch']
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])
    criterion = state['criterion']
    early_stopping = state['early_stopping']
    train_loss=state['train_loss']
    return model, criterion, optimizer, scheduler, early_stopping, train_loss, epoch

# patch
def make_patch(filein:str="", patch_size:int=256, num_patch:int=512, random_state:int=24771, inside=True,):
    """extract patch from WSI"""
    # set seed
    random.seed(random_state)
    # load
    wsi = OpenSlide(filein)
    # get patch mask
    if inside:
        mask=get_mask_inside(wsi, patch_size=patch_size, )
    else:
        mask = get_patch_mask(wsi, patch_size=patch_size)
    mask_shape=mask.shape
    # extract / append
    lst_number=np.array(range(len(mask.flatten())))[mask.flatten()]
    lst_number=random.sample(list(lst_number), num_patch)
    res = []
    ap = res.append
    for number in lst_number:
        v_h, v_w = divmod(number, mask_shape[1])
        patch_image=wsi.read_region(
            location=(int(v_w*patch_size), int(v_h*patch_size)),
            level=0,
            size=(patch_size, patch_size))
        ap(np.array(patch_image, np.uint8)[:,:,:3])
    res=np.stack(res).astype(np.uint8)
    return res, lst_number

def get_patch_mask(image, patch_size, threshold=None,):
    """
    Parameters
    ----------
    iamge: openslide.OpenSlide
    patch_size: int
    threshold: float or None (OTSU)
    Returns
    -------
    mask: np.array(int)[wsi_height, wsi_width]
    """
    level = image.get_best_level_for_downsample(patch_size)
    downsample = image.level_downsamples[level]
    ratio = patch_size / downsample
    whole = image.read_region(location=(0,0), level=level,
        size = image.level_dimensions[level]).convert('HSV')
    whole = whole.resize((int(whole.width / ratio), int(whole.height / ratio)))
    whole = np.array(whole, dtype=np.uint8)
    saturation = whole[:,:,1]
    if threshold is None:
        threshold, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_OTSU)
    mask = saturation > threshold
    return mask

def get_mask_inside(
    image, 
    patch_size:int=224,
    slice_min_patch:int=1000,
    adjacent_cells_slice=[(-1,0),(1,0),(0,-1),(0,1),],
    adjacent_cells_inside=[(-1,0),(1,0),(0,-1),(0,1),(1,1),(1,-1),(-1,1),(-1,-1)],
    ):
    # load
    level = image.get_best_level_for_downsample(patch_size)
    downsample = image.level_downsamples[level]
    ratio = patch_size / downsample
    whole = image.read_region(location=(0,0), level=level,
        size = image.level_dimensions[level]).convert('HSV')
    whole = whole.resize((int(whole.width / ratio), int(whole.height / ratio)))
    whole = np.array(whole, dtype=np.uint8)
    # get mask
    saturation = whole[:,:,1] #saturation
    threshold, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_OTSU)
    mask = saturation > threshold
    # get slices > slice_min_patch
    left_mask = np.full((mask.shape[0]+1, mask.shape[1]+1), fill_value=False, dtype=bool)
    left_mask[:-1, :-1] = mask
    n_left_mask = np.sum(left_mask)
    slice_idx = np.full_like(mask, fill_value=-1, dtype=int)
    i_slice = 0
    while n_left_mask > 0:
        mask_i = np.where(left_mask)
        slice_left_indices = [(mask_i[0][0], mask_i[1][0])]
        while len(slice_left_indices) > 0:
            i, j = slice_left_indices.pop()
            if left_mask[i, j]:
                slice_idx[i, j] = i_slice
                for adj_i, adj_j in adjacent_cells_slice:
                    if left_mask[i+adj_i, j+adj_j]:
                        slice_left_indices.append((i+adj_i, j+adj_j))
                left_mask[i, j] = False
                n_left_mask -= 1
        slice_mask = slice_idx == i_slice
        if np.sum(slice_mask) < slice_min_patch:
            slice_idx[slice_mask] = -1
        else:
            i_slice += 1
    # re-difinition mask
    mask=slice_idx>-1
    # get inside mask
    mask_inside=np.zeros(mask.shape, dtype=bool)
    for i in range(1,mask.shape[0]-1):
        for v in range(1,mask.shape[1]-1):
            tf = mask[i][v]
            if tf:
                for adj in adjacent_cells_inside:
                    tf&=mask[i+adj[0]][v+adj[1]]
            mask_inside[i][v]=tf
    return mask_inside

def sampling_patch_from_wsi(patch_number:int=200, all_number:int=2000, len_df:int=0, seed:int=None):
    if seed is not None:
        random.seed(seed)
    random_lst = list(range(all_number))
    return [random.sample(random_lst, patch_number) for i in range(len_df)]

# others
def check_file(str_file):
    """check input file can be opened or can't"""
    try:
        temp = OpenSlide(str_file)
        print("OK")
    except:
        print("can't open")

def make_groupkfold(group_col, n_splits:int=5):
    temp_arr = np.zeros((len(group_col),1))
    kfold = GroupKFold(n_splits = n_splits).split(temp_arr, groups=group_col)
    kfold_arr = np.zeros((len(group_col),1), dtype=int)
    for n, (tr_ind, val_ind) in enumerate(kfold):
        kfold_arr[val_ind]=int(n)
    return kfold_arr

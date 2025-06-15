# -*- coding: utf-8 -*-
"""
# featurize module 
only ResNet18, BarlowTwins models 
for training classification models

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

sys.path.append("/workspace/tggate/src/SelfSupervisedLearningPathology")
import sslmodel
import sslmodel.sslutils as sslutils
import tggate.utils as utils

from analyzer.analyzer import Analyzer, prepare_dataset_location

# argument
parser = argparse.ArgumentParser(description='CLI inference')
parser.add_argument('--note', type=str, help='feature')
parser.add_argument('--seed', type=int, default=24771)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--patch_size1', type=int, default=224)
parser.add_argument('--patch_size2', type=int, default=448)
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--dir_result', type=str, default='')
parser.add_argument('--result_name', type=str, default='foldx_')

args = parser.parse_args()
sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

def main():
    # settings
    start = time.time() # for time stamp
    print(f"start: {start}")

    # 1. load file locations and names
    df_info=pd.read_csv(f"/workspace/tggate/data/tggate_info_ext.csv")
    lst_filein=df_info["DIR_temp"].tolist()
    lst_filename=[f"{args.result_name}{i}" for i in list(range(df_info.shape[0]))]

    # 2. inference & save results
    dat=Analyzer(DEVICE=DEVICE)
    dat.FindingClassifier.load_featurize_model(args.model_path)
    x_all=[]
    for filein in lst_filein:
        mask = dat.ImageProcessor.get_mask_inside(
            filein=filein, 
            patch_size=args.patch_size2,
            slice_min_patch=100,
        )
        locations=[
            dat.ImageProcessor.get_locations(
                mask=mask, 
                patch_size=args.patch_size2,
                model_patch_size=args.patch_size1,
            ),# small size
            dat.ImageProcessor.get_locations(
                mask=mask, 
                patch_size=args.patch_size2,
                model_patch_size=args.patch_size2,
            ),# large size
        ]
        data_loaders=[
            prepare_dataset_location(
                filein=filein,
                locations=locations[0], 
                batch_size=args.batch_size,
                patch_size=args.patch_size1,
            ),
            prepare_dataset_location(
                filein=filein,
                locations=locations[1], 
                batch_size=args.batch_size,
                patch_size=args.patch_size2
            ),            
        ]
        x =dat.FindingClassifier._featurize(
            data_loaders, num_pool=int(args.patch_size1/args.patch_size2)**2
        )
        x=np.max(x, axis=0) # (,1536)
        x_all.append(x.reshape(1,1536))
    x_all=np.concatenate(x_all, axis=0)
    np.save(f"{args.dir_result}/{args.result_name}_layer45.npy", x_all)

if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
    main()
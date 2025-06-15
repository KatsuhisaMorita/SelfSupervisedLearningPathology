# -*- coding: utf-8 -*-
"""
# create mask (inside patches in WSIs) with 4 diff sizes

@author: Katsuhisa MORITA
"""

root = "/workspace/tggate"

import argparse
import time
import os
import re
import sys
import datetime

from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from openslide import OpenSlide

sys.path.append(f"{root}/src/SelfSupervisedLearningPathology")
from tggate.utils import get_mask_inside

def save_mask(filein, fileout, patch_size:int=None, slice_min_patch:int=None,):
    # mask
    wsi = OpenSlide(filein)
    mask_inside=get_mask_inside(wsi, patch_size=patch_size, slice_min_patch=slice_min_patch)
    lst_number=np.array(range(len(mask_inside.flatten())))[mask_inside.flatten()]
    if len(lst_number)==0:
        print(fileout)
    lst_location=[[int(patch_size*v) for v in divmod(number, mask_inside.shape[1])] for number in lst_number]
    pd.to_pickle(lst_location, fileout)

if __name__=="__main__":
    # settings
    lst_patch_size=[224, 448, ]
    lst_min=[1000, 100, ]
    folder_out="/workspace/HDD3/TGGATEs/mask" # change to your dir
    # load
    df_info=pd.read_csv(f"{root}/data/tggate_info_ext.csv")
    lst_filein=df_info["DIR_temp"].tolist()
    for i in range(2):
        lst_fileout=[f"{folder_out}/{lst_patch_size[i]}/{v}_location.pickle" for v in list(range(df_info.shape[0]))]
        # masked patch
        for filein, fileout in tqdm(zip(lst_filein, lst_fileout)):
            if not os.path.isfile(fileout):
                save_mask(
                    filein, fileout,
                    patch_size=lst_patch_size[i],
                    slice_min_patch=lst_min[i],
                    )
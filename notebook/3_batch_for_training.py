"""
preparation of batch for ssl 
"""

root = "/workspace/tggate"

# import
import gc
import os
import sys
import copy
import random
from tqdm import tqdm
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openslide import OpenSlide
import cv2

# import
sys.path.append(f"{root}/src/SelfSupervisedLearningPathology")
from tggate.utils import make_patch

if __name__=="__main__":
    # parameters
    patch_size=256
    num_patch=256
    seed=24771
    filein_info=f"{root}/data/tggate_info_ext.csv"
    folder_out="/workspace/HDD2/TGGATEs/WSI/Liver/batchext"
    # load
    df_all=pd.read_csv(filein_info)
    # set seed
    random.seed(seed)
    df_all=df_all.sample(frac=1, random_state=seed)
    df_all=df_all.sort_values(by=["FOLD"])
    # index
    pd.to_pickle(df_all["INDEX"].tolist(), f"{folder_out}/index.pickle")
    # batch
    for fold in range(5):
        df_temp = df_all[df_all["FOLD"]==fold]
        lst_dir=df_temp["DIR"].tolist()
        v=0
        arr_res=[]
        ap=arr_res.append
        for i in tqdm(range(df_temp.shape[0])):
            res, _ = make_patch(
                filein=lst_dir[i], patch_size=patch_size, patch_number=num_patch, seed=seed
            )
            ap(res)
            if ((i+1)%256==0) and (i!=0):
                arr_res=np.concatenate(arr_res).astype(np.uint8)
                np.save(f"{folder_out}/fold{fold}_{v}.npy", arr_res)
                v+=1
                arr_res=[]
                ap=arr_res.append
                gc.collect()
        arr_res=np.concatenate(arr_res).astype(np.uint8)
        np.save(f"{folder_out}/fold{fold}_{v}.npy", arr_res)
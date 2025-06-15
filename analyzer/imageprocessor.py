 -*- coding: utf-8 -*-
"""
# Image processing module

@author: Katsuhisa MORITA
"""

import numpy as np
import pandas as pd

from openslide import OpenSlide
import cv2

class ImageProcessor:
    def __init__(self):
        return

    def load_patch(self, filein="", mask=None, patch_size=448, model_patch_size=224,):
        """load patch with mask"""
        image = OpenSlide(filein)
        res = []
        lst_number = []
        ap = res.append
        for number in np.where(mask.flatten())[0]:
            v_h, v_w = divmod(number, mask.shape[1])
            lst_number.append(number)
            for i in range(int(patch_size/model_patch_size)):
                for v in range(int(patch_size/model_patch_size)):
                    patch_image=image.read_region(
                        location=(
                            int(v_w*patch_size + i*model_patch_size), 
                            int(v_h*patch_size + v*model_patch_size)
                        ),
                        level=0,
                        size=(model_patch_size, model_patch_size))
                    ap(np.array(patch_image, np.uint8)[:,:,:3])
        res=np.stack(res).astype(np.uint8)
        return res, lst_number

    def get_locations(self, filein="", mask=None, patch_size=448, model_patch_size=224,):
        """return patch locations"""
        res = []
        ap = res.append
        for number in np.array(range(len(mask.flatten())))[mask.flatten()]:
            v_h, v_w = divmod(number, mask.shape[1])
            for i in range(int(patch_size/model_patch_size)):
                for v in range(int(patch_size/model_patch_size)):
                    location=(
                        int(v_w*patch_size + i*model_patch_size), 
                        int(v_h*patch_size + v*model_patch_size)
                    )
                    ap(location)
        return res

    def get_mask_inside(
        self,
        filein="", 
        patch_size:int=448,
        slice_min_patch:int=100,
        adjacent_cells_slice=[(-1,0),(1,0),(0,-1),(0,1),],
        adjacent_cells_inside=[(-1,0),(1,0),(0,-1),(0,1),(1,1),(1,-1),(-1,1),(-1,-1)],
        ):
        """
        image: OpenSlide object
        patch_size: int
        slice_min_patch: minimum size of patch group
        adjacent_cells_slice: definition of connections
        adjacent_cells_inside: definition of not outside condition
        """
        # load
        image = OpenSlide(filein)
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
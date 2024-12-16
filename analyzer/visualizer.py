# -*- coding: utf-8 -*-
"""
# Image Visualizer

@author: Katsuhisa MORITA
"""
import gc
import os
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openslide import OpenSlide
import cv2
import torch

import analyzer

class Visualizer():
    def __init__(self, DEVICE=None):
        if not DEVICE:
            DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.Analyzer=analyzer.Analyzer(DEVICE=DEVICE)
        self.filein=None

    def load_model(
        self, 
        dir_featurize_model="model.pt", 
        dir_classification_models="folder",
        style="dict"
        ):
        self.Analyzer.load_model(
            dir_featurize_model=dir_featurize_model, 
            dir_classification_models=dir_classification_models,
            style=style,    
        )

    def analyze_image(
        self, 
        filein="image", 
        batch_size=256,
        patch_size=448,
        model_patch_size=224,
        slice_min_patch=100,
        save_memory=True,
        ):
        self.Analyzer.analyze(
            filein=filein, 
            batch_size=batch_size,
            patch_size=patch_size,
            model_patch_size=model_patch_size,
            slice_min_patch=slice_min_patch,
        )
        # take results
        self.result_patch, self.result_all = self.Analyzer.result_patch, self.Analyzer.result_all
        self.anomaly_proba = pd.DataFrame(self.result_patch).max(axis=1).values
        self.mask=self.Analyzer.mask
        self.locations=self.Analyzer.locations[1]
        # set parameters
        self.filein=filein
        self.patch_size=patch_size
        if save_memory:
            del self.Analyzer
            gc.collect()

    def load_image(self, scale_factor=4, ):
        if not self.filein:
            print("Analyze before load image")
        else:
            # load image
            self.image=OpenSlide(self.filein)
            level = self.image.get_best_level_for_downsample(self.patch_size/scale_factor)
            downsample = self.image.level_downsamples[level]
            ratio = self.patch_size/ scale_factor / downsample
            # patch = (scale foctor, scale factor)
            image = self.image.read_region(
                location=(0,0), 
                level=level,
                size=self.image.level_dimensions[level]
            )
            self.image_scaled=np.array(
                image.resize((int(image.width / ratio), int(image.height / ratio))),
                np.uint8,
            )[:,:,:3] # RGB
            # scaled locations
            self.locations_scaled=[(
                int(i[0]/(self.patch_size/scale_factor)), 
                int(i[1]/(self.patch_size/scale_factor))
            ) for i in self.locations]
            # set
            self.scale_factor=scale_factor

    def plot_rawimage(self, savedir="", dpi=80):
        plt.imshow(self.image_scaled)
        plt.grid(False)
        if savedir:
            plt.savefig(f"{savedir}/wsi_rawimage.png", dpi=dpi)
        plt.show()

    def plot_anomaly(self, savedir="", dpi=80):
        # probability distributions
        _=sns.displot(
            data=self.anomaly_proba, 
            bins=30,
            kde=True,
            height=4,
            aspect=1.5,
        )
        plt.xlabel("Anomaly Probabilites")
        plt.ylabel("Crop Counts")
        plt.xlim([-.03,1.03])
        if savedir:
            plt.savefig(f"{savedir}/proba_dist.png", dpi=dpi)
        plt.show()

        # processing
        df_proba=pd.DataFrame({
            "proba":self.anomaly_proba, 
            "locate":self.locations_scaled
        })
        # plot
        if len(df_proba[df_proba["proba"]>0.5]["locate"].tolist())==0:
            print(f"No high probability crops")
        else:
            plt.imshow(self.image_scaled)
            # middle probabilities (0.5 ~ 0.8) square
            for locate in df_proba[(0.8>df_proba["proba"])&(df_proba["proba"]>0.5)]["locate"].tolist():
                self._plot_cropline(
                    locate, 
                    color="yellow", 
                    linewidth=0.8, 
                    scale_factor=self.scale_factor
                )
            # high probabilities (0.8 ~) square
            for locate in df_proba[df_proba["proba"]>0.8]["locate"].tolist():
                self._plot_cropline(
                    locate, 
                    color="red", 
                    linewidth=0.8, 
                    scale_factor=self.scale_factor
                )

            plt.grid(False)
            if savedir:
                plt.savefig(f"{savedir}/wsi_anomaly.png", dpi=dpi)
            plt.show()

    def plot_anomaly_patch(self, savedir="", dpi=80):
        # processing
        df_proba=pd.DataFrame({
            "proba":self.anomaly_proba, 
            "locate":self.locations, 
        }).sort_values(by="proba", ascending=False)

        fig=plt.figure(figsize=(10,10))
        for i in range(16):
            ax=fig.add_subplot(4,4,i+1)
            patch=self.image.read_region(
            location=df_proba.iloc[i,1], 
            level=0,
            size = (448,448)
            )
            ax.imshow(patch)
            ax.set_title(f"proba: {df_proba.iloc[i,0]:.3f}")
            ax.set_xticklabels("")
            ax.set_yticklabels("")
            plt.grid(False)
        fig.suptitle("High Anomaly Probability Crops")
        if savedir:
            plt.savefig(f"{savedir}/crops_anomaly.png", dpi=dpi)
        plt.show()

    def plot_findings(self, savedir="", dpi=80):
        # processing
        df_proba=pd.DataFrame(self.result_patch)
        df_proba["locate"]=self.locations_scaled
        # plot
        for key in self.result_patch.keys():
            if len(df_proba[df_proba[key]>0.5]["locate"].tolist())==0:
                print(f"No high probability crops: {key}")
            else:
                plt.imshow(self.image_scaled)
                for locate in df_proba[(0.8>df_proba[key])&(df_proba[key]>0.5)]["locate"].tolist():
                    self._plot_cropline(locate, color="yellow", linewidth=0.8, scale_factor=self.scale_factor)

                for locate in df_proba[df_proba[key]>0.8]["locate"].tolist():
                    self._plot_cropline(locate, color="red", linewidth=0.8, scale_factor=self.scale_factor)
                
                plt.grid(False)
                plt.title(key)
                if savedir:
                    plt.savefig(f"{savedir}/wsi_{key}.png", dpi=dpi)
                plt.show()

    def plot_findings_patch(self, only_highscore=True, savedir="", dpi=80):
        # processing
        df_proba=pd.DataFrame(self.result_patch)
        df_proba["locate"]=self.locations
        if only_highscore:
            for key in self.result_patch.keys():
                if self.result_all[key]>0.5:
                    df_proba=df_proba.sort_values(by=key, ascending=False)
                    fig=plt.figure(figsize=(10,10))
                    for i in range(16):
                        ax=fig.add_subplot(4,4,i+1)
                        patch=self.image.read_region(
                        location=df_proba["locate"].iloc[i], 
                        level=0,
                        size = (448,448)
                        )
                        ax.imshow(patch)
                        ax.set_title(f"proba: {df_proba[key].iloc[i]:.3f}")
                        ax.set_xticklabels("")
                        ax.set_yticklabels("")
                        plt.grid(False)

                    fig.suptitle(f"High {key} Probability Crops")
                    if savedir:
                        plt.savefig(f"{savedir}/crops_{key}.png", dpi=dpi)

                    plt.show()
        
    def print_probabilities(self,):
        print("Findings Probabilities: ")
        for key, item in self.result_all.items():
            print(f"{key}: {item[0]:.3f}")

    def export_probabilities(self, savedir=""):
        with open(f"{savedir}/probs.txt", "w") as file:
            for key, item in self.result_all.items():
                file.write(f"{key}: {item[0]:.5f}\n")

    def _plot_cropline(
        self,
        locate, 
        color="red", 
        linewidth=1, 
        scale_factor=8
        ):
        plt.plot([
            locate[0], 
            locate[0]+scale_factor, 
            locate[0]+scale_factor,
            locate[0],
            locate[0],
            ],
            [
            locate[1], 
            locate[1], 
            locate[1]+scale_factor,
            locate[1]+scale_factor,
            locate[1],
            ],
            linestyle="-",
            color=color,
            linewidth=linewidth,
        )

if __name__=="__main__":
    # parameters
    parser = argparse.ArgumentParser(description='WSI Analyze & Visualize')
    # file dirs
    parser.add_argument('--filein', type=str, default='wsi.svs')
    parser.add_argument('--dir_featurize_model', type=str, default='model.pt')
    parser.add_argument('--dir_classification_models', type=str, default='model.pickle')
    parser.add_argument('--savedir', type=str, default='')
    # analysis settings
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=448)
    parser.add_argument('--model_patch_size', type=int, default=224)
    parser.add_argument('--slice_min_patch', type=int, default=100)
    parser.add_argument('--scale_factor', type=int, default=8)
    # plot settings
    parser.add_argument('--dpi', type=int, default=80)
    parser.add_argument('--rawimage', action='store_true')
    parser.add_argument('--anomaly', action='store_true')
    parser.add_argument('--anomaly_crops', action='store_true')
    parser.add_argument('--findings', action='store_true')
    parser.add_argument('--findings_crops', action='store_true')
    parser.add_argument('--only_highscore', action='store_true')
    args = parser.parse_args()
    # Analyze
    sns.set()
    dat=Visualizer()
    dat.load_model(
        dir_featurize_model=args.dir_featurize_model, 
        dir_classification_models=args.dir_classification_models,
        style="dict"
    )
    dat.analyze_image(
        args.filein, 
        batch_size=args.batch_size, 
        patch_size=args.patch_size, 
        model_patch_size=args.model_patch_size,
        slice_min_patch=args.slice_min_patch,
        save_memory=True,
    )
    dat.load_image(scale_factor=args.scale_factor)
    # Export results
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
    # probabilities
    try:
        dat.print_probabilities()
        dat.export_probabilities(savedir=args.savedir)
    except:
        pass
    # plot images
    if args.rawimage:
        dat.plot_rawimage(savedir=args.savedir, dpi=args.dpi)
    if args.anomaly:
        dat.plot_anomaly(savedir=args.savedir, dpi=args.dpi)
    if args.anomaly_crops:
        dat.plot_anomaly_patch(savedir=args.savedir, dpi=args.dpi)
    if args.findings:
        dat.plot_findings(savedir=args.savedir, dpi=args.dpi)
    if args.findings_crops:
        dat.plot_findings_patch(
            only_highscore=args.only_highscore, 
            savedir=args.savedir, dpi=args.dpi
        )

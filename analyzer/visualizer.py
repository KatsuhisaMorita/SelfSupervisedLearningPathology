# -*- coding: utf-8 -*-
"""
# Image Visualizer

@author: Katsuhisa MORITA
"""
import gc
import os
import random
import argparse
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openslide import OpenSlide
import cv2
import torch
from torch import nn

import analyzer


class Visualizer:
    """
    A class to visualize the results of WSI analysis.

    Args:
        DEVICE (Optional[torch.device]): Device to use for computation.
    """
    # Constants for thresholds
    PROB_HIGH_THRESHOLD = 0.8
    PROB_MID_THRESHOLD = 0.5
    
    def __init__(self, DEVICE: Optional[torch.device] = None):
        if not DEVICE:
            DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Analyzer = analyzer.Analyzer(DEVICE=DEVICE)
        self.filein: Optional[str] = None
        self.image: Optional[OpenSlide] = None
        self.image_scaled: Optional[np.ndarray] = None
        self.locations: Optional[List[Tuple[int, int]]] = None
        self.locations_scaled: Optional[List[Tuple[int, int]]] = None
        self.result_patch: Optional[Dict[str, np.ndarray]] = None
        self.result_all: Optional[Dict[str, np.ndarray]] = None
        self.anomaly_proba: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.patch_size: Optional[int] = None
        self.scale_factor: Optional[int] = None

    def load_model(
        self,
        dir_featurize_model: str = "model.pt",
        dir_classification_models: str = "folder",
        style: str = "dict",
    ) -> None:
        """Load featurize and classification models."""
        self.Analyzer.load_model(
            dir_featurize_model=dir_featurize_model,
            dir_classification_models=dir_classification_models,
            style=style,
        )

    def analyze_image(
        self,
        filein: str = "image",
        batch_size: int = 256,
        patch_size: int = 448,
        model_patch_size: int = 224,
        slice_min_patch: int = 100,
        save_memory: bool = True,
    ) -> None:
        """
        Analyze a WSI image.

        Args:
            filein (str): Path to the input image file.
            batch_size (int): Batch size for analysis.
            patch_size (int): Size of patches to extract.
            model_patch_size (int): Patch size required by the model.
            slice_min_patch (int): Minimum number of patches in a slice.
            save_memory (bool): If True, delete the analyzer object to save memory.
        """
        self.Analyzer.analyze(
            filein=filein,
            batch_size=batch_size,
            patch_size=patch_size,
            model_patch_size=model_patch_size,
            slice_min_patch=slice_min_patch,
        )
        # take results
        self.result_patch, self.result_all = (
            self.Analyzer.result_patch,
            self.Analyzer.result_all,
        )
        self.anomaly_proba = pd.DataFrame(self.result_patch).max(axis=1).values
        self.mask = self.Analyzer.mask
        self.locations = self.Analyzer.locations[1]
        # set parameters
        self.filein = filein
        self.patch_size = patch_size
        if save_memory:
            del self.Analyzer
            gc.collect()

    def load_image(
        self,
        scale_factor: int = 4,
    ) -> None:
        """
        Load and scale the WSI image.

        Args:
            scale_factor (int): The factor by which to scale down the image.
        """
        if not self.filein or not self.patch_size:
            print("Analyze before load image")
            return
        
        # load image
        self.image = OpenSlide(self.filein)
        level = self.image.get_best_level_for_downsample(
            self.patch_size / scale_factor
        )
        downsample = self.image.level_downsamples[level]
        ratio = self.patch_size / scale_factor / downsample
        # patch = (scale foctor, scale factor)
        image = self.image.read_region(
            location=(0, 0), level=level, size=self.image.level_dimensions[level]
        )
        self.image_scaled = np.array(
            image.resize((int(image.width / ratio), int(image.height / ratio))),
            np.uint8,
        )[
            :, :, :3
        ]  # RGB
        # scaled locations
        if self.locations:
            self.locations_scaled = [
                (
                    int(i[0] / (self.patch_size / scale_factor)),
                    int(i[1] / (self.patch_size / scale_factor)),
                )
                for i in self.locations
            ]
        # set
        self.scale_factor = scale_factor

    def plot_rawimage(self, savedir: str = "", dpi: int = 80) -> None:
        """Plots the raw scaled image."""
        if self.image_scaled is None:
            print("Image not loaded.")
            return
        plt.imshow(self.image_scaled)
        plt.grid(False)
        if savedir:
            plt.savefig(os.path.join(savedir, "wsi_rawimage.png"), dpi=dpi)
        plt.show()

    def _plot_probability_map(self, proba: np.ndarray, title: str, savedir: str = "", dpi: int = 80) -> None:
        """Helper function to plot a probability map on the WSI."""
        if self.image_scaled is None or self.locations_scaled is None:
            print("Image or locations not available.")
            return
        df_proba = pd.DataFrame(
            {"proba": proba, "locate": self.locations_scaled}
        )
        if df_proba[df_proba["proba"] > self.PROB_MID_THRESHOLD].empty:
            print(f"No high probability crops for {title}")
            return

        plt.imshow(self.image_scaled)
        # Middle probabilities
        for locate in df_proba[
            (self.PROB_HIGH_THRESHOLD > df_proba["proba"]) & (df_proba["proba"] > self.PROB_MID_THRESHOLD)
        ]["locate"].tolist():
            self._plot_cropline(locate, color="yellow", linewidth=0.8)
        # High probabilities
        for locate in df_proba[df_proba["proba"] > self.PROB_HIGH_THRESHOLD]["locate"].tolist():
            self._plot_cropline(locate, color="red", linewidth=0.8)

        plt.grid(False)
        plt.title(title)
        if savedir:
            plt.savefig(os.path.join(savedir, f"wsi_{title}.png"), dpi=dpi)
        plt.show()

    def plot_anomaly(self, savedir: str = "", dpi: int = 80) -> None:
        """Plots the anomaly probability distribution and map."""
        if self.anomaly_proba is None:
            print("Anomaly probabilities not available.")
            return
        # probability distributions
        _ = sns.displot(
            data=self.anomaly_proba,
            bins=30,
            kde=True,
            height=4,
            aspect=1.5,
        )
        plt.xlabel("Anomaly Probabilites")
        plt.ylabel("Crop Counts")
        plt.xlim([-0.03, 1.03])
        if savedir:
            plt.savefig(os.path.join(savedir, "proba_dist.png"), dpi=dpi)
        plt.show()

        self._plot_probability_map(self.anomaly_proba, "anomaly", savedir, dpi)

    def plot_anomaly_patch(self, savedir: str = "", dpi: int = 80) -> None:
        """Plots the 16 patches with the highest anomaly probability."""
        if self.anomaly_proba is None or self.locations is None or self.image is None or self.patch_size is None:
            print("Data for plotting anomaly patches is not available.")
            return
        # processing
        df_proba = pd.DataFrame(
            {
                "proba": self.anomaly_proba,
                "locate": self.locations,
            }
        ).sort_values(by="proba", ascending=False)

        fig = plt.figure(figsize=(10, 10))
        for i in range(min(16, len(df_proba))):
            ax = fig.add_subplot(4, 4, i + 1)
            patch = self.image.read_region(
                location=df_proba.iloc[i, 1], level=0, size=(self.patch_size, self.patch_size)
            )
            ax.imshow(patch)
            ax.set_title(f"proba: {df_proba.iloc[i,0]:.3f}")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.grid(False)
        fig.suptitle("High Anomaly Probability Crops")
        if savedir:
            plt.savefig(os.path.join(savedir, "crops_anomaly.png"), dpi=dpi)
        plt.show()

    def plot_findings(self, savedir: str = "", dpi: int = 80) -> None:
        """Plots probability maps for each finding."""
        if self.result_patch is None:
            print("Finding results not available.")
            return
        
        for key, proba in self.result_patch.items():
            self._plot_probability_map(proba, key, savedir, dpi)

    def plot_findings_patch(self, only_highscore: bool = True, savedir: str = "", dpi: int = 80) -> None:
        """Plots the 16 patches with the highest probability for each finding."""
        if self.result_patch is None or self.locations is None or self.image is None or self.patch_size is None:
            print("Data for plotting finding patches is not available.")
            return
        
        df_proba_base = pd.DataFrame(self.result_patch)
        df_proba_base["locate"] = self.locations

        for key in self.result_patch.keys():
            if only_highscore and (self.result_all is None or self.result_all[key] <= self.PROB_MID_THRESHOLD):
                continue

            df_proba = df_proba_base.sort_values(by=key, ascending=False)
            fig = plt.figure(figsize=(10, 10))
            for i in range(min(16, len(df_proba))):
                ax = fig.add_subplot(4, 4, i + 1)
                patch = self.image.read_region(
                    location=df_proba["locate"].iloc[i],
                    level=0,
                    size=(self.patch_size, self.patch_size),
                )
                ax.imshow(patch)
                ax.set_title(f"proba: {df_proba[key].iloc[i]:.3f}")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                plt.grid(False)

            fig.suptitle(f"High {key} Probability Crops")
            if savedir:
                plt.savefig(os.path.join(savedir, f"crops_{key}.png"), dpi=dpi)
            plt.show()

    def print_probabilities(self) -> None:
        """Prints the probabilities of findings."""
        if self.result_all is None:
            print("Probabilities not available.")
            return
        print("Findings Probabilities: ")
        for key, item in self.result_all.items():
            print(f"{key}: {item[0]:.3f}")

    def export_probabilities(self, savedir: str = "") -> None:
        """Exports the probabilities of findings to a text file."""
        if self.result_all is None:
            print("Probabilities not available to export.")
            return
        if not savedir:
            print("Save directory not specified.")
            return
        with open(os.path.join(savedir, "probs.txt"), "w") as file:
            for key, item in self.result_all.items():
                file.write(f"{key}: {item[0]:.5f}\n")

    def _plot_cropline(self, locate: Tuple[int, int], color: str = "red", linewidth: float = 1) -> None:
        """Plots a bounding box for a crop on the scaled image."""
        if self.scale_factor is None:
            return
        plt.plot(
            [
                locate[0],
                locate[0] + self.scale_factor,
                locate[0] + self.scale_factor,
                locate[0],
                locate[0],
            ],
            [
                locate[1],
                locate[1],
                locate[1] + self.scale_factor,
                locate[1] + self.scale_factor,
                locate[1],
            ],
            linestyle="-",
            color=color,
            linewidth=linewidth,
        )


def main():
    """Main function to run the analysis and visualization pipeline."""
    parser = argparse.ArgumentParser(description="WSI Analyze & Visualize")
    # file dirs
    parser.add_argument("--filein", type=str, required=True, help="Path to WSI file.")
    parser.add_argument("--dir_featurize_model", type=str, default="model.pt", help="Path to featurize model.")
    parser.add_argument("--dir_classification_models", type=str, default="model.pickle", help="Path to classification models.")
    parser.add_argument("--savedir", type=str, default="", help="Directory to save results.")
    # analysis settings
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=448)
    parser.add_argument("--model_patch_size", type=int, default=224)
    parser.add_argument("--slice_min_patch", type=int, default=100)
    parser.add_argument("--scale_factor", type=int, default=8)
    # plot settings
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--rawimage", action="store_true", help="Plot raw image.")
    parser.add_argument("--anomaly", action="store_true", help="Plot anomaly map.")
    parser.add_argument("--anomaly_crops", action="store_true", help="Plot high anomaly crops.")
    parser.add_argument("--findings", action="store_true", help="Plot findings maps.")
    parser.add_argument("--findings_crops", action="store_true", help="Plot high probability finding crops.")
    parser.add_argument("--only_highscore", action="store_true", help="Plot only high score findings.")
    args = parser.parse_args()
    
    # Create save directory if specified
    if args.savedir:
        os.makedirs(args.savedir, exist_ok=True)

    # Analyze
    sns.set()
    dat = Visualizer()
    dat.load_model(
        dir_featurize_model=args.dir_featurize_model,
        dir_classification_models=args.dir_classification_models,
        style="dict",
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
    try:
        dat.print_probabilities()
        if args.savedir:
            dat.export_probabilities(savedir=args.savedir)
    except Exception as e:
        print(f"Error exporting probabilities: {e}")

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
            only_highscore=args.only_highscore, savedir=args.savedir, dpi=args.dpi
        )

if __name__ == "__main__":
    main()

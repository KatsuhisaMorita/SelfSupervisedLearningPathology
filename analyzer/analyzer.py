# -*- coding: utf-8 -*-
"""
# Image Analyzer

Provides functionality to analyze Whole Slide Images (WSI) by extracting patches,
featurizing them, and classifying them for pathological findings.

@author: Katsuhisa MORITA
"""
from typing import List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from openslide import OpenSlide, OpenSlideError
from PIL import Image

import torch
import torch.utils.data
import torchvision.transforms as transforms

# Assuming these are custom modules from the same project
from .imageprocessor import ImageProcessor
from .model import FindingClassifier


class Analyzer:
    """
    Orchestrates the WSI analysis pipeline.

    This class manages loading models, processing images to find tissue regions,
    creating data loaders for patches, and running classification to get results.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initializes the Analyzer.

        Args:
            device (str): The device to run the models on ('cpu' or 'cuda').
        """
        self.finding_classifier = FindingClassifier(DEVICE=device)
        self.image_processor = ImageProcessor()
        self.mask: Optional[np.ndarray] = None
        self.locations_small: Optional[List[Tuple[int, int]]] = None
        self.locations_large: Optional[List[Tuple[int, int]]] = None
        self.result_patch: Optional[pd.DataFrame] = None
        self.result_all: Optional[pd.DataFrame] = None

    def load_model(
        self,
        dir_featurize_model: str = "model.pt",
        dir_classification_models: str = "folder/",
        style: str = "dict",
    ):
        """
        Loads the featurization and classification models.

        Note:
            The broad except clauses can hide specific errors. It's recommended
            to catch more specific exceptions (e.g., FileNotFoundError) for
            better error handling and debugging.
        """
        try:
            self.finding_classifier.load_featurize_model(dir_model=dir_featurize_model)
        except Exception as e:
            print(f"Could not load featurize model '{dir_featurize_model}': {e}")
        try:
            self.finding_classifier.load_classification_models(
                dir_models=dir_classification_models, style=style
            )
        except Exception as e:
            print(
                f"Could not load classification models from '{dir_classification_models}': {e}"
            )

    def analyze(
        self,
        filein: str,
        batch_size: int = 256,
        patch_size: int = 448,
        model_patch_size: int = 224,
        slice_min_patch: int = 100,
        num_workers: int = 4,
    ):
        """
        Performs a full analysis of a given WSI file.

        The process involves:
        1. Identifying tissue regions in the WSI.
        2. Generating locations for small and large patches.
        3. Creating PyTorch DataLoaders for these patches.
        4. Running classification to get patch-level and slide-level predictions.

        Args:
            filein (str): Path to the WSI file.
            batch_size (int): Batch size for the DataLoader.
            patch_size (int): The size of the larger patches to extract.
            model_patch_size (int): The size of the smaller patches (input to the model).
            slice_min_patch (int): Minimum number of tissue pixels for a patch to be included.
            num_workers (int): Number of worker processes for the DataLoader.
        """
        # 1. Get organ regions
        self.mask = self.image_processor.get_mask_inside(
            filein=filein, patch_size=patch_size, slice_min_patch=slice_min_patch
        )

        # 2. Get patch locations for two different scales
        self.locations_small = self.image_processor.get_locations(
            mask=self.mask, patch_size=patch_size, model_patch_size=model_patch_size
        )
        self.locations_large = self.image_processor.get_locations(
            mask=self.mask, patch_size=patch_size, model_patch_size=patch_size
        )

        # 3. Set up data loaders
        loader_small = prepare_dataset_location(
            filein=filein,
            locations=self.locations_small,
            batch_size=batch_size,
            patch_size=model_patch_size,
            num_workers=num_workers,
        )
        loader_large = prepare_dataset_location(
            filein=filein,
            locations=self.locations_large,
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
        )
        data_loaders = [loader_small, loader_large]

        # 4. Featurize and classify
        # The number of small patches within a large patch, used for pooling.
        if model_patch_size == 0:
            raise ValueError("model_patch_size cannot be zero.")
        pool_factor = (patch_size // model_patch_size) ** 2
        self.result_patch, self.result_all = self.finding_classifier.classify(
            data_loaders, num_pool=pool_factor
        )


class PatchDatasetLocation(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to load patches from a WSI given their locations.

    This dataset opens the WSI file and reads specific regions (patches) on demand.
    It is designed to work with PyTorch's DataLoader for efficient, parallelized
    data loading. Each worker process will have its own OpenSlide file handle.
    """

    def __init__(
        self,
        filein: str,
        locations: List[Tuple[int, int]],
        transform: Optional[Callable] = None,
        patch_size: int = 224,
    ):
        """
        Initializes the dataset.

        Args:
            filein (str): Path to the WSI file.
            locations (List[Tuple[int, int]]): A list of (x, y) coordinates for the top-left
                                               corner of each patch at level 0.
            transform (Callable, optional): A function/transform to apply to each patch.
            patch_size (int): The size of the square patch to extract.
        """
        self.filein = filein
        self.wsi = OpenSlide(filein)
        self.locations = locations
        self.patch_size = patch_size
        self._transform = transform

    def __len__(self) -> int:
        """Returns the total number of patches."""
        return len(self.locations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a patch by its index.

        It reads the specified region, converts it to RGB, and applies transforms.

        Args:
            idx (int): The index of the patch to retrieve.

        Returns:
            torch.Tensor: The transformed patch as a tensor.
        """
        location = self.locations[idx]
        try:
            # read_region returns a PIL Image
            patch_img = self.wsi.read_region(
                location=location,
                level=0,
                size=(self.patch_size, self.patch_size),
            )
            # Convert to RGB (handles RGBA) and apply transforms
            patch_img = patch_img.convert("RGB")
            if self._transform:
                patch_img = self._transform(patch_img)
            return patch_img
        except OpenSlideError as e:
            print(f"Error reading region at {location} from {self.filein}: {e}")
            # Return a dummy tensor on error, assuming model input size is 224x224
            return torch.zeros((3, 224, 224))


def prepare_dataset_location(
    filein: str,
    locations: List[Tuple[int, int]],
    batch_size: int = 128,
    patch_size: int = 224,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """
    Creates a DataLoader for WSI patches.

    Sets up the necessary transformations (resizing, tensor conversion, normalization)
    and wraps the PatchDatasetLocation in a DataLoader.

    Args:
        filein (str): Path to the WSI file.
        locations (List[Tuple[int, int]]): List of patch coordinates.
        batch_size (int): Number of samples per batch.
        patch_size (int): The size of patches to be extracted.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        torch.utils.data.DataLoader: The configured DataLoader.
    """
    # Standard normalization for ImageNet-pretrained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # Note: The model input size is hardcoded to 224x224 here.
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = PatchDatasetLocation(
        filein=filein,
        locations=locations,
        transform=data_transform,
        patch_size=patch_size,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
    )
    return data_loader


def _worker_init_fn(worker_id: int):
    """
    Ensures that data loading is reproducible across multiple workers.
    It sets a unique random seed for each worker process.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

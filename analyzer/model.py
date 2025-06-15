# -*- coding: utf-8 -*-
"""
# wrapper of finding prediction models
# barlow twins module (for featurize)

reference:
https://arxiv.org/abs/2103.03230
https://docs.lightly.ai/self-supervised-learning/examples/barlowtwins.html

@author: Katsuhisa MORITA
"""
from tqdm import tqdm
import numpy as np
import pandas as pd

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class FindingClassifier:
    def __init__(self, DEVICE="cpu"):
        self.featurize_model = None
        self.classification_models = list()
        self.DEVICE = DEVICE

    def load_featurize_model(self, dir_model=""):
        """only resnet18 x barlowtwins model is allowed"""
        encoder = torchvision.models.resnet18(weights=None)
        model = BarlowTwins(
            nn.Sequential(
                *list(encoder.children())[:-1],
            ),
            head_size=[512, 512, 128],
        )
        model.load_state_dict(torch.load(dir_model))
        model = model.backbone
        self.featurize_model = model

    def load_classification_models(self, dir_models="", style="dict"):
        if style == "dict":
            self.classification_models = pd.read_pickle(dir_models)
        else:
            print("not dict style is not implemented")
        self.style = style

    def classify(self, data_loaders, num_pool=4):
        """predict all class probability"""
        print("Featurizing WSI")
        x = self._featurize(data_loaders, num_pool=num_pool)  # sample x feature
        print("Finding Classsifying")
        result_patch = self._predict_proba(x, style=self.style)
        result_all = self._predict_proba(
            np.max(x, axis=0).reshape(1, 1536), style=self.style
        )
        return result_patch, result_all

    def _featurize(
        self,
        data_loaders,
        num_pool=4,
    ):
        """small size, large size and layer 4, 5"""
        # featurize
        self.featurize_model = self.featurize_model.to(self.DEVICE)
        lst_out = [[] * 4]
        with torch.inference_mode():
            x4_small = []
            x5_small = []
            for data in data_loaders[0]:
                data = data.to(self.DEVICE)
                x4, x5 = self._extraction_layer45(self.featurize_model, data)
                x4_small.append(x4)
                x5_small.append(x5)
            x4_small = np.max(
                np.concatenate(x4_small).reshape(-1, num_pool, 256), axis=1
            )
            x5_small = np.max(
                np.concatenate(x5_small).reshape(-1, num_pool, 512), axis=1
            )
            x4_large = []
            x5_large = []
            for data in data_loaders[1]:
                data = data.to(self.DEVICE)
                x4, x5 = self._extraction_layer45(self.featurize_model, data)
                x4_large.append(x4)
                x5_large.append(x5)
            x4_large = np.concatenate(x4_large).reshape(-1, 256)
            x5_large = np.concatenate(x5_large).reshape(-1, 512)
        return np.concatenate([x5_small, x5_large, x4_small, x4_large], axis=1)

    def _extraction_layer45(self, model, x):
        x = model[0](x)  # conv1
        x = model[1](x)  # bn
        x = model[2](x)  # relu
        x = model[3](x)  # maxpool
        # x1 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,64)
        x = model[4](x)  # layer1
        # x2 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,64)
        x = model[5](x)  # layer2
        # x3 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,128)
        x = model[6](x)  # layer3
        x4 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1, 256)
        x = model[7](x)  # layer4
        x5 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1, 512)
        return x4, x5

    def _predict_proba(self, x, style="dict"):
        """predict all class probability"""
        if style == "dict":
            result = dict()
            for name, model in tqdm(self.classification_models.items()):
                result[name] = model.predict_proba(x)[:, 1]
        return result


class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.
    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer).
    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])
    """

    def __init__(
        self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        super(ProjectionHead, self).__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head.
        Args:
            x:
                Input of shape bsz x num_ftrs.
        """
        return self.layers(x)


class BarlowTwinsProjectionHead(ProjectionHead):
    """Projection head used for Barlow Twins.
    "The projector network has three linear layers, each with 8192 output
    units. The first two layers of the projector are followed by a batch
    normalization layer and rectified linear units." [0]
    [0]: 2021, Barlow Twins, https://arxiv.org/abs/2103.03230
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 8192, output_dim: int = 8192
    ):
        super(BarlowTwinsProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class BarlowTwinsLoss(torch.nn.Module):
    """Implementation of the Barlow Twins Loss from Barlow Twins[0] paper.
    This code specifically implements the Figure Algorithm 1 from [0].

    [0] Zbontar,J. et.al, 2021, Barlow Twins... https://arxiv.org/abs/2103.03230
        Examples:
        >>> # initialize loss function
        >>> loss_fn = BarlowTwinsLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimSiam model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def __init__(self, lambda_param: float = 5e-3, gather_distributed: bool = False):
        """Lambda param configuration with default value like in [0]
        Args:
            lambda_param:
                Parameter for importance of redundancy reduction term.
                Defaults to 5e-3 [0].
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are
                gathered and summed before the loss calculation.
        """
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:

        device = z_a.device

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD
        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD

        # sum cross-correlation matrix between multiple gpus
        # if self.gather_distributed and dist.is_initialized():
        #    world_size = dist.get_world_size()
        #    if world_size > 1:
        #        c = c / world_size
        #        dist.all_reduce(c)

        # loss
        c_diff = (c - torch.eye(D, device=device)).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss


class BarlowTwins(nn.Module):
    def __init__(self, backbone, head_size=[2048, 512, 128]):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(*head_size)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

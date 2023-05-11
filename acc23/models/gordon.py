"""
ACC23 main multi-classification model: prototype "Gordon". Fusion model
inspired by

    A Novel Attention-Based Multi-Modal Modeling Technique on Mixed Type Data
    for Improving TFT-LCD Repair Process
"""
__docformat__ = "google"

from typing import Any, Dict, List, Union

import torch
from torch import Tensor, nn
from transformers.activations import get_activation
from transformers.models.resnet.modeling_resnet import ResNetConvLayer

from acc23.constants import IMAGE_RESIZE_TO, N_CHANNELS, N_FEATURES, N_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .utils import ResNetLinearLayer, concat_tensor_dict


class ConvFusionLayer(nn.Module):
    """
    See Figure 2 of

        A Novel Attention-Based Multi-Modal Modeling Technique on Mixed
        Type Data for Improving TFT-LCD Repair Process
    """

    block_1: nn.Module
    block_2: nn.Module
    conv: nn.Module
    linear: nn.Module

    def __init__(
        self, in_channels: int, n_features: int, activation: str = "silu"
    ) -> None:
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.Softmax(dim=1),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.BatchNorm2d(in_channels),
            get_activation(activation),
        )
        self.linear = nn.Linear(n_features, in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x: Tensor, h: Tensor, *_, **__) -> Tensor:
        """Override"""
        # print(">", x.shape)
        # print(self.block_1(x).shape)
        u = self.block_1(x) * x
        u = self.block_2(u)
        v = self.linear(h)
        v = torch.stack([v] * u.shape[2] * u.shape[3], dim=-1).reshape(u.shape)
        w = u + v  # add v along the channels of every location ("pixel") of u
        w = self.conv(w)
        return x + w


class FusionEncoder(nn.Module):
    """
    Fusion encoder (without attention). It is a succession of blocks that look
    like
    1. `ResNetConvLayer`: a residual convolutional block that cuts the image
        size (height and width) by half;
    2. `ConvFusionLayer` that incorporates the feature vector into the channels
        of the image.
    Note that the last `ResNetConvLayer` block is not followed by a fusion
    layer. Instead, the output is flattened.
    """

    convolution_layers: nn.ModuleList
    fusion_layers: nn.ModuleList

    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        n_features: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        c = [in_channels] + out_channels
        self.convolution_layers = nn.ModuleList(
            [
                ResNetConvLayer(c[i - 1], c[i], 3, 2, activation)
                for i in range(1, len(c))
            ]
        )
        self.fusion_layers = nn.ModuleList(
            [
                ConvFusionLayer(a, n_features, activation)
                for a in out_channels[:-1]
            ]
        )

    def forward(self, x: Tensor, h: Tensor, *_, **__) -> Tensor:
        """Override"""
        for c, f in zip(self.convolution_layers[:-1], self.fusion_layers):
            x = f(c(x), h)
        x = self.convolution_layers[-1](x)
        return x.flatten(1)


class Gordon(BaseMultilabelClassifier):
    """See module documentation"""

    _module_a: nn.Module  # Dense input branch
    _module_b: nn.Module  # Conv. pool input branch
    _module_c: nn.Module  # Conv fusion encoder
    _module_d: nn.Module  # Fusion branch

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        n_features = 256
        self._module_a = nn.Sequential(
            ResNetLinearLayer(N_FEATURES, 256),
            ResNetLinearLayer(256, n_features),
        )
        self._module_b = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 8, 3, 1, 1),  # IMAGE_RESIZE_TO = 512 -> 512
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.MaxPool2d(9, 2, 3),  # 256 -> 256
        )
        self._module_c = FusionEncoder(
            in_channels=8,
            out_channels=[
                8,  # 256 -> 128
                16,  # -> 64
                16,  # -> 32
                32,  # -> 16
                32,  # -> 8
                64,  # -> 4
                128,  # -> 2
                n_features,  # -> 1
            ],
            n_features=n_features,
        )
        self._module_d = nn.Sequential(
            nn.Linear(2 * n_features, N_TARGETS),
            nn.Sigmoid(),
        )
        self.example_input_array = (
            torch.zeros((32, N_FEATURES)),
            torch.zeros((32, N_CHANNELS, IMAGE_RESIZE_TO, IMAGE_RESIZE_TO)),
        )
        self.forward(*self.example_input_array)

    def forward(
        self,
        x: Union[Tensor, Dict[str, Tensor]],
        img: Tensor,
        *_,
        **__,
    ):
        # One operation per line for easier troubleshooting
        if isinstance(x, dict):
            x = concat_tensor_dict(x)
        x = x.float().to(self.device)  # type: ignore
        img = img.to(self.device)  # type: ignore
        a = self._module_a(x)
        b = self._module_b(img)
        c = self._module_c(b, a)
        ac = torch.concatenate([a, c], dim=-1)
        d = self._module_d(ac)
        return d

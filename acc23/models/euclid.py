"""
ACC23 main multi-classification model: prototype "Euclid". Basically "Ampere"
but with convopooling blocks instead of resnet blocks
"""
__docformat__ = "google"

from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor, nn
from transformers.activations import get_activation

from acc23.constants import IMAGE_RESIZE_TO, N_CHANNELS, N_FEATURES, N_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .utils import ResNetLinearLayer, concat_tensor_dict


class ConvolutionalBlock(nn.Module):
    convolution: nn.Module
    normalization: nn.Module
    activation: nn.Module
    pooling: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        pooling: Optional[str] = None,
        pooling_size: int = 3,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, 4, 2, 1, bias=False
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)
        if pooling == "max":
            self.pooling = nn.MaxPool2d(pooling_size, 1, int(pooling_size / 2))
        elif pooling == "avg":
            self.pooling = nn.AvgPool2d(pooling_size, 1, int(pooling_size / 2))
        else:
            self.pooling = nn.Identity()

    def forward(self, x: Tensor, *_, **__) -> Tensor:
        """Override"""
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.pooling(x)
        return x


class Euclid(BaseMultilabelClassifier):
    """See module documentation"""

    _module_a: nn.Module  # Dense input branch
    _module_b: nn.Module  # Conv. input branch
    _module_c: nn.Module  # Merge branch

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        d = 256
        self._module_b = nn.Sequential(
            ConvolutionalBlock(
                N_CHANNELS, 8, pooling="max", pooling_size=7
            ),  # 128 -> 64
            # nn.AvgPool2d(7, 1, 3),
            ConvolutionalBlock(8, 16, pooling="max", pooling_size=7),  # -> 32
            # nn.AvgPool2d(5, 1, 2),
            ConvolutionalBlock(16, 32, pooling="max", pooling_size=3),  # -> 16
            # nn.AvgPool2d(5, 1, 2),
            ConvolutionalBlock(32, 32, pooling="avg", pooling_size=3),  # -> 8
            # nn.AvgPool2d(3, 1, 1),
            ConvolutionalBlock(32, 64),  # -> 4
            ConvolutionalBlock(64, 128),  # -> 2
            ConvolutionalBlock(128, d),  # -> 1
            nn.Flatten(),
        )
        self._module_a = nn.Sequential(
            ResNetLinearLayer(N_FEATURES, 256),
            ResNetLinearLayer(256, d),
        )
        self._module_c = nn.Sequential(
            ResNetLinearLayer(2 * d, 256),
            ResNetLinearLayer(256, 256),
            ResNetLinearLayer(256, 256),
            ResNetLinearLayer(256, 256),
            ResNetLinearLayer(256, 64),
            ResNetLinearLayer(64, N_TARGETS, last_activation="sigmoid"),
        )
        for p in self.parameters():
            if p.ndim >= 2:
                torch.nn.init.xavier_normal_(p)
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
        ab = torch.concatenate([a, b], dim=-1)
        c = self._module_c(ab)
        return c

"""
ACC23 main multi-classification model: prototype "Euclid". Basically "Ampere"
but with convopooling blocks instead of resnet blocks
"""
__docformat__ = "google"

from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor, nn
from transformers.activations import get_activation

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .layers import ResNetLinearLayer, concat_tensor_dict


class ConvolutionalBlock(nn.Module):
    """A simple convolution - normalization - activation - pooling block"""

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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
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

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor, *_, **__) -> Tensor:
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
            ResNetLinearLayer(64, N_TRUE_TARGETS),
        )
        for p in self.parameters():
            if p.ndim >= 2:
                torch.nn.init.xavier_normal_(p)
        self.example_input_array = (
            torch.zeros((32, N_FEATURES)),
            torch.zeros((32, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)),
        )
        self.forward(*self.example_input_array)

    def forward(
        self,
        x: Union[Tensor, Dict[str, Tensor]],
        img: Tensor,
        *_,
        **__,
    ) -> Tensor:
        """
        Args:
            x (Tensor): Tabular data with shape `(N, N_FEATURES)`, where `N` is
                the batch size, or alternatively, a string dict, where each key
                is a `(N,)` tensor.
            img (Tensor): Batch of images, i.e. a tensor of shape
                `(N, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)`
        """
        if isinstance(x, dict):
            x = concat_tensor_dict(x)
        x = x.float().to(self.device)  # type: ignore
        img = img.to(self.device)  # type: ignore
        a = self._module_a(x)
        b = self._module_b(img)
        ab = torch.concatenate([a, b], dim=-1)
        c = self._module_c(ab)
        return c

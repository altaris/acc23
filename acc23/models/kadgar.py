"""
ACC23 main multi-classification model: prototype "Kadgar". Like Gordon but the
vision encoder integrates CBAM modules from

    Woo, S., Park, J., Lee, JY., Kweon, I.S. (2018). CBAM: Convolutional
    Block Attention Module. In: Ferrari, V., Hebert, M., Sminchisescu, C.,
    Weiss, Y. (eds) Computer Vision - ECCV 2018. ECCV 2018. Lecture Notes
    in Computer Science(), vol 11211. Springer, Cham.
    https://doi.org/10.1007/978-3-030-01234-2_1
"""
__docformat__ = "google"

from itertools import zip_longest
from typing import Any, Dict, List, Union

import torch
from torch import Tensor, nn
from transformers.models.resnet.modeling_resnet import (
    ResNetConvLayer,
    ResNetShortCut,
)

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TARGETS
from acc23.models.cbam import CBAM

from .base_mlc import BaseMultilabelClassifier
from .imagetabnet import AttentionModule
from .layers import ResNetLinearLayer, concat_tensor_dict


class ConvCBAM(nn.Module):
    """
    A convolution layer followed by a CBAM block. The convolution kernel size,
    stride and padding is set so that the image size is halved. The whole block
    is residual.
    """

    conv: nn.Module
    cbam: nn.Module
    skip: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cbam_kernel_size: int = 7,
        cbam_reduction_ratio: int = 1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.conv = ResNetConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=2,
            activation=activation,
        )
        self.cbam = CBAM(
            in_channels=out_channels,
            reduction_ratio=cbam_reduction_ratio,
            kernel_size=cbam_kernel_size,
            activation="sigmoid",
        )
        self.skip = ResNetShortCut(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    # pylint: disable=missing-function-docstring
    def forward(self, img: Tensor, *_, **__) -> Tensor:
        a, b = self.conv(img), self.skip(img)
        c = self.cbam(a)
        return b + c


class VisionEncoder(nn.Module):
    """
    It is a succession of blocks that look like
    1. `ResNetConvLayer`: a residual convolutional block that cuts the image
        size (height and width) by half;
    2. `AttentionModule` that incorporates the feature vector into the channels
       of the image.
    """

    encoder_layers: nn.ModuleList
    fusion_layers: nn.ModuleList

    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        in_features: int,
        activation: str = "silu",
        attention_after_last: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int):
            out_channels (List[int]):
            in_features (int): Dimension of the feature vector to inject after
                each convolution stage
            n_decision_steps (int): Number of decision steps
            activation (str): Defaults to silu
            attention_after_last (bool): Wether to add an attention module
                after the last residual block, defaults to `False`.
        """
        super().__init__()
        c = [in_channels] + out_channels
        self.encoder_layers = nn.ModuleList(
            [
                ConvCBAM(c[i - 1], c[i], 3, 1, activation)
                for i in range(1, len(c))
            ]
        )
        k = len(out_channels) if attention_after_last else -1
        self.fusion_layers = nn.ModuleList(
            [
                AttentionModule(
                    a,
                    in_features,
                    activation,
                )
                for a in out_channels[:k]
            ]
        )

    def forward(self, img: Tensor, h: Tensor, *_, **__) -> Tensor:
        """
        Args:
            img (Tensor):
            h (Tensor):
        """
        # itertool.zip_longest pads the shorter sequence with None's
        for c, f in zip_longest(self.encoder_layers, self.fusion_layers):
            img = c(img)
            if f is not None:
                img = f(img, h)
        return img.flatten(1)


class Kadgar(BaseMultilabelClassifier):
    """See module documentation"""

    tabular_branch: nn.Module  # Dense input branch
    vision_branch_a: nn.Module  # Conv. pool input branch
    vision_branch_b: nn.Module  # Conv fusion encoder
    main_branch: nn.Module  # Fusion branch

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        n_features = 256
        self.tabular_branch = nn.Sequential(
            ResNetLinearLayer(N_FEATURES, 256),
            ResNetLinearLayer(256, n_features),
        )
        # self.vision_branch_a = nn.Sequential(
        #     nn.MaxPool2d(5, 1, 2),  # IMAGE_RESIZE_TO = 512 -> 512
        #     nn.Conv2d(N_CHANNELS, 8, 4, 2, 1, bias=False),  # -> 256
        #     nn.BatchNorm2d(8),
        #     nn.SiLU(),
        #     # nn.MaxPool2d(3, 1, 1),  # 256 -> 256
        # )
        self.vision_branch_a = nn.Sequential(
            nn.MaxPool2d(5, 2, 2),  # IMAGE_RESIZE_TO = 512 -> 256
        )
        self.vision_branch_b = VisionEncoder(
            in_channels=N_CHANNELS,
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
            in_features=n_features,
        )
        self.main_branch = nn.Sequential(
            nn.Linear(2 * n_features, N_TARGETS),
        )
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
        u = self.tabular_branch(x)
        v = self.vision_branch_a(img)
        v = self.vision_branch_b(v, u)
        uv = torch.concatenate([u, v], dim=-1)
        w = self.main_branch(uv)
        return w

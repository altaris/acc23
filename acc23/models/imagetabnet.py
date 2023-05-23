"""
ImageTabNet components from

    Y. Liu, H. -P. Lu and C. -H. Lai, "A Novel Attention-Based Multi-Modal
    Modeling Technique on Mixed Type Data for Improving TFT-LCD Repair
    Process," in IEEE Access, vol. 10, pp. 33026-33036, 2022, doi:
    10.1109/ACCESS.2022.3158952.
"""
__docformat__ = "google"

from itertools import zip_longest
from typing import List

import torch
from torch import Tensor, nn
from transformers.activations import get_activation
from transformers.models.resnet.modeling_resnet import ResNetConvLayer

# from acc23.models.layers import ResNetEncoderLayer

# class DecisionAggregation(nn.Module):
#     """
#     Decision aggregation module. It aggregates the output vector of all
#     decision steps in a `TabNetEncoder`.
#     """

#     blocks: nn.ModuleList

#     def __init__(
#         self, n_d: int, n_decision_steps: int = 3, activation: str = "silu"
#     ) -> None:
#         """
#         Args:
#             n_d (int): Dimension of the decision vector of a decision step
#             n_decision_steps (int): Number of decision steps
#             activation: Defaults to silu
#         """
#         super().__init__()
#         # TODO: What is the actual output dimension?
#         self.blocks = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Linear(n_d, n_d),
#                     nn.BatchNorm1d(n_d),
#                     get_activation(activation),
#                 )
#                 for _ in range(n_decision_steps)
#             ]
#         )

#     def forward(self, ds: Tensor, *_, **__) -> Tensor:
#         """
#         Args:
#             ds (Tensor): A `(n_decision_steps, N, n_d)` tensor

#         Returns:
#             An aggregated tensor of shape `(N, n_decision_steps * n_d)`, where
#             `N` is the batch size.
#         """

#         def _eval(fx: Tuple[nn.Module, Tensor]) -> Tensor:
#             f, x = fx
#             return f(x)

#         # TODO: iterating over a tensor generates a warning
#         return torch.concat(list(map(_eval, zip(self.blocks, ds))), dim=-1)


class AttentionModule(nn.Module):
    """See Figure 2 of the paper"""

    block_1: nn.Module
    block_2: nn.Module
    conv: nn.Module
    linear: nn.Module

    def __init__(
        self, in_channels: int, in_features: int, activation: str = "silu"
    ) -> None:
        """
        Args:
            in_channels (int):
            in_features (int): Dimension of the output of tabnet
            activation (str): Defaults to silu
        """
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False),
            nn.Softmax(dim=1),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels),
            get_activation(activation),
        )
        # **Linear** projection
        self.linear = nn.Linear(in_features, in_channels, bias=False)
        self.conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False)

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor, h: Tensor, *_, **__) -> Tensor:
        u = self.block_1(x) * x
        u = self.block_2(u)
        v = self.linear(h)
        v = torch.stack([v] * u.shape[2] * u.shape[3], dim=-1).reshape(u.shape)
        w = u + v  # add v along the channels of every location ("pixel") of u
        w = self.conv(w)
        return x + w


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
                ResNetConvLayer(c[i - 1], c[i], 3, 2, activation)
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

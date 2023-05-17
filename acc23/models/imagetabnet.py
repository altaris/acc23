"""
ImageTabNet components from

    Y. Liu, H. -P. Lu and C. -H. Lai, "A Novel Attention-Based Multi-Modal
    Modeling Technique on Mixed Type Data for Improving TFT-LCD Repair
    Process," in IEEE Access, vol. 10, pp. 33026-33036, 2022, doi:
    10.1109/ACCESS.2022.3158952.
"""
__docformat__ = "google"

from typing import List, Tuple

import torch
from torch import Tensor, nn
from transformers.activations import get_activation
from transformers.models.resnet.modeling_resnet import ResNetConvLayer


class DecisionAggregation(nn.Module):
    """
    Decision aggregation module. It aggregates the output vector of all
    decision steps in a `TabNetEncoder`.
    """

    blocks: nn.ModuleList

    def __init__(
        self, n_d: int, n_decision_steps: int = 3, activation: str = "silu"
    ) -> None:
        """
        Args:
            n_d (int): Dimension of the decision vector of a decision step
            n_decision_steps (int): Number of decision steps
            activation: Defaults to silu
        """
        super().__init__()
        # TODO: What is the actual output dimension?
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_d, n_d),
                    nn.BatchNorm1d(n_d),
                    get_activation(activation),
                )
                for _ in range(n_decision_steps)
            ]
        )

    def forward(self, ds: Tensor, *_, **__) -> Tensor:
        """
        Args:
            ds (Tensor): A `(n_decision_steps, N, n_d)` tensor

        Returns:
            An aggregated tensor of shape `(N, n_decision_steps * n_d)`, where
            `N` is the batch size.
        """

        def _eval(fx: Tuple[nn.Module, Tensor]) -> Tensor:
            f, x = fx
            return f(x)

        # TODO: iterating over a tensor generates a warning
        return torch.concat(list(map(_eval, zip(self.blocks, ds))), dim=-1)


class ImageTabNetAttentionModule(nn.Module):
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
            in_features (int): Dimension of the aggregated decision vector,
                usually `n_d * n_decision_steps`, where `n_d` is the dimension
                of a decision vector
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

    def forward(self, x: Tensor, h: Tensor, *_, **__) -> Tensor:
        """Override"""
        u = self.block_1(x) * x
        u = self.block_2(u)
        v = self.linear(h)
        v = torch.stack([v] * u.shape[2] * u.shape[3], dim=-1).reshape(u.shape)
        w = u + v  # add v along the channels of every location ("pixel") of u
        w = self.conv(w)
        return x + w


class ImageTabNetVisionEncoder(nn.Module):
    """
    It is a succession of blocks that look like
    1. `ResNetConvLayer`: a residual convolutional block that cuts the image
        size (height and width) by half;
    2. `ImageTabNetAttentionModule` that incorporates the feature vector into
       the channels of the image.
    Note that the last `ResNetConvLayer` block is not followed by a
    `ImageTabNetAttentionModule`. Instead, the output is flattened.
    """

    aggregation: nn.Module
    convolution_layers: nn.ModuleList
    fusion_layers: nn.ModuleList

    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        n_d: int,
        n_decision_steps: int,
        activation: str = "silu",
    ) -> None:
        """
        Args:
            in_channels (int):
            out_channels (List[int]):
            n_d (int): Dimension of the decision vector of a decision step
            n_decision_steps (int): Number of decision steps
            activation: Defaults to silu
        """
        super().__init__()
        self.aggregation = DecisionAggregation(
            n_d, n_decision_steps, activation
        )
        c = [in_channels] + out_channels
        self.convolution_layers = nn.ModuleList(
            [
                ResNetConvLayer(c[i - 1], c[i], 3, 2, activation)
                for i in range(1, len(c))
            ]
        )
        self.fusion_layers = nn.ModuleList(
            [
                ImageTabNetAttentionModule(
                    a, n_d * n_decision_steps, activation
                )
                for a in out_channels[:-1]
            ]
        )

    def forward(self, img: Tensor, ds: Tensor, *_, **__) -> Tensor:
        """
        Args:
            img (Tensor):
            ds (Tensor): A `(n_decision_steps, N, n_d)` tensor
        """
        d = self.aggregation(ds)
        for c, f in zip(self.convolution_layers[:-1], self.fusion_layers):
            img = f(c(img), d)
        img = self.convolution_layers[-1](img)
        return img.flatten(1)

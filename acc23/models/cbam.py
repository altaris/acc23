"""
Implementation of modules from

    Woo, S., Park, J., Lee, JY., Kweon, I.S. (2018). CBAM: Convolutional
    Block Attention Module. In: Ferrari, V., Hebert, M., Sminchisescu, C.,
    Weiss, Y. (eds) Computer Vision - ECCV 2018. ECCV 2018. Lecture Notes
    in Computer Science(), vol 11211. Springer, Cham.
    https://doi.org/10.1007/978-3-030-01234-2_1
"""


import torch
from torch import Tensor, nn
from transformers.activations import get_activation


class ChannelAttentionModule(nn.Module):
    """Part of `CBAM`"""

    activation: nn.Module
    mlp: nn.Module

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 1,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if not 1 <= reduction_ratio <= in_channels:
            raise ValueError("Reduction ratio must be <= in_channels and >= 1")
        d = in_channels // reduction_ratio
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, d),
            nn.Linear(d, in_channels),
        )
        self.activation = get_activation(activation)

    # pylint: disable=missing-function-docstring
    def forward(self, img: Tensor, *_, **__) -> Tensor:
        """
        Args:
            img (Tensor): `(N, C, H, W)`

        Returns:
            A tensor of shape `(N, C)`
        """
        a, b = img.max(dim=-1).values.max(dim=-1).values, img.mean(dim=(2, 3))
        a, b = self.mlp(a), self.mlp(b)
        c = self.activation(a + b)
        return c


class SpatialAttentionModule(nn.Module):
    """Part of `CBAM`"""

    activation: nn.Module
    conv: nn.Module

    def __init__(
        self,
        kernel_size: int = 7,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if not (kernel_size >= 1 and kernel_size % 2 == 1):
            raise ValueError("Kernel size must be an odd number >= 1")
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.activation = get_activation(activation)

    # pylint: disable=missing-function-docstring
    def forward(self, img: Tensor, *_, **__) -> Tensor:
        """
        Args:
            img (Tensor): `(N, C, H, W)`

        Returns:
            A tensor of shape `(N, H, W)`
        """
        a = img.max(dim=1, keepdim=True).values
        b = img.mean(dim=1, keepdim=True)
        ab = torch.concat([a, b], dim=1)
        c = self.conv(ab)
        c = self.activation(c)
        c = c.squeeze(dim=1)
        return c


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    """

    cam: nn.Module
    sam: nn.Module

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 1,
        kernel_size: int = 7,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.cam = ChannelAttentionModule(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio,
            activation=activation,
        )
        self.sam = SpatialAttentionModule(
            kernel_size=kernel_size,
            activation=activation,
        )

    # pylint: disable=missing-function-docstring
    def forward(self, img: Tensor, *_, **__) -> Tensor:
        a = self.cam(img)  # (N, C)
        a = a.unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        u = img * a
        b = self.sam(u)  # (N, H, W)
        b = b.unsqueeze(1)  # (N, 1, H, W)
        v = u * b
        return v

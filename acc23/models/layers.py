"""Model utilities"""
__docformat__ = "google"

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers.models.resnet.modeling_resnet import (
    ResNetConvLayer,
    ResNetBasicLayer,
)
from transformers.activations import get_activation


from acc23.constants import IMAGE_SIZE


class MLP(nn.Sequential):
    """A MLP (yep)"""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        layer_norm: bool = True,
        activation: str = "gelu",
        dropout: float = 0.0,
        is_head: bool = False,
    ):
        """
        Args:
            in_dim (int):
            hidden_dims (List[int]):
            layer_norm (bool):
            activation (str):
            dropout (float):
            is_head (bool): If set to `True`, there will be no layer
                normalization, activation, nor dropout after the last dense
                layer
        """
        ns = [in_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(1, len(ns)):
            a, b = ns[i - 1], ns[i]
            layers.append(nn.Linear(a, b))
            if not (is_head and i == len(ns) - 1):
                # The only time we don't add ln/act/drop is after the last
                # layer if the MLP is a head
                if layer_norm:
                    layers.append(nn.LayerNorm(b))
                layers.append(get_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        super().__init__(*layers)


class ResidualLinearBlock(nn.Module):
    """
    Residual linear block from

        Lim, Bee, et al. "Enhanced deep residual networks for single image
        super-resolution." Proceedings of the IEEE conference on computer
        vision and pattern recognition workshops. 2017.

    Note that the input and output dimensions are the same
    """

    block: nn.Module

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "relu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        hidden_dim = hidden_dim or embed_dim
        self.block = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, embed_dim),
        )

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ResNetConvTransposeLayer(nn.Module):
    """
    Just like `transformers.models.resnet.modeling_resnet.ResNetConvLayer`
    (i.e. a convolution-normalization-activation) layer, but with a transposed
    convolution instead.
    """

    convolution: nn.Module
    # normalization: nn.Module
    activation: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.convolution = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        # self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = (
            get_activation(activation)
            if activation is not None
            else nn.Identity()
        )

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor) -> Tensor:
        x = self.convolution(x)
        # x = self.normalization(x)
        x = self.activation(x)
        return x


class ResNetDecoderLayer(nn.Module):
    """
    A resnet decoder layer is a sequential model composed of
    1. `n_blocks`
       `transformers.models.resnet.modeling_resnet.ResNetBasicLayer`s, which
       are just double convolution (and normalization-activation) with residual
       connection.
    2. a `ResNetConvTransposeLayer` that doubles the size (width or height) of
       the input image
    """

    convolution: nn.Module
    residual_blocks: nn.Sequential

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 1,
        activation: str = "silu",
        last_activation: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        last_activation = last_activation or activation
        self.residual_blocks = nn.Sequential(
            *[
                ResNetBasicLayer(
                    in_channels, in_channels, activation=activation
                )
                for _ in range(n_blocks)
            ]
        )
        self.convolution = ResNetConvTransposeLayer(
            in_channels,
            out_channels,
            activation=last_activation,
        )

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor) -> Tensor:
        x = self.residual_blocks(x)
        x = self.convolution(x)
        return x


class ResNetEncoderLayer(nn.Module):
    """
    A resnet encoder layer is a sequential model composed of
    1. a `transformers.models.resnet.modeling_resnet.ResNetConvLayer` that
       halves the size (width or height) of the input image.
    2. `n_blocks`
       `transformers.models.resnet.modeling_resnet.ResNetBasicLayer`s, which
       are just double convolution (and normalization-activation) with residual
       connection.
    """

    convolution: nn.Module
    residual_blocks: nn.Sequential

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 1,
        activation: str = "silu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.convolution = ResNetConvLayer(
            in_channels, out_channels, 5, 2, activation=activation
        )
        self.residual_blocks = nn.Sequential(
            *[
                ResNetBasicLayer(
                    out_channels, out_channels, activation=activation
                )
                for _ in range(n_blocks)
            ]
        )

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor) -> Tensor:
        x = self.convolution(x)
        x = self.residual_blocks(x)
        return x


class ResNetLinearLayer(nn.Module):
    """Linear block with skip connection"""

    linear: nn.Module
    residual: nn.Module
    last_activation: nn.Module
    dropout: nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        latent_features: Optional[int] = None,
        activation: str = "silu",
        last_activation: Optional[str] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        latent_features = latent_features or out_features
        last_activation = last_activation or activation
        self.last_activation = get_activation(last_activation)
        self.linear = nn.Sequential(
            nn.Linear(in_features, latent_features),
            nn.BatchNorm1d(latent_features),
            get_activation(activation),
            nn.Linear(latent_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        self.residual = (
            nn.Identity()
            if in_features == out_features
            else nn.Linear(in_features, out_features)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor) -> Tensor:
        a = self.linear(x)
        b = self.residual(x)
        ab = self.last_activation(a + b)
        z = self.dropout(ab)
        return z


def basic_encoder(
    in_channels: int,
    out_channels: List[int],
    activation: str = "silu",
    input_size: int = IMAGE_SIZE,
) -> Tuple[nn.Sequential, int]:
    """
    Basic image encoder that is just a succession of (non skipped) downsampling
    blocks, followed by a final flatten layer. A block is just a
    `transformers.models.resnet.modeling_resnet.ResNetConvLayer`. In
    particular, the output images' size will be `input_size / (2 **
    len(out_channels))`, and the output vector dimension will be

        out_channels[-1] * (input_size / (2 ** len(out_channels))) ** 2

    Args:
        in_channels (int): Number of channels of the input image
        out_channels (List[int]): Number of output channels of each
            block. Note that each blocks cuts the width and height of the input
            by half.
        activation (str): Activation function name.
        input_size (int): The width (or equivalently the height) of an imput
            image.

    Return:
        The encoder and the (flat) dimension of the output vector.
    """
    module, c = nn.Sequential(), [in_channels] + out_channels
    module = nn.Sequential(
        *[
            ResNetConvLayer(c[i - 1], c[i], 5, 2, activation)
            for i in range(1, len(c))
        ]
    )
    # for i in range(1, len(c)):
    #     module.append(nn.Conv2d(c[i - 1], c[i], 4, 2, 1))
    #     if i > 1:
    #         module.append(nn.BatchNorm2d(c[i]))
    #     module.append(get_activation(activation))
    module.append(nn.Flatten())
    # Height (eq. width) of the encoded image before flatten
    es = int(input_size / (2 ** len(out_channels)))
    return (module, out_channels[-1] * (es**2))


def concat_tensor_dict(d: Dict[str, Tensor]) -> Tensor:
    """
    Converts a dict of tensors to a tensor. The tensors (of the dict) are
    concatenated along the last axis. If they are 1-dimensional, they are
    unsqueezed along a new axis at the end. I know it's not super clear, so
    here is an example

        # The foo tensor will be unsqueezed
        d = {"foo": torch.ones(3), "bar": torch.zeros((3, 1))}
        concat_tensor_dict(d)
        >>> tensor([[1., 0.],
        >>>         [1., 0.],
        >>>         [1., 0.]])

    """

    def _maybe_unsqueeze(t: Tensor) -> Tensor:
        return t if t.ndim > 1 else t.unsqueeze(-1)

    return torch.concatenate(list(map(_maybe_unsqueeze, d.values())), dim=-1)


def linear_chain(
    n_inputs: int,
    n_neurons: List[int],
    activation: str = "silu",
    last_activation: Optional[str] = None,
) -> nn.Sequential:
    """A sequence of linear layers."""
    last_activation = last_activation or activation
    n, module = [n_inputs] + n_neurons, nn.Sequential()
    for i in range(1, len(n)):
        module.append(nn.Linear(n[i - 1], n[i]))
        module.append(
            get_activation(last_activation if i == len(n) - 1 else activation)
        )
    return module


def resnet_decoder(
    in_channels: int,
    out_channels: List[int],
    activation: str = "silu",
    last_activation: str = "sigmoid",
    n_blocks: int = 1,
) -> nn.Sequential:
    """
    Basic image decoder that is just a succession of resnet decoder (see
    `acc23.models.utils.ResNetDecoderLayer`). In particular, the output images'
    size will be `input_size / (2 ** len(out_channels))`.

    Args:
        in_channels (int): Number of channels of the input image
        out_channels (List[int]): Number of output channels of each
            encoder layer. Note that each layer cuts the width and height of
            the input by half.
        activation (str): Activation function name.
        last_activation (str): Activation applied at the end of the last block.
        n_blocks (int): Number of residual blocks per encoder layer.

    Return:
        The decoder
    """
    c = [in_channels] + out_channels
    module = nn.Sequential(
        *[
            ResNetDecoderLayer(
                c[i - 1],
                c[i],
                n_blocks,
                activation=activation,
                last_activation=(
                    last_activation if i == len(c) - 1 else activation
                ),
            )
            for i in range(1, len(c))
        ]
    )
    return module


def resnet_encoder(
    in_channels: int,
    out_channels: List[int],
    activation: str = "silu",
    n_blocks: int = 1,
    input_size: int = IMAGE_SIZE,
) -> Tuple[nn.Sequential, int]:
    """
    Basic image encoder that is just a succession of resnet encoders (see
    `acc23.models.utils.ResNetEncoderLayer`), followed by a final flatten
    layer. In particular, the output images' size will be `input_size / (2 **
    len(out_channels))`, and the output vector dimension will be

        out_channels[-1] * (input_size / (2 ** len(out_channels))) ** 2

    Args:
        in_channels (int): Number of channels of the input image
        out_channels (List[int]): Number of output channels of each
            encoder layer. Note that each layer cuts the width and height of
            the input by half.
        activation (str): Activation function name.
        n_blocks (int): Number of residual blocks per encoder layer.
        input_size (int): The width (or equivalently the height) of an imput
            image.

    Return:
        The encoder and the (flat) dimension of the output vector.
    """
    c = [in_channels] + out_channels
    module = nn.Sequential(
        *[
            ResNetEncoderLayer(c[i - 1], c[i], n_blocks, activation)
            for i in range(1, len(c))
        ]
    )
    module.append(nn.Flatten())
    # Height (eq. width) of the encoded image before flatten
    es = int(input_size / (2 ** len(out_channels)))
    return (module, out_channels[-1] * (es**2))

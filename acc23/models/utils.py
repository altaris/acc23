"""Model utilities"""
__docformat__ = "google"

from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from transformers.models.resnet.modeling_resnet import (
    ResNetConvLayer,
    ResNetBasicLayer,
)
from transformers.activations import get_activation


from acc23.constants import IMAGE_RESIZE_TO


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
        activation: str = "relu",
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

    def forward(self, x: Tensor) -> Tensor:
        """Override"""
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
        activation: str = "gelu",
        last_activation: str = "gelu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
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

    def forward(self, x: Tensor) -> Tensor:
        """Override"""
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
        activation: str = "relu",
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

    def forward(self, x: Tensor) -> Tensor:
        """Override"""
        x = self.convolution(x)
        x = self.residual_blocks(x)
        return x


def basic_encoder(
    in_channels: int,
    out_channels: List[int],
    activation: str = "relu",
    input_size: int = IMAGE_RESIZE_TO,
) -> Tuple[nn.Sequential, int]:
    """
    Basic image encoder that is just a succession of (non skipped) downsampling
    blocks, followed by a final flatten layer. A block is made of
    1. a convolution layer which cuts the inputs' height and width by half,
    2. a batch normalization layer, except for the first block,
    3. a sigmoid activation.
    In particular, the output images' size will be `input_size / (2 **
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
    activation: str = "sigmoid",
    last_activation: str = "sigmoid",
) -> nn.Sequential:
    """
    A sequence of linear layers with sigmoid activation (including at the end).
    """
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
    activation: str = "gelu",
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
    activation: str = "relu",
    n_blocks: int = 1,
    input_size: int = IMAGE_RESIZE_TO,
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

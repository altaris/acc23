"""ACC23 model prototype *Ampere*"""

from typing import Any, Dict, Union

import torch
from torch import Tensor, nn
from transformers.models.resnet.modeling_resnet import ResNetConvLayer

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .layers import MLP, concat_tensor_dict


class Ampere(BaseMultilabelClassifier):
    """
    This is perhaps the simplest deep neural network attempt at the problem.
    The image is fed through a stack of convolutional layers. The tabular data
    is fed through another branch made of dense layers. Then the result of both
    branches are concatenated and fed through the MLP head.
    """

    _tab_mlp: nn.Module
    _conv_stack: nn.Module
    _mlp_head: nn.Module

    def __init__(
        self, embed_dim: int = 256, activation: str = "gelu", **kwargs: Any
    ) -> None:
        """
        Args:
            embed_dim (int, optional):
            activation (str, optional):

        See also:
            `acc23.models.base_mlc.BaseMultilabelClassifier.__init__`
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        kw = {"kernel_size": 3, "stride": 2, "activation": activation}
        self._conv_stack = nn.Sequential(
            nn.MaxPool2d(5, 1, 2),  # 256 -> 128
            ResNetConvLayer(N_CHANNELS, 8, **kw),  # -> 128
            ResNetConvLayer(8, 16, **kw),  # -> 64
            ResNetConvLayer(16, 16, **kw),  # -> 32
            ResNetConvLayer(16, 32, **kw),  # -> 16
            ResNetConvLayer(32, 32, **kw),  # -> 8
            ResNetConvLayer(32, 64, **kw),  # -> 4
            ResNetConvLayer(64, 128, **kw),  # -> 2
            ResNetConvLayer(128, embed_dim, **kw),  # -> 1
            nn.Flatten(),
        )
        self._tab_mlp = MLP(
            N_FEATURES, [256, embed_dim], activation=activation
        )
        self._mlp_head = MLP(
            2 * embed_dim,
            [256, 256, 256, 256, 64, N_TRUE_TARGETS],
            activation=activation,
            is_head=True,
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
        # One operation per line for easier troubleshooting
        if isinstance(x, dict):
            x = concat_tensor_dict(x)
        x = x.float().to(self.device)  # type: ignore
        img = img.to(self.device)  # type: ignore
        a = self._tab_mlp(x)
        b = self._conv_stack(img)
        ab = torch.concatenate([a, b], dim=-1)
        c = self._mlp_head(ab)
        return c

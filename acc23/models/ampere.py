"""
ACC23 main multi-classification model: prototype "Ampere". This is perhaps the
simplest deep neural network attempt at the problem. The image is fed through a
stack of convolutional layers, called the _convolutional input branch_. The
data corresponding to the image is fed through another branch made of dense
(aka linear) layers, unsurprisingly called the _dense input branch_. Then, the
(flattened) encoded image and latent representation of the data are
concatenated and fed through yet another dense stack, which produces the
output. This is called the _merge branch_.
"""
__docformat__ = "google"

from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor, nn

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TARGETS

from .layers import (
    ResNetEncoderLayer,
    concat_tensor_dict,
    ResNetLinearLayer,
)
from .base_mlc import BaseMultilabelClassifier


class Ampere(BaseMultilabelClassifier):
    """See module documentation"""

    _module_a: nn.Module  # Dense input branch
    _module_b: nn.Module  # Conv. input branch
    _module_c: nn.Module  # Merge branch

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        d = 256
        self._module_b = nn.Sequential(
            nn.MaxPool2d(5, 1, 2),
            ResNetEncoderLayer(N_CHANNELS, 8),  # IMAGE_RESIZE_TO = 512 -> 256
            ResNetEncoderLayer(8, 8),  # -> 128
            ResNetEncoderLayer(8, 16),  # -> 64
            ResNetEncoderLayer(16, 16),  # -> 32
            ResNetEncoderLayer(16, 32),  # -> 16
            ResNetEncoderLayer(32, 32),  # -> 8
            ResNetEncoderLayer(32, 64),  # -> 4
            ResNetEncoderLayer(64, 128),  # -> 2
            ResNetEncoderLayer(128, d),  # -> 1
            nn.Flatten(),
        )
        # self._module_b = nn.Sequential(
        #     nn.MaxPool2d(7, 1, 3),
        #     ResNetEncoderLayer(N_CHANNELS, 8),  # 128 -> 64
        #     ResNetEncoderLayer(8, 16),  # -> 32
        #     ResNetEncoderLayer(16, 32),  # -> 16
        #     ResNetEncoderLayer(32, 32),  # -> 8
        #     ResNetEncoderLayer(32, 64),  # -> 4
        #     ResNetEncoderLayer(64, 128),  # -> 2
        #     ResNetEncoderLayer(128, d),  # -> 1
        #     nn.Flatten(),
        # )
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
            ResNetLinearLayer(64, N_TARGETS),
        )
        # for p in self.parameters():
        #     if p.ndim >= 2:
        #         torch.nn.init.xavier_normal_(p)
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
    ) -> Tuple[Tensor, Union[Tensor, float]]:
        """
        Args:
            x (Tensor): Tabular data with shape `(N, N_FEATURES)`, where `N` is
                the batch size, or alternatively, a string dict, where each key
                is a `(N,)` tensor.
            img (Tensor): Batch of images, i.e. a tensor of shape
                `(N, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)`

        Returns:
            1. Output logits
            2. An extra loss term (just return 0 if you have nothing to add)
        """
        # One operation per line for easier troubleshooting
        if isinstance(x, dict):
            x = concat_tensor_dict(x)
        x = x.float().to(self.device)  # type: ignore
        img = img.to(self.device)  # type: ignore
        a = self._module_a(x)
        b = self._module_b(img)
        ab = torch.concatenate([a, b], dim=-1)
        c = self._module_c(ab)
        return c, 0.0

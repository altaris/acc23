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

from typing import Any, Dict, Union

import torch
from torch import Tensor, nn

from acc23.constants import IMAGE_RESIZE_TO, N_CHANNELS, N_FEATURES, N_TARGETS

from .utils import (
    resnet_encoder,
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
        self._module_b, encoded_dim = resnet_encoder(
            N_CHANNELS,
            [
                8,  # IMAGE_RESIZE_TO = 512 -> 256
                8,  # -> 128
                16,  # -> 64
                16,  # -> 32
                32,  # -> 16
                32,  # -> 8
                64,  # -> 4
                128,  # -> 2
                256,  # -> 1
            ],
            # [
            #     8,  # IMAGE_RESIZE_TO = 128 -> 64
            #     16,  # -> 32
            #     32,  # -> 16
            #     32,  # -> 8
            #     64,  # -> 4
            # ],
            n_blocks=1,
        )
        self._module_a = nn.Sequential(
            ResNetLinearLayer(N_FEATURES, 256),
            ResNetLinearLayer(256, encoded_dim),
            ResNetLinearLayer(encoded_dim, encoded_dim),
            ResNetLinearLayer(encoded_dim, encoded_dim),
            ResNetLinearLayer(encoded_dim, encoded_dim),
        )
        self._module_c = nn.Sequential(
            ResNetLinearLayer(2 * encoded_dim, 512),
            ResNetLinearLayer(512, 512),
            ResNetLinearLayer(512, 512),
            ResNetLinearLayer(512, 512),
            ResNetLinearLayer(512, 256),
            ResNetLinearLayer(256, 256),
            ResNetLinearLayer(256, 256),
            ResNetLinearLayer(256, 256),
            ResNetLinearLayer(256, 64),
            ResNetLinearLayer(64, N_TARGETS, last_activation="sigmoid"),
        )
        # for p in self.parameters():
        #     if p.ndim >= 2:
        #         torch.nn.init.xavier_normal_(p)
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

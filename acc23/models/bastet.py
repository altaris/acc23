"""
ACC23 main multi-classification model: prototype "Bastet". The _merge branch_
"""
__docformat__ = "google"

from typing import Any, Dict, Union

import torch
from torch import Tensor, nn

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TARGETS

from .layers import (
    ResNetLinearLayer,
    resnet_encoder,
    concat_tensor_dict,
)
from .base_mlc import BaseMultilabelClassifier


class Bastet(BaseMultilabelClassifier):
    """See module documentation"""

    _module_a: nn.Module  # Dense input branch
    _module_b: nn.Module  # Conv. input branch
    _module_d: nn.Module  # Merge branch: transformer
    _module_e: nn.Module  # Merge branch: output

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self._module_b, encoded_dim = resnet_encoder(
            N_CHANNELS,
            [
                8,  # IMAGE_RESIZE_TO = 512 -> 256
                16,  # -> 128
                16,  # -> 64
                32,  # -> 32
                32,  # -> 16
                64,  # -> 8
                64,  # -> 4
                128,  # -> 2
            ],
            # [
            #     4,  # IMAGE_RESIZE_TO = 128 -> 64
            #     8,  # -> 32
            #     16,  # -> 16
            #     32,  # -> 8
            #     64,  # -> 4
            #     128,  # -> 2
            # ],
        )
        # self._module_a = linear_chain(N_FEATURES, [512, encoded_dim])
        self._module_a = nn.Sequential(
            ResNetLinearLayer(N_FEATURES, 256),
            ResNetLinearLayer(256, 256),
            ResNetLinearLayer(256, 256),
            ResNetLinearLayer(256, 512),
            ResNetLinearLayer(512, 512),
            ResNetLinearLayer(512, 512),
            ResNetLinearLayer(512, encoded_dim),
        )
        self._module_c = ResNetLinearLayer(2 * encoded_dim, encoded_dim)
        self._module_d = nn.Transformer(
            d_model=encoded_dim,
            nhead=8,
            batch_first=True,
            num_encoder_layers=6,
            num_decoder_layers=6,
        )
        self._module_e = nn.Sequential(
            ResNetLinearLayer(encoded_dim, 256),
            ResNetLinearLayer(256, 64),
            ResNetLinearLayer(64, N_TARGETS),
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
        d = self._module_d(c, a)
        e = self._module_e(d)
        return e

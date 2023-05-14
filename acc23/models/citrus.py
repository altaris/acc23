"""
ACC23 main multi-classification model: prototype "Citrus". This model uses
multihead attention layers to merge the dense and convolutional input branches.
"""
__docformat__ = "google"

from typing import Any, Dict, Union

import torch
from torch import Tensor, nn

from acc23.constants import IMAGE_RESIZE_TO, N_CHANNELS, N_FEATURES, N_TARGETS

from .utils import (
    ResNetLinearLayer,
    basic_encoder,
    concat_tensor_dict,
)
from .base_mlc import BaseMultilabelClassifier


class Citrus(BaseMultilabelClassifier):
    """See module documentation"""

    _module_a: nn.Module  # Input-dense part
    _module_b: nn.Module  # Input-conv part
    _module_c: nn.Module  # Attention part (transformer encoder)
    _module_d: nn.Module  # Output part

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self._module_b, encoded_dim = basic_encoder(
            N_CHANNELS,
            [
                16,  # IMAGE_RESIZE_TO = 512 -> 256
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
            #     64,  # -> 2
            # ],
        )
        # self._module_a = linear_chain(N_FEATURES, [encoded_dim])
        self._module_a = nn.Sequential(
            ResNetLinearLayer(N_FEATURES, 256),
            ResNetLinearLayer(256, 256),
            ResNetLinearLayer(256, 512),
            ResNetLinearLayer(512, encoded_dim),
        )
        self._module_c = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2 * encoded_dim,
                nhead=8,
                dim_feedforward=4 * encoded_dim,
                batch_first=True,
            ),
            num_layers=2,
        )
        self._module_d = nn.Sequential(
            ResNetLinearLayer(2 * encoded_dim, 512),
            ResNetLinearLayer(512, 256),
            ResNetLinearLayer(256, 128),
            ResNetLinearLayer(128, 64),
            ResNetLinearLayer(64, N_TARGETS),
        )
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
        d = self._module_d(c)
        return d

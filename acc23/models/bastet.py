"""
ACC23 main multi-classification model: prototype "Bastet". The _merge branch_
"""
__docformat__ = "google"

from typing import Dict, Union

import torch
from torch import Tensor, nn

from acc23.constants import IMAGE_RESIZE_TO, N_CHANNELS, N_FEATURES, N_TARGETS

from .utils import (
    BaseMultilabelClassifier,
    basic_encoder,
    concat_tensor_dict,
    linear_chain,
)


class Bastet(BaseMultilabelClassifier):
    """See module documentation"""

    _module_a: nn.Module  # Dense input branch
    _module_b: nn.Module  # Conv. input branch
    _module_c: nn.Module  # Merge branch: transformer
    _module_d: nn.Module  # Merge branch: output

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self._module_b, encoded_dim = basic_encoder(
            [
                4,  # IMAGE_RESIZE_TO = 128 -> 64
                8,  # -> 32
                16,  # -> 16
                32,  # -> 8
                64,  # -> 4
                128,  # -> 2
            ],
        )
        self._module_a = linear_chain(N_FEATURES, [512, encoded_dim])
        self._module_c = nn.Transformer(
            d_model=encoded_dim,
            nhead=8,
            batch_first=True,
            num_encoder_layers=1,
            num_decoder_layers=4,
        )
        self._module_d = linear_chain(encoded_dim, [256, 64, N_TARGETS])
        self.example_input_array = (
            torch.zeros((1, N_FEATURES)),
            torch.zeros((1, N_CHANNELS, IMAGE_RESIZE_TO, IMAGE_RESIZE_TO)),
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
        c = self._module_c(b, a)
        d = self._module_d(c)
        return d
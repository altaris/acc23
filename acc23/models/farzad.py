"""
ACC23 main multi-classification model: prototype "Farzad". Like "Ampere", but
the convolution branch is replaced by the encoder-sample part of a pretrained
VAE.
"""
__docformat__ = "google"

from typing import Dict, Union

import torch
from torch import Tensor, nn

from acc23.constants import N_FEATURES, N_TARGETS

from .layers import (
    concat_tensor_dict,
    ResNetLinearLayer,
)
from .base_mlc import BaseMultilabelClassifier


class Farzad(BaseMultilabelClassifier):
    """See module documentation"""

    _module_a: nn.Module  # Dense input branch
    _module_b: nn.Module  # Merge branch

    def __init__(self, vae_latent_dim: int = 256) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._module_a = nn.Sequential(
            ResNetLinearLayer(N_FEATURES, 256),
            ResNetLinearLayer(256, vae_latent_dim),
        )
        self._module_b = nn.Sequential(
            ResNetLinearLayer(2 * vae_latent_dim, 256),
            ResNetLinearLayer(256, 256),
            ResNetLinearLayer(256, 256),
            ResNetLinearLayer(256, 256),
            ResNetLinearLayer(256, 64),
            ResNetLinearLayer(64, N_TARGETS),
        )
        for p in self.parameters():
            if p.ndim >= 2:
                torch.nn.init.xavier_normal_(p)
        self.example_input_array = (
            torch.zeros((32, N_FEATURES)),
            torch.zeros((32, vae_latent_dim)),
        )
        self.forward(*self.example_input_array)

    def forward(
        self,
        x: Union[Tensor, Dict[str, Tensor]],
        z: Tensor,
        *_,
        **__,
    ):
        # One operation per line for easier troubleshooting
        if isinstance(x, dict):
            x = concat_tensor_dict(x)
        x = x.float().to(self.device)  # type: ignore
        z = z.to(self.device)  # type: ignore
        a = self._module_a(x)
        az = torch.concatenate([a, z], dim=-1)
        b = self._module_b(az)
        return b

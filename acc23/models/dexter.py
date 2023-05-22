"""
ACC23 main multi-classification model: prototype "Dexter". Works the same as
"Ampere" but encodes the source image using a pre-trained autoencoder first.
Therefore, there is no convolutional input branch. The dense input branch still
exists though.
"""
__docformat__ = "google"

from typing import Dict, Union

import torch
from torch import Tensor, nn
from traitlets import Any

from acc23.constants import N_FEATURES, N_TARGETS

from .layers import (
    ResNetLinearLayer,
    concat_tensor_dict,
)
from .base_mlc import BaseMultilabelClassifier


class Dexter(BaseMultilabelClassifier):
    """See module documentation"""

    _module_a: nn.Module  # Input dense branch
    _module_b: nn.Module  # Merge branch

    def __init__(
        self,
        ae_latent_dim: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self._module_a = nn.Sequential(
            ResNetLinearLayer(N_FEATURES, 256),
            ResNetLinearLayer(256, 512),
            ResNetLinearLayer(512, 512),
            ResNetLinearLayer(512, 512),
            ResNetLinearLayer(512, 512),
            ResNetLinearLayer(512, ae_latent_dim),
        )
        self._module_b = nn.Sequential(
            ResNetLinearLayer(2 * ae_latent_dim, ae_latent_dim),
            ResNetLinearLayer(ae_latent_dim, ae_latent_dim),
            ResNetLinearLayer(ae_latent_dim, ae_latent_dim),
            ResNetLinearLayer(ae_latent_dim, ae_latent_dim),
            ResNetLinearLayer(ae_latent_dim, 128),
            ResNetLinearLayer(128, 128),
            ResNetLinearLayer(128, 128),
            ResNetLinearLayer(128, 128),
            ResNetLinearLayer(128, 64),
            ResNetLinearLayer(64, N_TARGETS),
        )
        self.example_input_array = (
            torch.zeros((32, N_FEATURES)),
            torch.zeros((32, ae_latent_dim)),
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
        if isinstance(x, dict):
            x = concat_tensor_dict(x)
        x = x.float().to(self.device)  # type: ignore
        img = img.to(self.device)  # type: ignore
        a = self._module_a(x)
        ai = torch.concatenate([a, img], dim=-1)
        b = self._module_b(ai)
        return b

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

from acc23.constants import N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .layers import concat_tensor_dict, linear_chain


class Dexter(BaseMultilabelClassifier):
    """See module documentation"""

    tabular_branch: nn.Module  # Dense input branch
    main_branch: nn.Module  # Merge branch

    def __init__(self, ae_latent_dim: int = 256) -> None:
        super().__init__()
        self.save_hyperparameters()
        embed_dim, activation = 512, "gelu"
        self.tabular_branch = linear_chain(
            N_FEATURES,
            [256, 256, embed_dim],
            activation=activation,
        )
        self.main_branch = nn.Linear(embed_dim + ae_latent_dim, N_TRUE_TARGETS)
        self.example_input_array = (
            torch.zeros((32, N_FEATURES)),
            torch.zeros((32, ae_latent_dim)),
        )
        self.forward(*self.example_input_array)

    def forward(
        self,
        x: Union[Tensor, Dict[str, Tensor]],
        z: Tensor,
        *_,
        **__,
    ) -> Tensor:
        """
        Args:
            x (Tensor): Tabular data with shape `(N, N_FEATURES)`, where `N` is
                the batch size, or alternatively, a string dict, where each key
                is a `(N,)` tensor.
            z (Tensor): Batch of encoded images, i.e. a tensor of shape
                `(N, vae_latent_dim)`
        """
        if isinstance(x, dict):
            x = concat_tensor_dict(x)
        x = x.float().to(self.device)  # type: ignore
        z = z.to(self.device)  # type: ignore
        a = self.tabular_branch(x)
        az = torch.concatenate([a, z], dim=-1)
        b = self.main_branch(az)
        return b

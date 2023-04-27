"""
ACC23 main multi-classification model: prototype "Dexter". Works the same as
"Ampere" but encodes the source image using a pre-trained autoencoder first.
Therefore, there is no convolutional input branch. The dense input branch still
exists though.
"""
__docformat__ = "google"

from pathlib import Path
from typing import Dict, Union
from loguru import logger as logging

import torch
from torch import Tensor, nn
from traitlets import Any

from acc23.constants import IMAGE_RESIZE_TO, N_CHANNELS, N_FEATURES, N_TARGETS
from acc23.utils import last_checkpoint_path

from .autoencoder import Autoencoder
from .utils import (
    concat_tensor_dict,
    linear_chain,
)
from .base_mlc import BaseMultilabelClassifier


class Dexter(BaseMultilabelClassifier):
    """See module documentation"""

    _module_a: nn.Module  # Input dense branch
    _module_b: nn.Module  # Merge branch

    _autoencoder: Autoencoder

    def __init__(
        self,
        autoencoder_ckpt_path: Union[str, Path],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        logging.debug(
            "Loading autoencoder from checkpoint '{}'", autoencoder_ckpt_path
        )
        self._autoencoder = Autoencoder.load_from_checkpoint(
            autoencoder_ckpt_path
        )
        self._autoencoder.freeze()
        d = self._autoencoder.hparams["latent_space_dim"]
        logging.debug("Dexter's autoencoder latent space dimension: {}", d)
        self._module_a = linear_chain(N_FEATURES, [d], "relu")
        self._module_b = linear_chain(
            2 * d,
            [d, 128, 64, N_TARGETS],
            activation="relu",
            last_activation="sigmoid",
        )
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
        l = self._autoencoder.encode(img)
        al = torch.concatenate([a, l], dim=-1)
        b = self._module_b(al)
        return b

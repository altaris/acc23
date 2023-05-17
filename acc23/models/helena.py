"""
ACC23 main multi-classification model: prototype "Helena" aka TabVisionNet from

    Y. Liu, H. -P. Lu and C. -H. Lai, "A Novel Attention-Based Multi-Modal
    Modeling Technique on Mixed Type Data for Improving TFT-LCD Repair
    Process," in IEEE Access, vol. 10, pp. 33026-33036, 2022, doi:
    10.1109/ACCESS.2022.3158952.
"""
__docformat__ = "google"

from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .imagetabnet import ImageTabNetVisionEncoder
from .layers import concat_tensor_dict
from .tabnet import TabNetEncoder


class Helena(BaseMultilabelClassifier):
    """See module documentation"""

    tabular_encoder: nn.Module
    vision_encoder_a: nn.Module
    vision_encoder_b: nn.Module
    fusion_branch: nn.Module

    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        n_decision_steps, n_a, n_d, gamma = 3, 64, 64, 2
        n_encoded_features = n_d
        self.tabular_encoder = TabNetEncoder(
            N_FEATURES, n_encoded_features, n_a, n_d, n_decision_steps, gamma
        )
        self.vision_encoder_a = nn.Sequential(
            nn.MaxPool2d(5, 1, 2),  # IMAGE_RESIZE_TO = 512 -> 512
            nn.Conv2d(N_CHANNELS, 8, 4, 2, 1, bias=False),  # -> 256
            nn.BatchNorm2d(8),
            nn.SiLU(),
            # nn.MaxPool2d(3, 1, 1),  # 256 -> 256
        )
        self.vision_encoder_b = ImageTabNetVisionEncoder(
            in_channels=8,
            out_channels=[
                8,  # 256 -> 128
                16,  # -> 64
                16,  # -> 32
                32,  # -> 16
                32,  # -> 8
                64,  # -> 4
                128,  # -> 2
                n_encoded_features,  # -> 1
            ],
            n_d=n_d,
            n_decision_steps=n_decision_steps,
        )
        self.fusion_branch = nn.Sequential(
            nn.Linear(2 * n_encoded_features, N_TARGETS, bias=False),
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
        if isinstance(x, dict):
            x = concat_tensor_dict(x)
        x = x.float().to(self.device)  # type: ignore
        img = img.to(self.device)  # type: ignore
        u, _, sparse_loss, ds = self.tabular_encoder(x)
        v = self.vision_encoder_a(img)
        v = self.vision_encoder_b(v, ds)
        uv = torch.concatenate([u, v], dim=-1)  # TODO FIX ME!
        w = self.fusion_branch(uv)
        return w, sparse_loss * 1e-3

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

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .imagetabnet import VisionEncoder
from .layers import concat_tensor_dict
from .tabnet import TabNetEncoder


class Helena(BaseMultilabelClassifier):
    """See module documentation"""

    tabular_encoder: nn.Module
    vision_encoder_a: nn.Module
    vision_encoder_b: nn.Module
    fusion_branch: nn.Module
    sparse_loss_weight: float

    def __init__(
        self,
        n_a: int = 512,
        n_d: int = 512,
        n_decision_steps: int = 5,
        n_encoded_features: int = 256,
        gamma: float = 2.0,
        sparse_loss_weight: float = 5e-3,
    ) -> None:
        """
        Args:
            n_a (int): Dimension of the feature vector (to be passed down to
                the next decision step)
            n_d (int): Dimension of the output vector. The paper recommends
                `n_a = n_d`.
            n_decision_steps (int): The paper recommends between 3 and 10
            n_encoded_features (int): Dimension of an output vector from the
                tabular encoder. The vision encoder also produces encoded
                vectors of that dimension.
            gamma (float): Relaxation parameter for the attentive transformers.
                The paper recommends larger values if the number of decision
                steps is large.
            activation (str): Defaults to relu, as in the paper
        """
        super().__init__()
        self.save_hyperparameters()
        self.sparse_loss_weight = sparse_loss_weight
        self.tabular_encoder = TabNetEncoder(
            in_features=N_FEATURES,
            out_features=n_encoded_features,
            n_a=n_a,
            n_d=n_d,
            n_decision_steps=n_decision_steps,
            gamma=gamma,
        )
        self.vision_encoder_a = nn.Sequential(
            nn.MaxPool2d(5, 1, 2),  # IMAGE_RESIZE_TO = 512 -> 512
            nn.Conv2d(N_CHANNELS, 8, 4, 2, 1, bias=False),  # -> 256
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.vision_encoder_b = VisionEncoder(
            in_channels=8,
            out_channels=[
                # 8,  # 512 -> 256
                8,  # 256 -> 128
                16,  # -> 64
                16,  # -> 32
                32,  # -> 16
                32,  # -> 8
                64,  # -> 4
                128,  # -> 2
                n_encoded_features,  # -> 1
            ],
            in_features=n_encoded_features,
        )
        self.fusion_branch = nn.Sequential(
            nn.Linear(2 * n_encoded_features, N_TRUE_TARGETS, bias=False),
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
        u, _, sl = self.tabular_encoder(x)
        v = self.vision_encoder_a(img)
        v = self.vision_encoder_b(v, u)
        uv = torch.concatenate([u, v], dim=-1)
        w = self.fusion_branch(uv)
        return w, sl * self.sparse_loss_weight

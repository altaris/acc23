"""
ACC23 main multi-classification model: prototype "Gordon". Fusion model
**inspired** by

    Y. Liu, H. -P. Lu and C. -H. Lai, "A Novel Attention-Based Multi-Modal
    Modeling Technique on Mixed Type Data for Improving TFT-LCD Repair
    Process," in IEEE Access, vol. 10, pp. 33026-33036, 2022, doi:
    10.1109/ACCESS.2022.3158952.
"""
__docformat__ = "google"

from typing import Any, Dict, Union

import torch
from torch import Tensor, nn

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .layers import ResNetLinearLayer, concat_tensor_dict
from .imagetabnet import VisionEncoder


class Gordon(BaseMultilabelClassifier):
    """See module documentation"""

    tabular_branch: nn.Module  # Dense input branch
    vision_branch_a: nn.Module  # Conv. pool input branch
    vision_branch_b: nn.Module  # Conv fusion encoder
    main_branch: nn.Module  # Fusion branch

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        embed_dim, activation = 256, "gelu"
        self.tabular_branch = nn.Sequential(
            ResNetLinearLayer(N_FEATURES, 256, activation=activation),
            ResNetLinearLayer(256, embed_dim, activation=activation),
        )
        # self.vision_branch_a = nn.Sequential(
        #     nn.MaxPool2d(5, 1, 2),  # IMAGE_RESIZE_TO = 512 -> 512
        #     nn.Conv2d(N_CHANNELS, 8, 4, 2, 1, bias=False),  # -> 256
        #     nn.BatchNorm2d(8),
        #     get_activation(activation),
        #     # nn.MaxPool2d(3, 1, 1),  # 256 -> 256
        # )
        self.vision_branch_a = nn.Sequential(
            nn.MaxPool2d(5, 2, 2),  # IMAGE_RESIZE_TO = 512 -> 256
        )
        self.vision_branch_b = VisionEncoder(
            in_channels=N_CHANNELS,
            out_channels=[
                8,  # 256 -> 128
                16,  # -> 64
                16,  # -> 32
                32,  # -> 16
                32,  # -> 8
                64,  # -> 4
                128,  # -> 2
                embed_dim,  # -> 1
            ],
            in_features=embed_dim,
            activation=activation,
        )
        self.main_branch = nn.Sequential(
            nn.Linear(2 * embed_dim, N_TRUE_TARGETS),
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
        u = self.tabular_branch(x)
        v = self.vision_branch_a(img)
        v = self.vision_branch_b(v, u)
        uv = torch.concatenate([u, v], dim=-1)
        w = self.main_branch(uv)
        return w

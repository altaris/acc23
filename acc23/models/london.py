"""
ACC23 main multi-classification model: prototype "London". Like Ampere but the
convolutional branch is a vision transformer
"""
__docformat__ = "google"

from typing import Dict, Union

import torch
from torch import Tensor, nn

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .layers import concat_tensor_dict, linear_chain
from .vit import VisionTransformer


class London(BaseMultilabelClassifier):
    """See module documentation"""

    tabular_encoder: nn.Module  # Dense input branch
    vision_transformer: nn.Module  # Conv. input branch
    main_branch: nn.Module  # Merge branch

    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        embed_dim, activation = 256, "gelu"
        # patch_size, n_transformers, n_heads, dropout = 32, 8, 8, 0.2
        patch_size, n_transformers, n_heads, dropout = 16, 16, 8, 0.0
        # self.tabular_encoder = nn.Sequential(
        #     nn.Linear(N_FEATURES, 256),
        #     get_activation(activation),
        #     ResidualLinearBlock(256, activation=activation),
        #     ResidualLinearBlock(256, activation=activation),
        #     ResidualLinearBlock(256, activation=activation),
        #     nn.Linear(256, embed_dim),
        #     get_activation(activation),
        # )
        self.tabular_encoder = linear_chain(
            N_FEATURES,
            [512, embed_dim],
            activation=activation,
        )
        self.vision_transformer = nn.Sequential(
            nn.MaxPool2d(5, 2, 2),  # 512 -> 256
            VisionTransformer(
                patch_size=patch_size,
                input_shape=(N_CHANNELS, IMAGE_SIZE // 2, IMAGE_SIZE // 2),
                embed_dim=embed_dim,
                hidden_dim=2 * embed_dim,
                out_features=embed_dim,
                n_transformers=n_transformers,
                n_heads=n_heads,
                dropout=dropout,
                activation=activation,
            ),
        )
        # self.main_branch = nn.Sequential(
        #     nn.Linear(2 * embed_dim, 256),
        #     get_activation(activation),
        #     ResidualLinearBlock(256, activation=activation),
        #     ResidualLinearBlock(256, activation=activation),
        #     ResidualLinearBlock(256, activation=activation),
        #     nn.Linear(256, N_TRUE_TARGETS),
        # )
        self.main_branch = linear_chain(
            2 * embed_dim,
            [embed_dim, N_TRUE_TARGETS],
            activation=activation,
            last_activation="linear",
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
        # One operation per line for easier troubleshooting
        if isinstance(x, dict):
            x = concat_tensor_dict(x)
        x = x.float().to(self.device)  # type: ignore
        img = img.to(self.device)  # type: ignore
        a = self.tabular_encoder(x)
        b = self.vision_transformer(img)
        ab = torch.concatenate([a, b], dim=-1)
        c = self.main_branch(ab)
        return c

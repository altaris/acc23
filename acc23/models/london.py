"""
ACC23 main multi-classification model: prototype "London". Like Ampere but the
convolutional branch is a vision transformer
"""
__docformat__ = "google"

from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn
from transformers.activations import get_activation

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .layers import concat_tensor_dict, linear_chain
from .transformers import VisionTransformer


class London(BaseMultilabelClassifier):
    """See module documentation"""

    tabular_encoder: nn.Module  # Dense input branch
    vision_transformer: nn.Module  # Conv. input branch
    main_branch: nn.Module  # Merge branch

    def __init__(
        self,
        n_features: int = N_FEATURES,
        image_shape: Tuple[int, int, int] = (
            N_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        out_dim: int = N_TRUE_TARGETS,
        embed_dim: int = 64,
        patch_size: int = 16,
        n_transformers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        nc, s, _ = image_shape
        self.tabular_encoder = linear_chain(
            n_features,
            [embed_dim],
            activation=activation,
        )
        self.vision_transformer = nn.Sequential(
            nn.MaxPool2d(5, 2, 2),  # 512 -> 256
            VisionTransformer(
                patch_size=patch_size,
                # input_shape=(nc, s, s),
                input_shape=(nc, s // 2, s // 2),
                embed_dim=embed_dim,
                out_dim=embed_dim,
                num_transformers=n_transformers,
                num_heads=n_heads,
                dropout=0.1,
                activation=activation,
            ),
            # ViT(
            #     image_size=s,
            #     channels=nc,
            #     patch_size=patch_size,
            #     num_classes=embed_dim,
            #     dim=embed_dim,
            #     depth=n_transformers,
            #     heads=n_heads,
            #     mlp_dim=embed_dim,
            #     dropout=0,
            #     emb_dropout=0,
            # )
        )
        self.main_branch = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            get_activation(activation),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(embed_dim, out_dim),
        )
        # self.main_branch = linear_chain(
        #     2 * embed_dim,
        #     [embed_dim, out_dim],
        #     activation=activation,
        #     last_activation="linear",
        # )
        self.example_input_array = (
            torch.zeros((32, n_features)),
            torch.zeros((32, nc, s, s)),
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
        b = (
            self.vision_transformer(img)
            # if img.max() > 0
            # else torch.zeros_like(a)
        )
        ab = torch.concatenate([a, b], dim=-1)
        c = self.main_branch(ab)
        return c

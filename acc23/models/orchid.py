"""
ACC23 main multi-classification model: prototype "Orchid". Like Ampere but the
convolutional branch is a vision transformer and the tabular branch is a
`TabTransformer`
"""
__docformat__ = "google"

from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor, nn
from transformers.activations import get_activation

from acc23.constants import (
    IMAGE_SIZE,
    N_CHANNELS,
    N_FEATURES,
    N_TRUE_TARGETS,
    CLASSES,
    FEATURES,
)

from .base_mlc import BaseMultilabelClassifier
from .layers import concat_tensor_dict, linear_chain
from .transformers import VisionTransformer, TabTransformer


class Orchid(BaseMultilabelClassifier):
    """See module documentation"""

    tabular_transformer: nn.Module  # Dense input branch
    vision_transformer: nn.Module  # Conv. input branch
    main_branch: nn.Module  # Merge branch

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (
            N_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        out_dim: int = N_TRUE_TARGETS,
        embed_dim: int = 32,
        patch_size: int = 8,
        n_transformers: int = 32,
        n_heads: int = 16,
        dropout: float = 0.1,
        activation: str = "gelu",
        pooling: bool = False,
        mlp_dim: int = 2048,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        nc, s, _ = image_shape
        n_classes = {k: len(v) for k, v in CLASSES.items()}
        self.tabular_transformer = TabTransformer(
            n_num_features=N_FEATURES - sum(n_classes.values()),
            n_classes=n_classes,
            out_dim=embed_dim,
            embed_dim=embed_dim,
            n_transformers=n_transformers,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            mlp_dim=mlp_dim,
        )
        self.vision_transformer = nn.Sequential(
            nn.MaxPool2d(5, 2, 2) if pooling else nn.Identity(),
            VisionTransformer(
                patch_size=patch_size,
                input_shape=((nc, s // 2, s // 2) if pooling else (nc, s, s)),
                embed_dim=embed_dim,
                out_dim=embed_dim,
                num_transformers=n_transformers,
                num_heads=n_heads,
                dropout=dropout,
                activation=activation,
                mlp_dim=mlp_dim,
            ),
        )
        self.main_branch = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, 4 * embed_dim),
            get_activation(activation),
            nn.Linear(4 * embed_dim, out_dim),
        )
        self.example_input_array = (
            {k: torch.zeros(32) for k in FEATURES if k != "Chip_Image_Name"},
            torch.zeros((32, nc, s, s)),
        )
        self.forward(*self.example_input_array)

    def forward(
        self,
        x: Dict[str, Tensor],
        img: Tensor,
        *_,
        **__,
    ) -> Tensor:
        cat_columns = [k + "_" + c for k, v in CLASSES.items() for c in v]
        x_cat: Dict[str, Tensor] = {}
        for k, v in CLASSES.items():
            a = concat_tensor_dict({c: x[k + "_" + c] for c in v})
            x_cat[k] = a.float().to(self.device)  # type: ignore
        x_num = {k: v for k, v in x.items() if k not in cat_columns}
        x_num = concat_tensor_dict(x_num)
        x_num = x_num.float().to(self.device)  # type: ignore
        img = img.to(self.device)  # type: ignore
        a = self.tabular_transformer(x_cat, x_num)
        b = (
            self.vision_transformer(img)
            # if img.max() > 0
            # else torch.zeros_like(a)
        )
        ab = torch.concatenate([a, b], dim=-1)
        c = self.main_branch(ab)
        return c

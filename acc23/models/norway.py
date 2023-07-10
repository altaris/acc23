"""
ACC23 main multi-classification model: prototype "Norway". Like London, but the
vision transformer is replaced by a co-attention modal vision transformer.
"""
__docformat__ = "google"

from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier, to_hierarchical_logits
from .layers import MLP, concat_tensor_dict
from .transformers import CoAttentionVisionTransformer


class Norway(BaseMultilabelClassifier):
    """See module documentation"""

    tat: nn.Module  # Dense input branch
    pooling: nn.Module
    vit: nn.Module
    mlp_head: nn.Module  # Merge branch

    def __init__(
        self,
        n_features: int = N_FEATURES,
        image_shape: Tuple[int, int, int] = (
            N_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        out_dim: int = N_TRUE_TARGETS,
        embed_dim: int = 768,
        patch_size: int = 16,
        n_transformers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        activation: str = "gelu",
        pooling: bool = False,
        mlp_dim: int = 3072,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        nc, s, _ = image_shape
        self.tat = MLP(
            in_dim=n_features,
            hidden_dims=[mlp_dim, embed_dim],
            dropout=dropout,
            activation=activation,
            is_head=False,
        )
        self.pooling = nn.MaxPool2d(7, 2, 3) if pooling else nn.Identity()
        self.vit = CoAttentionVisionTransformer(
            patch_size=patch_size,
            input_shape=((nc, s // 2, s // 2) if pooling else (nc, s, s)),
            embed_dim=embed_dim,
            out_dim=embed_dim,
            num_transformers=n_transformers,
            num_heads=n_heads,
            dropout=0.0,
            activation=activation,
            headless=True,
        )
        self.mlp_head = MLP(
            in_dim=3 * embed_dim,
            hidden_dims=[mlp_dim, out_dim],
            dropout=dropout,
            activation=activation,
            is_head=True,
        )
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
        a, b = self.pooling(img), self.tat(x)
        a, c = self.vit(a, b)
        abc = torch.concatenate([a, b, c], dim=-1)
        abc = to_hierarchical_logits(abc, mode="max")
        return self.mlp_head(abc)

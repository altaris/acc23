"""ACC23 model prototype *Norway*"""

from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier, to_hierarchical_logits
from .layers import MLP, concat_tensor_dict
from .transformers import CoAttentionVisionTransformer


class Norway(BaseMultilabelClassifier):
    """
    A multimodal model that uses a
    `acc23.models.transformers.CoAttentionVisionTransformer`. In a nutshell,
    the tabular data is encoded by a MLP as usual. Then, the image and encoded
    tabular feature are fed side by side into a
    `acc23.models.transformers.CoAttentionVisionTransformer` which consists of
    two intertwined stacks of transformer encoder layers.
    """

    _tab_mlp: nn.Module  # Dense input branch
    _vit_pool: nn.Module
    _vit: nn.Module
    _mlp_head: nn.Module  # Merge branch

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
        """
        Args:
            n_features (int, optional): Number of numerical tabular features
            image_shape (Tuple[int, int, int], optional):
            out_dim (int, optional):
            embed_dim (int, optional):
            patch_size (int, optional):
            n_transformers (int, optional):
            n_heads (int, optional):
            dropout (float, optional):
            activation (str, optional):
            pooling (bool, optional):
            mlp_dim (int, optional):

        See also:
            `acc23.models.base_mlc.BaseMultilabelClassifier.__init__`
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        nc, s, _ = image_shape
        self._tab_mlp = MLP(
            in_dim=n_features,
            hidden_dims=[mlp_dim, embed_dim],
            dropout=dropout,
            activation=activation,
            is_head=False,
        )
        self._vit_pool = nn.MaxPool2d(7, 2, 3) if pooling else nn.Identity()
        self._vit = CoAttentionVisionTransformer(
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
        self._mlp_head = MLP(
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
        a, b = self._vit_pool(img), self._tab_mlp(x)
        a, c = self._vit(a, b)
        abc = torch.concatenate([a, b, c], dim=-1)
        abc = to_hierarchical_logits(abc, mode="max")
        return self._mlp_head(abc)

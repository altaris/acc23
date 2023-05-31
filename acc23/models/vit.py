"""
Vision transformer of

    Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers
    for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

See also:
    https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html
"""

from typing import Tuple

import torch
from torch import Tensor, nn


def make_patches(imgs: Tensor, patch_size: int) -> Tensor:
    """
    Args:
        imgs (Tensor): A batch of channel-first square images, so a tensor of
            shape `(N, C, H, W)` where `H = W`.
        patch_size (int): Should really divide `H`

    Returns:
        A tensor of shape `(N, K * K, C * patch_size * patch_size)`, where `K =
        H / patch_size` (or equivalently `W / patch_size`).
    """
    b, c, s, _ = imgs.shape
    k = s // patch_size
    imgs = imgs.reshape(b, c, k, patch_size, k, patch_size)
    imgs = imgs.permute(0, 2, 4, 1, 3, 5)  # (b, k, k, c, ps, ps)
    imgs = imgs.flatten(1, 2)  # (b, k * k, c, ps, ps)
    imgs = imgs.flatten(2, 4)  # (b, k * k, c * ps * ps)
    return imgs


# class TransformerEncoder(nn.Module):
#     """Transformer encoder"""

#     norm_1: nn.Module
#     norm_2: nn.Module
#     mha: nn.Module
#     mlp: nn.Module

#     def __init__(
#         self,
#         embed_dim: int,
#         hidden_dim: Optional[int] = None,
#         n_heads: int = 4,
#         dropout: float = 0.0,
#         activation: str = "gelu",
#     ) -> None:
#         super().__init__()
#         hidden_dim = hidden_dim or embed_dim
#         self.norm_1 = nn.LayerNorm(embed_dim)
#         self.norm_2 = nn.LayerNorm(embed_dim)
#         self.mha = nn.MultiheadAttention(embed_dim, n_heads, dropout)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             get_activation(activation),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, embed_dim),
#             nn.Dropout(dropout),
#         )

#     # pylint: disable=missing-function-docstring
#     def forward(self, x: Tensor) -> Tensor:
#         a = self.norm_1(x)
#         b = x + self.mha(a, a, a)[0]
#         c = self.norm_2(b)
#         d = x + self.mlp(c)
#         return d


class VisionTransformer(nn.Module):
    """Vision transformer"""

    patch_size: int
    projection: nn.Module
    transformers: nn.Module
    mlp_head: nn.Module
    dropout: nn.Module

    class_token: nn.Parameter
    pos_embed: nn.Parameter

    def __init__(
        self,
        patch_size: int,
        input_shape: Tuple[int, int, int],
        embed_dim: int,
        hidden_dim: int,
        out_features: int,
        n_transformers: int,
        n_heads: int,
        dropout: float = 0.2,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        c, s, _ = input_shape
        np = (s // patch_size) ** 2
        self.patch_size = patch_size
        self.projection = nn.Linear(
            c * patch_size * patch_size,
            embed_dim,
            bias=False,
        )
        self.transformers = nn.Sequential(
            *[
                # TransformerEncoder(
                #     embed_dim=embed_dim,
                #     hidden_dim=hidden_dim,
                #     n_heads=n_heads,
                #     dropout=dropout,
                #     activation=activation,
                # )
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim,
                    activation=activation,
                )
                for _ in range(n_transformers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_features),
        )
        self.dropout = nn.Dropout(dropout)
        self.class_token = nn.Parameter(torch.randn(embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(np + 1, embed_dim))

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor) -> Tensor:
        x = make_patches(x, self.patch_size)  # (b, np, c * ps * ps)
        # Layers only act on last dim(s) =)
        x = self.projection(x)  # (b, np, ed)
        ct = self.class_token.repeat(x.shape[0], 1, 1)  # (b, 1, ed)
        z = torch.concat([ct, x], dim=1)  # (b, np + 1, ed)
        z = z + self.pos_embed  # (b, np + 1, ed)
        z = self.dropout(z)
        z = z.transpose(0, 1)  # (np + 1, b, ed)
        z = self.transformers(z)
        z = z[0]  # (b, ed)
        y = self.mlp_head(z)
        return y

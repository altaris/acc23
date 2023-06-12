"""
Attention & transformer related modules
"""

from typing import Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from acc23.models.layers import linear_chain


def make_patches(imgs: Tensor, patch_size: int) -> Tuple[Tensor, int, int]:
    """
    Args:
        imgs (Tensor): A batch of channel-first square images, so a tensor of
            shape `(N, C, H, W)` where `H = W`.
        patch_size (int): Should really divide `H`

    Returns:
        * a tensor of shape `(N, K * K, C * patch_size * patch_size)`, where `K
          = H / patch_size` (or equivalently `W / patch_size`).
        * the value of `K * K`, i.e. the number of patches
        * the value of `C * patch_size * patch_size` i.e. the number of
          elements per patch
    """
    b, c, s, _ = imgs.shape
    k = s // patch_size
    imgs = imgs.reshape(b, c, k, patch_size, k, patch_size)
    imgs = imgs.permute(0, 2, 4, 1, 3, 5)  # (b, k, k, c, ps, ps)
    imgs = imgs.flatten(1, 2)  # (b, k * k, c, ps, ps)
    imgs = imgs.flatten(2, 4)  # (b, k * k, c * ps * ps)
    return imgs, k * k, c * patch_size * patch_size


class CrossModalTransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer that uses cross-modal multihead attention,
    which combines two token streams into one. See also figure 2 of

        P. Xu, X. Zhu and D. A. Clifton, "Multimodal Learning With
        Transformers: A Survey," in IEEE Transactions on Pattern Analysis and
        Machine Intelligence, doi: 10.1109/TPAMI.2023.3275156.
    """

    method: Literal["a", "b", "c", "f"]
    mhas: nn.ModuleList
    mlp: nn.Module
    norm: nn.Module

    def __init__(
        self,
        method: Literal["a", "b", "c", "f"],
        embed_dim: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        """
        The `method` argument specifies how the two input streams are combined,
        following the naming of figure 2 of

            P. Xu, X. Zhu and D. A. Clifton, "Multimodal Learning With
            Transformers: A Survey," in IEEE Transactions on Pattern Analysis
            and Machine Intelligence, doi: 10.1109/TPAMI.2023.3275156.

        which is to say:
        * `a`: early summation, uses 1 attention module;
        * `b`: early concatenation, uses 1 attention module;
        * `c`: hierarchical (multi-stream to one), uses 3 attention modules;
        * `f`: cross-attention to concatenation, uses 3 attention modules.

        In the case of method `a`, a skip connection over the MLP part is
        added. Each attention module is a
        `acc23.models.transformer.ResidualMultiheadSelfAttention`, which is a
        regular multihead self attention with an early layer normalization and
        a skip connection. In the case of methods `c` and `f`, an additional
        skip connection that goes over the whole attention stack is added.

        Args:
            method (str): See above
            embed_dim (int): The embedding dimension of both input streams.
            hidden_dim (int): Hidden dimension of the feed forward part of the
                transformer. Defaults to `embed_dim`
            num_heads (int): Passed to all attention modules
            dropout (float): Passed to all attention modules
        """
        super().__init__()
        self.method = method
        hidden_dim = hidden_dim or embed_dim
        if method == "a":
            self.mhas = nn.ModuleList(
                [
                    ResidualMultiheadAttention(
                        embed_dim, num_heads, dropout=dropout
                    )
                ]
            )
            self.norm = nn.LayerNorm(embed_dim)
            self.mlp = linear_chain(
                embed_dim, [hidden_dim, embed_dim], activation
            )
        elif method == "b":
            self.mhas = nn.ModuleList(
                [
                    ResidualMultiheadAttention(
                        2 * embed_dim, num_heads, dropout=dropout
                    )
                ]
            )
            self.norm = nn.LayerNorm(2 * embed_dim)
            self.mlp = linear_chain(
                2 * embed_dim, [hidden_dim, embed_dim], activation
            )
        elif method in ["c", "f"]:
            self.mhas = nn.ModuleList(
                [
                    ResidualMultiheadAttention(
                        embed_dim, num_heads, dropout=dropout
                    ),
                    ResidualMultiheadAttention(
                        embed_dim, num_heads, dropout=dropout
                    ),
                    ResidualMultiheadAttention(
                        2 * embed_dim, num_heads, dropout=dropout
                    ),
                ]
            )
            self.norm = nn.LayerNorm(2 * embed_dim)
            self.mlp = linear_chain(
                2 * embed_dim, [hidden_dim, embed_dim], activation
            )
        else:
            raise ValueError(
                f"Invalid cross modal attention method '{method}'. Refer to "
                "the documentation of "
                "acc23.models.transformers.CrossModalMultiHeadAttention "
                "for more details."
            )

    def _forward_a(self, u: Tensor, v: Tensor, *_, **__) -> Tensor:
        """
        Forward if the attention part follows cross-modal attention method a,
        aka. early summation.
        """
        uv, mha = u + v, self.mhas[0]
        w = mha(uv, uv, uv)
        return self.mlp(self.norm(w)) + w

    def _forward_b(self, u: Tensor, v: Tensor, *_, **__) -> Tensor:
        """
        Forward if the attention part follows cross-modal attention method b,
        aka. early concatenation.
        """
        uv, mha = torch.concat([u, v], dim=-1), self.mhas[0]
        w = mha(uv, uv, uv)
        return self.mlp(self.norm(w))

    def _forward_c(self, u: Tensor, v: Tensor, *_, **__) -> Tensor:
        """
        Forward if the attention part follows cross-modal attention method c,
        aka. hierarchical (multi-stream to one).
        """
        mha_u, mha_v, mha_ab = self.mhas[0], self.mhas[1], self.mhas[2]
        a, b = mha_u(u, u, u), mha_v(v, v, v)
        uv, ab = torch.concat([u, v], dim=-1), torch.concat([a, b], dim=-1)
        w = mha_ab(ab, ab, ab) + uv
        return self.mlp(self.norm(w))

    def _forward_f(self, u: Tensor, v: Tensor, *_, **__) -> Tensor:
        """
        Forward if the attention part follows cross-modal attention method f,
        aka. cross-attention to concatenation.
        """
        mha_u, mha_v, mha_ab = self.mhas[0], self.mhas[1], self.mhas[2]
        a, b = mha_u(v, u, u), mha_v(u, v, v)
        uv, ab = torch.concat([u, v], dim=-1), torch.concat([a, b], dim=-1)
        w = mha_ab(ab, ab, ab) + uv
        return self.mlp(self.norm(w))

    def forward(self, u: Tensor, v: Tensor, *_, **__) -> Tensor:
        """
        Args:
            u (Tensor): Must have shape `(L, B, E)`, where `L` is the sequence
            length, `B` is the batch size, and `E` is the embedding dimension
            v (Tensor): Must have the same shape as `u`

        Returns:
            A tensor with shape `(L, B, E)`.
        """
        if self.method == "a":
            return self._forward_a(u, v)
        if self.method == "b":
            return self._forward_b(u, v)
        if self.method == "c":
            return self._forward_c(u, v)
        return self._forward_f(u, v)


class CrossModalVisionTransformer(nn.Module):
    """
    Vision transformer of

        Dosovitskiy, Alexey, et al. "An image is worth 16x16 words:
        Transformers for image recognition at scale." arXiv preprint
        arXiv:2010.11929 (2020).

    except the transformer encoder layers are replaced by
    `CrossModalTransformerEncoderLayer`'s. In particular, this module takes two
    inputs, the image and an embedded (batch of) vector.

    See also:
        https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html
    """

    patch_size: int
    projection: nn.Module
    transformers: nn.ModuleList
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
        num_transformers: int,
        num_heads: int,
        method: Literal["a", "b", "c", "f"] = "a",
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        """
        Args:
            patch_size (int):
            input_shape (Tuple[int, int, int]):
            embed_dim (int):
            hidden_dim (int):
            out_features (int):
            num_transformers (int):
            num_heads (int):
            method (Literal["a", "b", "c", "f"]): Cross modal transformer
                method (see `CrossModalTransformerEncoderLayer`). Defaults to
                `a`
            dropout (float): Defaults to `0.0`
            activation (str): Defaults to `gelu`
        """
        super().__init__()
        c, s, _ = input_shape
        np = (s // patch_size) ** 2
        self.patch_size = patch_size
        self.projection = nn.Linear(
            c * patch_size * patch_size,
            embed_dim,
            bias=False,
        )
        self.transformers = nn.ModuleList(
            [
                CrossModalTransformerEncoderLayer(
                    method=method,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_transformers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_features),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.class_token = nn.Parameter(torch.randn(embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(np + 1, embed_dim))

    # pylint: disable=missing-function-docstring
    def forward(self, img: Tensor, u: Tensor) -> Tensor:
        x, n_patches, _ = make_patches(img, self.patch_size)  # (b, np, N)
        # Layers only act on last dim(s) =)
        x = self.projection(x)  # (b, np, ed)
        ct = self.class_token.repeat(x.shape[0], 1, 1)  # (b, 1, ed)
        z = torch.concat([ct, x], dim=1)  # (b, np + 1, ed)
        z = z + self.pos_embed  # (b, np + 1, ed)
        z = self.dropout(z)
        z = z.transpose(0, 1)  # (np + 1, b, ed)
        uu = u.repeat(n_patches + 1, 1, 1)
        for transformer in self.transformers:
            z = transformer(z, uu)
        z = z[0]  # (b, ed)
        y = self.mlp_head(z)
        return y


class ResidualMultiheadAttention(nn.Module):
    """
    A multihead self-attention with early layer normalization and an overall
    skip connection. Mathematically:
    $$Z = \\mathrm{MHA}(q', k', v') + \\frac{q + k + v}{3}$$
    where $\\mathrm{MHA}$ is the multihead attention module, where $q'$ is the
    layer normalization of $q$, and likewise for $k$ and $v$
    """

    mha: nn.Module
    norm_k: nn.Module
    norm_q: nn.Module
    norm_v: nn.Module

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, **kwargs)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)

    # pylint: disable=missing-function-docstring
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        a, _ = self.mha(
            self.norm_q(query),
            self.norm_k(key),
            self.norm_v(value),
            need_weights=False,
        )
        b = (query + key + value) / 3
        return a + b


class VisionTransformer(nn.Module):
    """
    Vision transformer of

        Dosovitskiy, Alexey, et al. "An image is worth 16x16 words:
        Transformers for image recognition at scale." arXiv preprint
        arXiv:2010.11929 (2020).

    See also:
        https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html
    """

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
        out_dim: int,
        num_transformers: int,
        num_heads: int,
        mlp_dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """
        Args:
            patch_size (int):
            input_shape (Tuple[int, int, int]):
            embed_dim (int):
            out_dim (int):
            num_transformers (int):
            num_heads (int):
            method (Literal["a", "b", "c", "f"]): Cross modal transformer
                method (see `CrossModalTransformerEncoderLayer`). Defaults to
                `a`
            dropout (float): Defaults to `0.0`
            activation (str): Defaults to `gelu`
        """
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
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim,
                    activation=activation,
                    dropout=0.0,
                )
                for _ in range(num_transformers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
        self.dropout = (
            nn.Dropout(mlp_dropout) if mlp_dropout > 0 else nn.Identity()
        )
        self.class_token = nn.Parameter(torch.randn(embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(np + 1, embed_dim))

    # pylint: disable=missing-function-docstring
    def forward(self, img: Tensor) -> Tensor:
        x, _, _ = make_patches(img, self.patch_size)  # (b, np, c * ps * ps)
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

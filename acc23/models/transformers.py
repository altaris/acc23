"""
Attention & transformer related modules
"""

from typing import Dict, List, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers.activations import get_activation

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


class CoAttentionTransformerEncoderLayer(nn.Module):
    """
    Co-attention mechanism of

        Lu, Jiasen, et al. "Vilbert: Pretraining task-agnostic visiolinguistic
        representations for vision-and-language tasks." Advances in neural
        information processing systems 32 (2019).

    """

    mhas: nn.ModuleList
    norms: nn.ModuleList
    mlps: nn.ModuleList
    last: bool

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        last: bool = False,
    ) -> None:
        """
        Args:
            embed_dim (int):
            num_heads (int):
            mlp_dim (int):
            dropout (float):
            activation (str):
            last (bool): If set to `True`, the right/second/`w` branch is just
                the identity
        """
        super().__init__()
        self.mhas = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim, num_heads, dropout)
                for _ in range(1 if last else 2)  # One module if last, 2 o.w.
            ],
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(embed_dim)
                for _ in range(1 if last else 2)  # One module if last, 2 o.w.
            ],
        )
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, mlp_dim),
                    get_activation(activation),
                    nn.Linear(mlp_dim, embed_dim),
                    get_activation(activation),
                    nn.Dropout(dropout),
                )
                for _ in range(1 if last else 2)  # One module if last, 2 o.w.
            ],
        )
        self.last = last

    # pylint: disable=missing-function-docstring
    def forward(self, v: Tensor, w: Tensor) -> Tuple[Tensor, Tensor]:
        if self.last:
            mha_v, norm_v, norm_w = self.mhas[0], self.norms[0], self.norms[1]
            mlp_v = self.mlps[0]
            a, b = norm_v(v), norm_w(w)
            c, _ = mha_v(query=a, key=b, value=b, need_weights=False)
            v = c + v
            v = mlp_v(v) + v
        else:
            mha_v, mha_w = self.mhas[0], self.mhas[1]
            norm_v, norm_w = self.norms[0], self.norms[1]
            mlp_v, mlp_w = self.mlps[0], self.mlps[1]
            a, b = norm_v(v), norm_w(w)
            c, _ = mha_v(query=a, key=b, value=b, need_weights=False)
            d, _ = mha_w(query=b, key=a, value=a, need_weights=False)
            v, w = c + v, d + w
            v, w = mlp_v(v) + v, mlp_w(w) + w
        return v, w


class CoAttentionVisionTransformer(nn.Module):
    """
    Multimodal vision transformer using the co-attention mechanism of

        Lu, Jiasen, et al. "Vilbert: Pretraining task-agnostic visiolinguistic
        representations for vision-and-language tasks." Advances in neural
        information processing systems 32 (2019).
    """

    patch_size: int
    projection: nn.Module
    transformers: nn.ModuleList
    mlp_head: Optional[nn.Module] = None
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
        dropout: float = 0.1,
        activation: str = "gelu",
        headless: bool = False,
    ) -> None:
        """
        Args:
            patch_size (int):
            input_shape (Tuple[int, int, int]):
            embed_dim (int):
            out_dim (int):
            num_transformers (int):
            num_heads (int):
            dropout (float): Defaults to `0.1`
            activation (str): Defaults to `gelu`
            headless (bool):
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
                CoAttentionTransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=2 * embed_dim,
                    dropout=dropout,
                    activation=activation,
                    last=(not headless and i == num_transformers - 1),
                )
                for i in range(num_transformers)
            ]
        )
        if not headless:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, out_dim),
            )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.class_token = nn.Parameter(torch.randn(embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(np + 1, embed_dim))

    # pylint: disable=missing-function-docstring
    def forward(self, img: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
        x, n_patches, _ = make_patches(img, self.patch_size)  # (b, np, N)
        # Layers only act on last dim(s) =)
        x = self.projection(x)  # (b, np, ed)
        ct = self.class_token.repeat(x.shape[0], 1, 1)  # (b, 1, ed)
        z = torch.concat([ct, x], dim=1)  # (b, np + 1, ed)
        z = z + self.pos_embed  # (b, np + 1, ed)
        z = self.dropout(z)
        z = z.transpose(0, 1)  # (np + 1, b, ed)
        u = u.repeat(n_patches + 1, 1, 1)
        for l in self.transformers:
            z, u = l(z, u)
        z, u = z[0], u[0]  # (b, ed)
        if self.mlp_head is not None:
            z = self.mlp_head(z)
        return z, u


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
    norms: nn.ModuleList

    def __init__(
        self,
        method: Literal["a", "b", "c", "f"],
        embed_dim: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
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
        hidden_dim = hidden_dim or 2 * embed_dim
        if method == "a":
            self.mhas = nn.ModuleList(
                [nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)]
            )
            self.norms = nn.ModuleList(
                [
                    nn.LayerNorm(embed_dim),
                    nn.LayerNorm(embed_dim),
                ]
            )
            self.mlp = linear_chain(
                embed_dim, [hidden_dim, embed_dim], activation
            )
        elif method == "b":
            self.mhas = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        2 * embed_dim, num_heads, dropout=dropout
                    )
                ]
            )
            self.norms = nn.ModuleList(
                [
                    nn.LayerNorm(2 * embed_dim),
                    nn.LayerNorm(2 * embed_dim),
                ]
            )
            self.mlp = linear_chain(
                2 * embed_dim, [hidden_dim, embed_dim], activation
            )
        elif method in ["c", "f"]:
            self.mhas = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim, num_heads, dropout=dropout
                    ),
                    nn.MultiheadAttention(
                        embed_dim, num_heads, dropout=dropout
                    ),
                    nn.MultiheadAttention(
                        2 * embed_dim, num_heads, dropout=dropout
                    ),
                ]
            )
            self.norms = self.norms = nn.ModuleList(
                [
                    nn.LayerNorm(embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.LayerNorm(2 * embed_dim),
                    nn.LayerNorm(2 * embed_dim),
                ]
            )
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
        a = u + v
        b = self.norms[0](a)
        c = self.mhas[0](b, b, b)[0] + a
        return self.mlp(self.norms[1](c)) + c

    def _forward_b(self, u: Tensor, v: Tensor, *_, **__) -> Tensor:
        """
        Forward if the attention part follows cross-modal attention method b,
        aka. early concatenation.
        """
        a = torch.concat([u, v], dim=-1)
        b = self.norms[0](a)
        c = self.mhas[0](b, b, b)[0] + a
        return self.mlp(self.norms[1](c))

    def _forward_c(self, u: Tensor, v: Tensor, *_, **__) -> Tensor:
        """
        Forward if the attention part follows cross-modal attention method c,
        aka. hierarchical (multi-stream to one).
        """
        a, b = self.norms[0](u), self.norms[1](v)
        c, d = self.mhas[0](a, a, a)[0] + u, self.mhas[1](b, b, b)[0] + v
        e = torch.concat([c, d], dim=-1)
        f = self.norms[2](e)
        g = self.mhas[2](f, f, f)[0] + e
        return self.mlp(self.norms[3](g))

    def _forward_f(self, u: Tensor, v: Tensor, *_, **__) -> Tensor:
        """
        Forward if the attention part follows cross-modal attention method f,
        aka. cross-attention to concatenation.
        """
        a, b = self.norms[0](u), self.norms[1](v)
        c, d = self.mhas[0](b, a, a)[0] + u, self.mhas[1](a, b, b)[0] + v
        e = torch.concat([c, d], dim=-1)
        f = self.norms[2](e)
        g = self.mhas[2](f, f, f)[0] + e
        return self.mlp(self.norms[3](g))

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
        nc, s, _ = input_shape
        np = (s // patch_size) ** 2
        self.patch_size = patch_size
        self.projection = nn.Linear(
            nc * patch_size * patch_size,
            embed_dim,
            bias=False,
        )
        self.transformers = nn.ModuleList(
            [
                CrossModalTransformerEncoderLayer(
                    method=method,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
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
        for transformer in self.transformers:
            z = transformer(z, u.repeat(n_patches + 1, 1, 1))
        z = z[0]  # (b, ed)
        y = self.mlp_head(z)
        return y


class TabTransformer(nn.Module):
    """
    TabTransformer from

        Huang, Xin, et al. "Tabtransformer: Tabular data modeling using
        contextual embeddings." arXiv preprint arXiv:2012.06678 (2020).

    """

    col_embed: nn.ModuleDict
    mlp: nn.Module
    transformers: nn.Module
    norm: nn.Module

    def __init__(
        self,
        n_num_features: int,
        n_classes: Dict[str, int],  # Un-one-hot stuff :/
        out_dim: int,
        embed_dim: int = 32,
        n_transformers: int = 16,
        n_heads: int = 8,
        dropout: float = 0.25,
        activation: str = "gelu",
        mlp_dim: int = 2048,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.col_embed = nn.ModuleDict(
            {
                k: nn.Linear(n, embed_dim, bias=False)
                for k, n in n_classes.items()
            }
        )
        self.transformers = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dropout=dropout,
                    activation=activation,
                    dim_feedforward=mlp_dim,
                )
                for _ in range(n_transformers)
            ]
        )
        self.norm = nn.LayerNorm(n_num_features)
        self.mlp = nn.Sequential(
            nn.Linear(len(n_classes) * embed_dim + n_num_features, mlp_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, out_dim),
            get_activation(activation),
        )

    # pylint: disable=missing-function-docstring
    def forward(self, x_cat: Dict[str, Tensor], x_num: Tensor) -> Tensor:
        embs: List[Tensor] = []
        for k, v in x_cat.items():
            w = self.col_embed[k](v)
            w = w.unsqueeze(0)  # for the cat later
            embs.append(w)
        u = torch.cat(embs, dim=0)
        u = self.transformers(u)
        u = u.permute(1, 0, 2)
        u = u.reshape(u.shape[0], -1)
        v = torch.cat([u, self.norm(x_num)], dim=-1)
        w = self.mlp(v)
        return w


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
        dropout: float = 0.1,
        activation: str = "gelu",
        mlp_dim: int = 2048,
    ) -> None:
        """
        Args:
            patch_size (int):
            input_shape (Tuple[int, int, int]):
            embed_dim (int):
            out_dim (int):
            num_transformers (int):
            num_heads (int):
            dropout (float): Defaults to `0.1`
            activation (str): Defaults to `gelu`
            mlp_dim (int):
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
                    dim_feedforward=mlp_dim,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(num_transformers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
            get_activation(activation),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
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

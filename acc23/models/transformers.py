"""
Attention & transformer related modules
"""

from typing import Dict, List, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers.activations import get_activation

from acc23.models.layers import MLP


def make_patches(imgs: Tensor, patch_size: int) -> Tuple[Tensor, int, int]:
    """
    Args:
        imgs (Tensor): A batch of channel-first square images, so a tensor of
            shape `(N, C, H, W)` where `H = W`.
        patch_size (int): Should really divide `H`

    Returns:
        - a tensor of shape `(N, K * K, C * patch_size * patch_size)`, where `K
          = H / patch_size` (or equivalently `W / patch_size`).
        - the value of `K * K`, i.e. the number of patches
        - the value of `C * patch_size * patch_size` i.e. the number of
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

    See `acc23.models.transformers.CoAttentionVisionTransformer`
    """

    _mhas: nn.ModuleList
    _norms: nn.ModuleList
    _mlps: nn.ModuleList
    _last: bool

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
            last (bool): If set to `True`, the right (aka second, aka `w`)
                branch is just the identity
        """
        super().__init__()
        self._mhas = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim, num_heads, dropout)
                for _ in range(1 if last else 2)  # One module if last, 2 o.w.
            ],
        )
        self._norms = nn.ModuleList(
            [
                nn.LayerNorm(embed_dim)
                for _ in range(1 if last else 2)  # One module if last, 2 o.w.
            ],
        )
        self._mlps = nn.ModuleList(
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
        self._last = last

    def forward(self, v: Tensor, w: Tensor) -> Tuple[Tensor, Tensor]:
        """Override"""
        if self._last:
            mha_v, norm_v, norm_w = (
                self._mhas[0],
                self._norms[0],
                self._norms[1],
            )
            mlp_v = self._mlps[0]
            a, b = norm_v(v), norm_w(w)
            c, _ = mha_v(query=a, key=b, value=b, need_weights=False)
            v = c + v
            v = mlp_v(v) + v
        else:
            mha_v, mha_w = self._mhas[0], self._mhas[1]
            norm_v, norm_w = self._norms[0], self._norms[1]
            mlp_v, mlp_w = self._mlps[0], self._mlps[1]
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

    In a nutshell, whereas a vision transformer contanins a stack of
    transformer encoder layers, a co-attention vision transfomer is a
    multimodal module that contains two intertwined stacks of transformer
    encoder layers.
    """

    _patch_size: int
    _proj: nn.Module
    _transformers: nn.ModuleList
    _mlp_head: Optional[nn.Module] = None
    _dropout: nn.Module

    _cls_token: nn.Parameter
    _pos_embed: nn.Parameter

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
            dropout (float):
            activation (str):
            headless (bool): If `True`, there is no final MLP head
        """
        super().__init__()
        c, s, _ = input_shape
        np = (s // patch_size) ** 2
        self._patch_size = patch_size
        self._proj = nn.Linear(
            c * patch_size * patch_size,
            embed_dim,
            bias=False,
        )
        self._transformers = nn.ModuleList(
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
            self._mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, out_dim),
            )
        self._dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._cls_token = nn.Parameter(torch.randn(embed_dim))
        self._pos_embed = nn.Parameter(torch.randn(np + 1, embed_dim))

    def forward(self, img: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
        """Override"""
        x, n_patches, _ = make_patches(img, self._patch_size)  # (b, np, N)
        # Layers only act on last dim(s) =)
        x = self._proj(x)  # (b, np, ed)
        ct = self._cls_token.repeat(x.shape[0], 1, 1)  # (b, 1, ed)
        z = torch.concat([ct, x], dim=1)  # (b, np + 1, ed)
        z = z + self._pos_embed  # (b, np + 1, ed)
        z = self._dropout(z)
        z = z.transpose(0, 1)  # (np + 1, b, ed)
        u = u.repeat(n_patches + 1, 1, 1)
        for l in self._transformers:
            z, u = l(z, u)
        z, u = z[0], u[0]  # (b, ed)
        if self._mlp_head is not None:
            z = self._mlp_head(z)
        return z, u


class CrossModalTransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer that uses cross-modal multihead attention,
    which combines two token streams into one. See also figure 2 of

        P. Xu, X. Zhu and D. A. Clifton, "Multimodal Learning With
        Transformers: A Survey," in IEEE Transactions on Pattern Analysis and
        Machine Intelligence, doi: 10.1109/TPAMI.2023.3275156.
    """

    _method: Literal["a", "b", "c", "f"]
    _mhas: nn.ModuleList
    _mlp: nn.Module
    _norms: nn.ModuleList

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
        following the naming of figure 2 of the paper, which is to say:
        - `a`: early summation, uses 1 attention module;
        - `b`: early concatenation, uses 1 attention module;
        - `c`: hierarchical (multi-stream to one), uses 3 attention modules;
        - `f`: cross-attention to concatenation, uses 3 attention modules.

        Args:
            method (Literal["a", "b", "c", "f"]): See above
            embed_dim (int): The embedding dimension of both input streams.
            hidden_dim (int): Hidden dimension of the feed forward part of the
                transformer. Defaults to `embed_dim`
            num_heads (int): Passed to all attention modules
            dropout (float): Passed to all attention modules
        """
        super().__init__()
        self._method = method
        hidden_dim = hidden_dim or 2 * embed_dim
        if method == "a":
            self._mhas = nn.ModuleList(
                [nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)]
            )
            self._norms = nn.ModuleList(
                [
                    nn.LayerNorm(embed_dim),
                    nn.LayerNorm(embed_dim),
                ]
            )
            self._mlp = MLP(
                embed_dim, [hidden_dim, embed_dim], activation=activation
            )
        elif method == "b":
            self._mhas = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        2 * embed_dim, num_heads, dropout=dropout
                    )
                ]
            )
            self._norms = nn.ModuleList(
                [
                    nn.LayerNorm(2 * embed_dim),
                    nn.LayerNorm(2 * embed_dim),
                ]
            )
            self._mlp = MLP(
                2 * embed_dim, [hidden_dim, embed_dim], activation=activation
            )
        elif method in ["c", "f"]:
            self._mhas = nn.ModuleList(
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
            self._norms = self._norms = nn.ModuleList(
                [
                    nn.LayerNorm(embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.LayerNorm(2 * embed_dim),
                    nn.LayerNorm(2 * embed_dim),
                ]
            )
            self._mlp = MLP(
                2 * embed_dim, [hidden_dim, embed_dim], activation=activation
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
        b = self._norms[0](a)
        c = self._mhas[0](b, b, b)[0] + a
        return self._mlp(self._norms[1](c)) + c

    def _forward_b(self, u: Tensor, v: Tensor, *_, **__) -> Tensor:
        """
        Forward if the attention part follows cross-modal attention method b,
        aka. early concatenation.
        """
        a = torch.concat([u, v], dim=-1)
        b = self._norms[0](a)
        c = self._mhas[0](b, b, b)[0] + a
        return self._mlp(self._norms[1](c))

    def _forward_c(self, u: Tensor, v: Tensor, *_, **__) -> Tensor:
        """
        Forward if the attention part follows cross-modal attention method c,
        aka. hierarchical (multi-stream to one).
        """
        a, b = self._norms[0](u), self._norms[1](v)
        c, d = self._mhas[0](a, a, a)[0] + u, self._mhas[1](b, b, b)[0] + v
        e = torch.concat([c, d], dim=-1)
        f = self._norms[2](e)
        g = self._mhas[2](f, f, f)[0] + e
        return self._mlp(self._norms[3](g))

    def _forward_f(self, u: Tensor, v: Tensor, *_, **__) -> Tensor:
        """
        Forward if the attention part follows cross-modal attention method f,
        aka. cross-attention to concatenation.
        """
        a, b = self._norms[0](u), self._norms[1](v)
        c, d = self._mhas[0](b, a, a)[0] + u, self._mhas[1](a, b, b)[0] + v
        e = torch.concat([c, d], dim=-1)
        f = self._norms[2](e)
        g = self._mhas[2](f, f, f)[0] + e
        return self._mlp(self._norms[3](g))

    def forward(self, u: Tensor, v: Tensor, *_, **__) -> Tensor:
        """
        Args:
            u (Tensor): Must have shape `(L, B, E)`, where `L` is the sequence
            length, `B` is the batch size, and `E` is the embedding dimension
            v (Tensor): Must have the same shape as `u`

        Returns:
            A tensor with shape `(L, B, E)`.
        """
        if self._method == "a":
            return self._forward_a(u, v)
        if self._method == "b":
            return self._forward_b(u, v)
        if self._method == "c":
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

    _patch_size: int
    _projs: nn.Module
    _transformers: nn.ModuleList
    _mlp_head: nn.Module
    _dropout: nn.Module

    _cls_token: nn.Parameter
    _pos_embed: nn.Parameter

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
                method, see `CrossModalTransformerEncoderLayer`
            dropout (float):
            activation (str):
        """
        super().__init__()
        nc, s, _ = input_shape
        np = (s // patch_size) ** 2
        self._patch_size = patch_size
        self._projs = nn.Linear(
            nc * patch_size * patch_size,
            embed_dim,
            bias=False,
        )
        self._transformers = nn.ModuleList(
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
        self._mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_features),
        )
        self._dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._cls_token = nn.Parameter(torch.randn(embed_dim))
        self._pos_embed = nn.Parameter(torch.randn(np + 1, embed_dim))

    def forward(self, img: Tensor, u: Tensor) -> Tensor:
        """Override"""
        x, n_patches, _ = make_patches(img, self._patch_size)  # (b, np, N)
        # Layers only act on last dim(s) =)
        x = self._projs(x)  # (b, np, ed)
        ct = self._cls_token.repeat(x.shape[0], 1, 1)  # (b, 1, ed)
        z = torch.concat([ct, x], dim=1)  # (b, np + 1, ed)
        z = z + self._pos_embed  # (b, np + 1, ed)
        z = self._dropout(z)
        z = z.transpose(0, 1)  # (np + 1, b, ed)
        for transformer in self._transformers:
            z = transformer(z, u.repeat(n_patches + 1, 1, 1))
        z = z[0]  # (b, ed)
        y = self._mlp_head(z)
        return y


class TabTransformer(nn.Module):
    """
    TabTransformer from

        Huang, Xin, et al. "Tabtransformer: Tabular data modeling using
        contextual embeddings." arXiv preprint arXiv:2012.06678 (2020).

    In a nutshell, this module processes tabular data as follows:
    - categorical features go through a stack of transformer encoders,
    - numerical features are concatenated with the result of the transformer
    stack,

    and the whole thing is passed through a final MLP head.
    """

    _col_embed: nn.ModuleDict
    _mlp: nn.Module
    _transformers: nn.Module
    _norm: nn.Module

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
        """
        Args:
            n_num_features (int):
            n_classes (Dict[str, int]): This is a dict that maps a categorical
                feature name (e.g. `Chip_Type`) to the number of possible
                classes (in this case 3). See also `acc23.constants.CLASSES`.
            embed_dim (int, optional):
            n_transformers (int, optional):
            n_heads (int, optional):
            dropout (float, optional):
            activation (str, optional):
            mlp_dim (int, optional):
        """
        super().__init__(**kwargs)
        self._col_embed = nn.ModuleDict(
            {
                k: nn.Linear(n, embed_dim, bias=False)
                for k, n in n_classes.items()
            }
        )
        self._transformers = nn.Sequential(
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
        self._norm = nn.LayerNorm(n_num_features)
        self._mlp = nn.Sequential(
            nn.Linear(len(n_classes) * embed_dim + n_num_features, mlp_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, out_dim),
            get_activation(activation),
        )

    def forward(self, x_cat: Dict[str, Tensor], x_num: Tensor) -> Tensor:
        """
        Args:
            x_cat (Dict[str, Tensor]): Categorical features as a dict of binary
                tensors
            x_num (Tensor): Numerical features
        """
        embs: List[Tensor] = []
        for k, v in x_cat.items():
            w = self._col_embed[k](v)
            w = w.unsqueeze(0)  # for the cat later
            embs.append(w)
        u = torch.cat(embs, dim=0)
        u = self._transformers(u)
        u = u.permute(1, 0, 2)
        u = u.reshape(u.shape[0], -1)
        v = torch.cat([u, self._norm(x_num)], dim=-1)
        w = self._mlp(v)
        return w


class VisionTransformer(nn.Module):
    """
    Home-made implementation of the vision transformer of

        Dosovitskiy, Alexey, et al. "An image is worth 16x16 words:
        Transformers for image recognition at scale." arXiv preprint
        arXiv:2010.11929 (2020).

    See also:
        https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html
    """

    _patch_size: int
    _proj: nn.Module
    _transformers: nn.Module
    _mlp_head: nn.Module
    _dropout: nn.Module

    _cls_token: nn.Parameter
    _pos_embed: nn.Parameter

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
            dropout (float):
            activation (str):
            mlp_dim (int):
        """
        super().__init__()
        c, s, _ = input_shape
        np = (s // patch_size) ** 2
        self._patch_size = patch_size
        self._proj = nn.Linear(
            c * patch_size * patch_size,
            embed_dim,
            bias=False,
        )
        self._transformers = nn.Sequential(
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
        self._mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
            get_activation(activation),
        )
        self._dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._cls_token = nn.Parameter(torch.randn(embed_dim))
        self._pos_embed = nn.Parameter(torch.randn(np + 1, embed_dim))

    def forward(self, img: Tensor) -> Tensor:
        """Override"""
        x, _, _ = make_patches(img, self._patch_size)  # (b, np, c * ps * ps)
        # Layers only act on last dim(s) =)
        x = self._proj(x)  # (b, np, ed)
        ct = self._cls_token.repeat(x.shape[0], 1, 1)  # (b, 1, ed)
        z = torch.concat([ct, x], dim=1)  # (b, np + 1, ed)
        z = z + self._pos_embed  # (b, np + 1, ed)
        z = self._dropout(z)
        z = z.transpose(0, 1)  # (np + 1, b, ed)
        z = self._transformers(z)
        z = z[0]  # (b, ed)
        y = self._mlp_head(z)
        return y

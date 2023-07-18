"""ACC23 model prototype *Primus*"""

from typing import Any, Dict, Literal, Tuple, Union

import torch
from torch import Tensor, nn
from torchvision.transforms import Resize
from transformers import ViTConfig, ViTModel

from acc23.constants import (
    CLASSES,
    FEATURES,
    IMAGE_SIZE,
    N_CHANNELS,
    N_FEATURES,
    N_TRUE_TARGETS,
)

from .base_mlc import BaseMultilabelClassifier, to_hierarchical_logits
from .layers import MLP, concat_tensor_dict


class TabularPreprocessor(nn.Module):
    """
    Separates numerical features fron binary features, and recombines the
    latter

    TODO: Same as in Orchid
    """

    def forward(
        self,
        x: Dict[str, Tensor],
        device: Union[torch.device, str] = "cpu",
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Returns:
            The categorical feature tensor and the numerical feature tensor
        """
        cat_columns = [k + "_" + c for k, v in CLASSES.items() for c in v]
        x_cat: Dict[str, Tensor] = {}
        for k, v in CLASSES.items():
            x_cat[k] = concat_tensor_dict({c: x[k + "_" + c] for c in v})
            x_cat[k] = x_cat[k].float().to(device)  # type: ignore
        x_num = concat_tensor_dict(
            {k: v for k, v in x.items() if k not in cat_columns}
        ).float()
        x_num = x_num.to(device)  # type: ignore
        return x_cat, x_num


class CategoricalEmbedding(nn.Module):
    """Embeds a dict of one-hot tensors"""

    emb: nn.ModuleDict

    def __init__(self, n_classes: Dict[str, int], embed_dim: int) -> None:
        """
        Args:
            n_classes (Dict[str, int]): A dict that maps the categorical
                feature names to the number of classes in that feature
            embed_dim (int)
        """
        super().__init__()
        self.emb = nn.ModuleDict(
            {
                a: nn.Linear(b, embed_dim, bias=False)
                for a, b in n_classes.items()
            }
        )

    def forward(
        self,
        x: Dict[str, Tensor],
    ) -> Tensor:
        """Override"""
        a = {k: self.emb[k](v) for k, v in x.items()}
        return concat_tensor_dict(a)


class Primus(BaseMultilabelClassifier):
    """
    Like Orchid but the `acc23.models.transformers.TabTransformer` is replaced
    by a simpler module: the categorical features are embedded, merged with the
    continuous features, and passed through a simple MLP. As usual, the image
    is passed through a vision transformer. The result of both branches are
    concatenated and fed through a MLP head.
    """

    _mlp_head: nn.Module
    _tab_pre: nn.Module
    _cat_emb: CategoricalEmbedding
    _tab_mlp: nn.Module
    _vit_proj: nn.Module
    _vit_resize: nn.Module
    _vit: nn.Module

    def __init__(
        self,
        n_features: int = N_FEATURES,
        image_shape: Tuple[int, int, int] = (
            N_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        out_dim: int = N_TRUE_TARGETS,
        cat_embed_dim: int = 16,
        embed_dim: int = 128,
        dropout: float = 0.25,
        activation: str = "gelu",
        mlp_dim: int = 2048,
        vit: Literal["pretrained", "frozen"] = "pretrained",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            n_features (int, optional): Number of numerical tabular features
            image_shape (Tuple[int, int, int], optional):
            out_dim (int, optional):
            cat_embed_dim (int, optional): Embedding dimension for categorical
                features
            dropout (float, optional):
            activation (str, optional):
            mlp_dim (int, optional):
            vit (Literal["new", "pretrained", "frozen"], optional): How to
                create the vision transformer:
                - `pretrained`: a pretrained `ViTModel` is used (specifically
                `google/vit-base-patch16-224`)
                - `frozen`: same, but the ViT is frozen

        See also:
            `acc23.models.base_mlc.BaseMultilabelClassifier.__init__`
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        n_classes = {a: len(b) for a, b in CLASSES.items()}
        n_cont_features = n_features - sum(n_classes.values())
        self._tab_pre = TabularPreprocessor()
        self._cat_emb = CategoricalEmbedding(n_classes, cat_embed_dim)
        self._tab_mlp = MLP(
            len(n_classes) * cat_embed_dim + n_cont_features,
            [embed_dim],
            dropout=dropout,
        )
        self._vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self._vit.requires_grad_(vit != "frozen")
        if self._vit.pooler is not None:
            self._vit.pooler.requires_grad_(False)
        if not isinstance(self._vit, ViTModel):
            raise RuntimeError(
                "Pretrained ViT is not a transformers.ViTModel object"
            )
        if not isinstance(self._vit.config, ViTConfig):
            raise RuntimeError(
                "Pretrained ViT confit is not a transformers.ViTConfig "
                "object"
            )
        self._vit_resize = Resize(self._vit.config.image_size, antialias=True)
        self._vit_proj = nn.Linear(
            self._vit.config.hidden_size, embed_dim, bias=False
        )
        self._mlp_head = MLP(
            in_dim=2 * embed_dim,
            hidden_dims=[mlp_dim, out_dim],
            dropout=dropout,
            activation=activation,
            is_head=True,
        )
        nc, s, _ = image_shape
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
        x_cat, x_num = self._tab_pre(x, self.device)
        img = img.to(self.device)  # type: ignore
        x_cat = self._cat_emb(x_cat)
        u = torch.concatenate([x_cat, x_num], dim=-1)
        u = self._tab_mlp(u)
        v = self._vit_resize(img)
        v = self._vit(v).last_hidden_state[:, 0]
        v = self._vit_proj(v)
        uv = torch.concatenate([u, v], dim=-1)
        w = self._mlp_head(uv)
        # w = to_hierarchical_logits(w, mode="max")
        return w

"""
ACC23 main multi-classification model: prototype "Primus". Like Orchid but the
`TabTransformer` branch is simplified: the categorical features are embedded,
merged with the continuous features, and pass through a simple mlp.
"""
__docformat__ = "google"

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

    # pylint: disable=missing-function-docstring
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

    # pylint: disable=missing-function-docstring
    def forward(
        self,
        x: Dict[str, Tensor],
    ) -> Tensor:
        a = {k: self.emb[k](v) for k, v in x.items()}
        return concat_tensor_dict(a)


class Primus(BaseMultilabelClassifier):
    """See module documentation"""

    mlp_head: nn.Module
    tab_pre: nn.Module  # Tabular feature preprocessing
    cat_emb: CategoricalEmbedding  # Embedding for each categorical feature
    tab_mlp: nn.Module  # Tabular MLP
    vit_proj: nn.Module
    vit_resize: nn.Module
    vit: nn.Module  # Vision transformer

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (
            N_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        out_dim: int = N_TRUE_TARGETS,
        cat_embed_dim: int = 16,  # Embedding dimension for categorical features
        embed_dim: int = 128,
        dropout: float = 0.5,
        activation: str = "gelu",
        mlp_dim: int = 2048,
        # pooling: bool = False,
        # freeze_vit: bool = False,
        vit: Literal["pretrained", "frozen"] = "pretrained",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        n_classes = {a: len(b) for a, b in CLASSES.items()}
        n_cont_features = N_FEATURES - sum(n_classes.values())
        self.tab_pre = TabularPreprocessor()
        self.cat_emb = CategoricalEmbedding(n_classes, cat_embed_dim)
        self.tab_mlp = MLP(
            len(n_classes) * cat_embed_dim + n_cont_features,
            [embed_dim],
            dropout=dropout,
        )
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.vit.requires_grad_(vit != "frozen")
        if self.vit.pooler is not None:
            self.vit.pooler.requires_grad_(False)
        if not isinstance(self.vit, ViTModel):
            raise RuntimeError(
                "Pretrained ViT is not a transformers.ViTModel object"
            )
        if not isinstance(self.vit.config, ViTConfig):
            raise RuntimeError(
                "Pretrained ViT confit is not a transformers.ViTConfig "
                "object"
            )
        self.vit_resize = Resize(self.vit.config.image_size, antialias=True)
        self.vit_proj = nn.Linear(
            self.vit.config.hidden_size, embed_dim, bias=False
        )
        self.mlp_head = MLP(
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
        x_cat, x_num = self.tab_pre(x, self.device)
        img = img.to(self.device)  # type: ignore
        x_cat = self.cat_emb(x_cat)
        u = torch.concatenate([x_cat, x_num], dim=-1)
        u = self.tab_mlp(u)
        v = self.vit_resize(img)
        v = self.vit(v).last_hidden_state[:, 0]
        v = self.vit_proj(v)
        uv = torch.concatenate([u, v], dim=-1)
        w = self.mlp_head(uv)
        # w = to_hierarchical_logits(w, mode="max")
        return w

"""
ACC23 main multi-classification model: prototype "Orchid". Like Ampere but the
convolutional branch is a vision transformer and the tabular branch is a
`TabTransformer`
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
from .transformers import TabTransformer


class TabularPreprocessor(nn.Module):
    """
    Separates numerical features fron binary features, and recombines the
    latter
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


class Orchid(BaseMultilabelClassifier):
    """See module documentation"""

    mlp_head: nn.Module
    tat_pre: nn.Module  # Tabular feature preprocessing
    tat: nn.Module  # Tabular transformer
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
        embed_dim: int = 512,
        patch_size: int = 8,
        n_transformers: int = 16,
        n_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        mlp_dim: int = 2048,
        # pooling: bool = False,
        # freeze_vit: bool = False,
        vit: Literal["new", "pretrained", "frozen"] = "pretrained",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        nc, s, _ = image_shape
        n_classes = {k: len(v) for k, v in CLASSES.items()}
        self.tat_pre = TabularPreprocessor()
        self.tat = TabTransformer(
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
        if vit == "new":
            self.vit = ViTModel(
                config=ViTConfig(
                    hidden_size=embed_dim,
                    num_hidden_layers=n_transformers,
                    num_attention_heads=n_heads,
                    intermediate_size=mlp_dim,
                    hidden_act=activation,
                    hidden_dropout_prob=dropout,
                    attention_probs_dropout_prob=dropout,
                    image_size=s,
                    num_channels=nc,
                    patch_size=patch_size,
                ),
                add_pooling_layer=False,
            )
            self.vit_resize = nn.Identity()
            self.vit_proj = nn.Identity()
        else:
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
            self.vit_resize = Resize(
                self.vit.config.image_size, antialias=True
            )
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
        x_cat, x_num = self.tat_pre(x, self.device)
        img = img.to(self.device)  # type: ignore
        a = self.tat(x_cat, x_num)
        b = self.vit_resize(img)
        b = self.vit(b).last_hidden_state[:, 0]
        b = self.vit_proj(b)
        ab = torch.concatenate([a, b], dim=-1)
        c = self.mlp_head(ab)
        c = to_hierarchical_logits(c, mode="max")
        return c

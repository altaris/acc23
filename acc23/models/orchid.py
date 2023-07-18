"""ACC23 model prototype *Orchid*"""

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
    """Separates numerical features from the binary features"""

    def forward(
        self,
        x: Dict[str, Tensor],
        device: Union[torch.device, str] = "cpu",
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Returns:
            The categorical feature tensor dict and the numerical feature
            tensor
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
    """
    Like Ampere but the convolutional branch is a vision transformer and the
    tabular branch is a `acc23.models.transformers.TabTransformer`.
    """

    _mlp_head: nn.Module
    _tat_pre: nn.Module
    _tat: nn.Module
    _vit_proj: nn.Module
    _vit_resize: nn.Module
    _vit: nn.Module
    _pool: nn.Module

    def __init__(
        self,
        n_features: int = N_FEATURES,
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
        dropout: float = 0.25,
        activation: str = "gelu",
        mlp_dim: int = 2048,
        vit: Literal["new", "pretrained", "frozen"] = "pretrained",
        pooling: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            n_features (int, optional): Number of numerical tabular features
            image_shape (Tuple[int, int, int], optional):
            out_dim (int, optional):
            embed_dim (int, optional):
            patch_size (int, optional):
            n_transformers (int, optional): Ignored if `vit` is not `new`
            n_heads (int, optional): Ignored if `vit` is not `new`
            dropout (float, optional):
            activation (str, optional):
            mlp_dim (int, optional):
            vit (Literal["new", "pretrained", "frozen"], optional): How to
                create the vision transformer:
                - `new`: a new vision transformer is created (see [Hugging Face
                Transformer's
                `ViTModel`s](https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/vit#transformers.ViTModel))
                - `pretrained`: a pretrained `ViTModel` is used (specifically
                `google/vit-base-patch16-224`)
                - `frozen`: same, but the ViT is frozen

        See also:
            `acc23.models.base_mlc.BaseMultilabelClassifier.__init__`
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        nc, s, _ = image_shape
        n_classes = {k: len(v) for k, v in CLASSES.items()}
        self._tat_pre = TabularPreprocessor()
        self._tat = TabTransformer(
            n_num_features=n_features - sum(n_classes.values()),
            n_classes=n_classes,
            out_dim=embed_dim,
            embed_dim=embed_dim,
            n_transformers=n_transformers,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            mlp_dim=mlp_dim,
        )
        self._pool = nn.MaxPool2d(5, 1, 2) if pooling else nn.Identity()
        if vit == "new":
            self._vit = ViTModel(
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
            self._vit_resize = nn.Identity()
            self._vit_proj = nn.Identity()
        else:
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
            self._vit_resize = Resize(
                self._vit.config.image_size, antialias=True
            )
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
        """Override"""
        x_cat, x_num = self._tat_pre(x, self.device)
        img = img.to(self.device)  # type: ignore
        a = self._tat(x_cat, x_num)
        b = self._pool(img)
        b = self._vit_resize(b)
        b = self._vit(b).last_hidden_state[:, 0]
        b = self._vit_proj(b)
        ab = torch.concatenate([a, b], dim=-1)
        c = self._mlp_head(ab)
        # c = to_hierarchical_logits(c, mode="max")
        return c

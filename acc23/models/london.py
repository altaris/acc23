"""
ACC23 main multi-classification model: prototype "London". Like Ampere but the
convolutional branch is a vision transformer
"""
__docformat__ = "google"

from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor, nn
from torchvision.transforms import Resize
from transformers import ViTConfig, ViTModel

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .layers import MLP, concat_tensor_dict


class London(BaseMultilabelClassifier):
    """See module documentation"""

    mlp_head: nn.Module  # Fusion branch
    tae: nn.Module  # Tabular encoder
    vit_proj: nn.Module
    vit_resize: nn.Module
    vit: nn.Module  # Vision transformer

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
        # patch_size: int = 8,
        # n_transformers: int = 16,
        # n_heads: int = 8,
        dropout: float = 0.5,
        activation: str = "gelu",
        mlp_dim: int = 4096,
        pooling: bool = False,
        fine_tune_vit: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        nc, s, _ = image_shape
        self.tae = MLP(
            in_dim=n_features,
            hidden_dims=[mlp_dim, embed_dim],
            dropout=dropout,
            activation=activation,
            is_head=False,
        )
        # self.vit = ViTModel(
        #     config=ViTConfig(
        #         hidden_size=embed_dim,
        #         num_hidden_layers=n_transformers,
        #         num_attention_heads=n_heads,
        #         intermediate_size=mlp_dim,
        #         hidden_act=activation,
        #         hidden_dropout_prob=dropout,
        #         attention_probs_dropout_prob=dropout,
        #         image_size=s,
        #         num_channels=nc,
        #         patch_size=patch_size,
        #     ),
        #     add_pooling_layer=pooling,
        # )
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        if fine_tune_vit:
            for p in self.vit.parameters():
                p.requires_grad = False
        elif not pooling and self.vit.pooler is not None:
            self.vit.pooler.requires_grad_(False)
        if not isinstance(self.vit, ViTModel):
            raise RuntimeError(
                "Pretrained ViT is not a transformers.ViTModel object"
            )
        if not isinstance(self.vit.config, ViTConfig):
            raise RuntimeError(
                "Pretrained ViT confit is not a transformers.ViTConfig object"
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
        a = self.tae(x)
        b = self.vit_resize(img)
        b = self.vit(b).last_hidden_state[:, 0]
        b = self.vit_proj(b)
        ab = torch.concatenate([a, b], dim=-1)
        c = self.mlp_head(ab)
        return c

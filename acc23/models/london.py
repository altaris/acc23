"""ACC23 model prototype *London*"""

from typing import Any, Dict, Literal, Tuple, Union

import torch
from torch import Tensor, nn
from torchvision.transforms import Resize
from transformers import ViTConfig, ViTModel

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .layers import MLP, concat_tensor_dict


class London(BaseMultilabelClassifier):
    """
    Like `acc23.models.ampere.Ampere` but the convolutional branch is a vision
    transformer.
    """

    _mlp_head: nn.Module
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
        embed_dim: int = 128,
        patch_size: int = 8,
        n_transformers: int = 16,
        n_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        mlp_dim: int = 2048,
        vit: Literal["new", "pretrained", "frozen"] = "pretrained",
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
        self._tab_mlp = MLP(
            in_dim=n_features,
            hidden_dims=[mlp_dim, embed_dim],
            dropout=dropout,
            activation=activation,
            is_head=False,
        )
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
            self._vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
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
        a = self._tab_mlp(x)
        b = self._vit_resize(img)
        b = self._vit(b).last_hidden_state[:, 0]
        b = self._vit_proj(b)
        ab = torch.concatenate([a, b], dim=-1)
        c = self._mlp_head(ab)
        return c

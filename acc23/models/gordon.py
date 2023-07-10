"""ACC23 model prototype *Gordon*"""

from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor, nn
from transformers.activations import get_activation

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .imagetabnet import VisionEncoder
from .layers import MLP, concat_tensor_dict


class Gordon(BaseMultilabelClassifier):
    """
    Fusion model inspired by

        Y. Liu, H. -P. Lu and C. -H. Lai, "A Novel Attention-Based Multi-Modal
        Modeling Technique on Mixed Type Data for Improving TFT-LCD Repair
        Process," in IEEE Access, vol. 10, pp. 33026-33036, 2022, doi:
        10.1109/ACCESS.2022.3158952.

    In a nutshell, the tabular data is encoded by a MLP as usual. The image, on
    the other hand, goes through several
    `acc23.models.imagetabnet.VisionEncoder`, which also involves the encoded
    tabular data. The result of both branches are concatenated and fed through
    a MLP head.
    """

    _tab_mlp: nn.Module
    _vis_pool: nn.Module
    _vis_enc: nn.Module
    _vis_proj: nn.Module
    _mlp_head: nn.Module

    def __init__(
        self,
        n_features: int = N_FEATURES,
        image_shape: Tuple[int, int, int] = (
            N_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        out_dim: int = N_TRUE_TARGETS,
        embed_dim: int = 16,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            n_features (int, optional): Number of numerical tabular features
            image_shape (Tuple[int, int, int], optional):
            out_dim (int, optional):
            embed_dim (int, optional):
            dropout (float, optional):
            activation (str, optional):

        See also:
            `acc23.models.base_mlc.BaseMultilabelClassifier.__init__`
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        nc, s, _ = image_shape
        self._tab_mlp = MLP(
            n_features,
            [embed_dim],
            activation=activation,
        )
        self._vis_pool = nn.MaxPool2d(5, 2, 2)
        # self.vision_branch_a = nn.Identity()
        self._vis_enc = VisionEncoder(
            # image_shape=image_shape,
            image_shape=(nc, s // 2, s // 2),
            out_channels=[
                # 4,  # 512 -> 256
                4,  # 256 -> 128
                4,  # -> 64
                4,  # -> 32
                4,  # -> 16
                4,  # -> 8
                4,  # -> 4
            ],
            in_features=embed_dim,
            activation=activation,
        )
        self._vis_proj = nn.Linear(4 * 4 * 4, embed_dim, bias=False)
        # self.main_branch = nn.Sequential(
        #     nn.Linear(2 * embed_dim, out_dim),
        # )
        self._mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            get_activation(activation),
            nn.Dropout1d(dropout),
            nn.Linear(embed_dim, out_dim),
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
        if isinstance(x, dict):
            x = concat_tensor_dict(x)
        x = x.float().to(self.device)  # type: ignore
        img = img.to(self.device)  # type: ignore
        u = self._tab_mlp(x)
        v = self._vis_pool(img)
        v = self._vis_enc(v, u)
        v = self._vis_proj(v)
        # uv = torch.concatenate([u, v], dim=-1)
        uv = u * v
        w = self._mlp_head(uv)
        return w

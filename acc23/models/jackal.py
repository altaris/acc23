"""
ACC23 main multi-classification model: prototype "Jackal". Essentially like
Gordon but all channels are processed separately with their own vision encoder.
"""
__docformat__ = "google"

from typing import Any, Dict, Union

import torch
from torch import Tensor, nn

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .layers import concat_tensor_dict
from .imagetabnet import VisionEncoder


class Jackal(BaseMultilabelClassifier):
    """See module documentation"""

    tabular_branch: nn.Module  # Dense input branch
    pooling: nn.Module  # Image pooling / convolution before vision encoding
    vision_encoders: nn.ModuleList  # Vision encoding
    main_branch: nn.Module  # Fusion branch

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        n_features = 256
        self.tabular_branch = nn.Sequential(
            nn.Linear(N_FEATURES, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, n_features),
            nn.SiLU(),
        )
        self.pooling = nn.Sequential(
            nn.MaxPool2d(5, 2, 2),  # IMAGE_RESIZE_TO = 512 -> 256
            nn.Conv2d(N_CHANNELS, N_CHANNELS, 3, 2, 1, bias=False),  # -> 256
            nn.BatchNorm2d(N_CHANNELS),
            nn.SiLU(),
        )
        # self.pooling = nn.MaxPool2d(5, 2, 2)  # IMAGE_RESIZE_TO = 512 -> 256
        self.vision_encoders = nn.ModuleList(
            [
                VisionEncoder(
                    in_channels=1,
                    out_channels=[
                        2,  # 256 -> 128
                        4,  # -> 64
                        8,  # -> 32
                        16,  # -> 16
                        32,  # -> 8
                        64,  # -> 4
                        128,  # -> 2
                        n_features,  # -> 1
                    ],
                    in_features=n_features,
                )
                for _ in range(N_CHANNELS)
            ]
        )
        self.main_branch = nn.Sequential(
            nn.Linear(4 * n_features, n_features),
            nn.SiLU(),
            nn.Linear(n_features, N_TARGETS),
        )
        self.example_input_array = (
            torch.zeros((32, N_FEATURES)),
            torch.zeros((32, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)),
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
        a = self.tabular_branch(x)
        b = self.pooling(img)
        cs = [ve(s, a) for ve, s in zip(self.vision_encoders, b.split(1, 1))]
        d = torch.concatenate([a] + cs, dim=-1)
        e = self.main_branch(d)
        return e

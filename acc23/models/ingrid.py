"""
ACC23 main multi-classification model: prototype "Ingrid". Variation of
ImageTabNet vision encoder, where that vision attention module is replaced by
actual multihead attention blocks.
"""
__docformat__ = "google"

from itertools import zip_longest
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor, nn
from transformers.activations import get_activation
from transformers.models.resnet.modeling_resnet import ResNetConvLayer

from acc23.constants import IMAGE_SIZE, N_CHANNELS, N_FEATURES, N_TRUE_TARGETS

from .base_mlc import BaseMultilabelClassifier
from .layers import ResNetLinearLayer, concat_tensor_dict


class AttentionModule(nn.Module):
    """See Figure 2 of the paper"""

    block_1: nn.Module
    block_2: nn.Module

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        in_features: int,
        activation: str = "silu",
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            in_channels (int):
            in_features (int): Dimension of the output of tabnet
            activation (str): Defaults to silu
        """
        super().__init__(**kwargs)
        c, s, _ = image_shape  # TODO: assert square image?
        self.block_1 = nn.Sequential(
            nn.Linear(in_features, c),
            get_activation(activation),
        )
        self.block_2 = nn.MultiheadAttention(
            embed_dim=c * s * s,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )

    def forward(self, img: Tensor, h: Tensor, *_, **__) -> Tensor:
        """
        Reshapes and inflates the input vectors so that they can be passed to
        the multihead attension module as embedding vectors. It's a bit
        convoluted (pun intended) so heres how it works:

            1. Create dummy batch of 3 channels `2 x 2` images:

            >>> img = Tensor(
            >>>     [
            >>>         [
            >>>             [[1, 2], [3, 4]],
            >>>             [[10, 20], [30, 40]],
            >>>             [[100, 200], [300, 400]],
            >>>         ],
            >>>         [
            >>>             [[5, 6], [7, 8]],
            >>>             [[50, 60], [70, 80]],
            >>>             [[500, 600], [700, 800]],
            >>>         ]
            >>>     ]
            >>> )
            >>> b, c, s, _ = img.shape
            >>> b, c, s, s
            (2, 3, 2, 2)

            2. Reshape it into a 2-dimensional tensor, so that every row is
               every the concatenation of every pixel. Here, a pixel is a
               triple of channel values.

            >>> img = img.reshape((b, c * s * s))
            >>> img.shape, img
            (
                torch.Size([2, 12]),
                tensor([
                    [  1.,   2.,   3.,   4.,  10.,  20.,  30.,  40., 100., 200., 300., 400.],
                    [  5.,   6.,   7.,   8.,  50.,  60.,  70.,  80., 500., 600., 700., 800.]
                ])
            )

            3. Create a dummy batch of length 3 (the number of channels)
               feature vectors (`u = self.block_1(h)`):

            >>> v = Tensor([[.1, .2, .3], [.5, .6, .7]])
            >>> v.shape
            torch.Size([2, 3])

            4. "Inflate" it so that it shapes matches that of `img`

            >>> v = v.unsqueeze(-1)
            >>> v = torch.concat([v] * s * s, dim=-1)
            >>> v = v.reshape((b, c * s * s))
            >>> v.shape, v
            (
                torch.Size([2, 3, 1, 1]),
                tensor([
                    [0.1000, 0.1000, 0.1000, 0.1000, 0.2000, 0.2000, 0.2000, 0.2000, 0.3000, 0.3000, 0.3000, 0.3000],
                    [0.5000, 0.5000, 0.5000, 0.5000, 0.6000, 0.6000, 0.6000, 0.6000, 0.7000, 0.7000, 0.7000, 0.7000]
                ])
            )

            5. At this stage, `img` and `v` have the same shape and can be
               passed to the multihead attention module. The output vector can
               then simply be reshaped to `(b, c, s, s)`.

        """
        b, c, s = img.shape[0], img.shape[1], img.shape[2]
        u = img.reshape((b, c * s * s))
        v = self.block_1(h)
        v = v.unsqueeze(-1)
        v = torch.concat([v] * s * s, dim=-1)
        v = v.reshape((b, c * s * s))
        w, _ = self.block_2(
            query=u,
            key=v,
            value=v,
            need_weights=False,
        )
        return (u + w).reshape((b, c, s, s))


class VisionEncoder(nn.Module):
    """
    It is a succession of blocks that look like
    1. `ResNetConvLayer`: a residual convolutional block that cuts the image
        size (height and width) by half;
    2. `AttentionModule` that incorporates the feature vector into the channels
       of the image using multihead attention.
    """

    encoder_layers: nn.ModuleList
    fusion_layers: nn.ModuleList

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        out_channels: List[int],
        in_features: int,
        activation: str = "silu",
        attention_after_last: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int):
            out_channels (List[int]):
            in_features (int): Dimension of the feature vector to inject after
                each convolution stage
            n_decision_steps (int): Number of decision steps
            activation (str): Defaults to silu
            attention_after_last (bool): Whether to add an attention module
                after the last residual block, defaults to `False`.
        """
        super().__init__()
        c, s, _ = image_shape
        all_c = [c] + out_channels
        self.encoder_layers = nn.ModuleList(
            [
                # nn.Sequential(
                #     nn.Conv2d(c[i - 1], c[i], 4, 2, 1, bias=False),
                #     get_activation(activation),
                # )
                ResNetConvLayer(all_c[i - 1], all_c[i], 3, 2, activation)
                # ResNetEncoderLayer(
                #     c[i - 1],
                #     c[i],
                #     n_blocks=3,
                #     activation=activation,
                # )
                for i in range(1, len(all_c))
            ]
        )
        k = len(out_channels) if attention_after_last else -1
        self.fusion_layers = nn.ModuleList(
            [
                AttentionModule(
                    (out_c, int(s / (2 ** (i + 1))), int(s / (2 ** (i + 1)))),
                    in_features,
                    activation,
                )
                for i, out_c in enumerate(out_channels[:k])
            ]
        )

    def forward(self, img: Tensor, h: Tensor, *_, **__) -> Tensor:
        """
        Args:
            img (Tensor):
            h (Tensor):
        """
        # itertool.zip_longest pads the shorter sequence with None's
        for c, f in zip_longest(self.encoder_layers, self.fusion_layers):
            img = c(img)
            if f is not None:
                img = f(img, h)
        return img.flatten(1)


class Ingrid(BaseMultilabelClassifier):
    """See module documentation"""

    tabular_branch: nn.Module  # Dense input branch
    vision_branch_a: nn.Module  # Conv. pool input branch
    vision_branch_b: nn.Module  # Conv fusion encoder
    main_branch: nn.Module  # Fusion branch

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        n_features = 256
        self.tabular_branch = nn.Sequential(
            ResNetLinearLayer(N_FEATURES, 256),
            ResNetLinearLayer(256, n_features),
        )
        # self.vision_branch_a = nn.Sequential(
        #     nn.MaxPool2d(5, 1, 2),  # IMAGE_RESIZE_TO = 512 -> 512
        #     nn.Conv2d(N_CHANNELS, 8, 4, 2, 1, bias=False),  # -> 256
        #     nn.BatchNorm2d(8),
        #     nn.SiLU(),
        #     # nn.MaxPool2d(3, 1, 1),  # 256 -> 256
        # )
        self.vision_branch_a = nn.Sequential(
            nn.MaxPool2d(5, 2, 2),  # IMAGE_RESIZE_TO = 512 -> 256
            ResNetConvLayer(N_CHANNELS, 8, 3, 2, "silu"),  # -> 128
            ResNetConvLayer(8, 8, 3, 2, "silu"),  # -> 64
            ResNetConvLayer(8, 16, 3, 2, "silu"),  # -> 32
            # ResNetConvLayer(16, 16, 3, 2, "silu"),  # -> 16
            # ResNetConvLayer(16, 32, 3, 2, "silu"),  # -> 8
        )
        self.vision_branch_b = VisionEncoder(
            image_shape=(16, 32, 32),
            out_channels=[
                32,  # -> 16
                32,  # -> 8
                64,  # -> 4
                128,  # -> 2
                n_features,  # -> 1
            ],
            in_features=n_features,
        )
        self.main_branch = nn.Sequential(
            nn.Linear(2 * n_features, N_TRUE_TARGETS),
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
        u = self.tabular_branch(x)
        v = self.vision_branch_a(img)
        v = self.vision_branch_b(v, u)
        uv = torch.concatenate([u, v], dim=-1)
        w = self.main_branch(uv)
        return w

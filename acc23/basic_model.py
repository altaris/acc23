"""First attempt at a model"""
__docformat__ = "google"

from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import nn


def basic_encoder(
    blocks_channels: List[int],
    input_size: int = 512,
    input_channels: int = 3,
) -> Tuple[nn.Module, int]:
    """
    Basic image encoder that is just a succession of (non skipped) downsampling
    blocks, followed by a final flatten layer.

    Args:
        blocks_channels (List[int]): Number of output channels of each
            block. Note that each blocks cuts the width and height of the input
            by half.
        input_size (int): The width (or equivalently the height) of an imput
            image.
        input_channels (int): Number of channels of the input image

    Return:
        The encoder and the (flat) dimension of the output vector.
    """
    module = nn.Sequential(
        nn.Conv2d(input_channels, blocks_channels[0], 4, 2, 1),
        nn.LeakyReLU(0.2),
    )
    for i in range(1, len(blocks_channels)):
        a, b = blocks_channels[i - 1], blocks_channels[i]
        module.append(nn.Conv2d(a, b, 4, 2, 1))
        module.append(nn.BatchNorm2d(b))
        module.append(nn.LeakyReLU(0.2))
    module.append(nn.Flatten())
    # Height (eq. width) of the encoded image before flatten
    es = int(input_size / (2 ** len(blocks_channels)))
    return (module, blocks_channels[-1] * (es**2))


def concat_tensor_dict(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Converts a dict of tensors to a tensor. The tensors (of the dict) are
    concatenated along the last axis. If they are 1-dimensional, they are
    unsqueezed along a new axis at the end. I know it's not super clear, so
    here is an example

        # The foo tensor will be unsqueezed
        d = {"foo": torch.ones(3), "bar": torch.zeros((3, 1))}
        concat_tensor_dict(d)
        >>> tensor([[1., 0.],
        >>>         [1., 0.],
        >>>         [1., 0.]])

    """

    def _maybe_unsqueeze(t: torch.Tensor) -> torch.Tensor:
        return t if t.ndim > 1 else t.unsqueeze(-1)

    return torch.concatenate(list(map(_maybe_unsqueeze, d.values())), dim=-1)


def linear_chain(n_inputs: int, n_neurons: List[int]) -> nn.Module:
    """
    A sequence of linear layers with sigmoid activation (including at the end).
    """
    module = nn.Sequential(
        nn.Linear(n_inputs, n_neurons[0]),
        nn.Sigmoid(),
    )
    for i in range(1, len(n_neurons)):
        a, b = n_neurons[i - 1], n_neurons[i]
        module.append(nn.Linear(a, b))
        module.append(nn.Sigmoid())
    return module


class ACCModel(pl.LightningModule):
    """A basic model :]"""

    _module_a: nn.Module  # Input-dense part
    _module_b: nn.Module  # Input-conv part
    _module_c: nn.Module  # Output part

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        # 474 features, a 512x512 image, 29 targets
        self._module_a = linear_chain(474, [512, 256])
        self._module_b, encoded_dim = basic_encoder(
            [
                8,  # 512 -> 256
                8,  # -> 128
                16,  # -> 64
                16,  # -> 32
                32,  # -> 16
                32,  # -> 8
                64,  # -> 4
            ],
            input_size=512,
            input_channels=3,
        )
        self._module_c = linear_chain(256 + encoded_dim, [256, 64, 29])
        self.example_input_array = {
            "x": {str(i): torch.zeros((1, 1)) for i in range(474)},
            "img": torch.zeros((1, 3, 512, 512)),
        }
        self.forward(**self.example_input_array)

    def _evaluate(
        self,
        x: Dict[str, torch.Tensor],
        y: Dict[str, torch.Tensor],
        img: torch.Tensor,
        stage: Optional[str] = None,
    ) -> torch.Tensor:
        _y = concat_tensor_dict(y).float().to(self.device)  # type: ignore
        criterion = nn.functional.binary_cross_entropy
        bce_loss = criterion(self(x, img), _y)
        if stage is not None:
            self.log(f"{stage}/loss", bce_loss, sync_dist=True)
        return bce_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def forward(self, x: Dict[str, torch.Tensor], img: torch.Tensor, *_, **__):
        x = concat_tensor_dict(x).float().to(self.device)  # type: ignore
        img = img.to(self.device)  # type: ignore
        a = self._module_a(x)
        b = self._module_b(img)
        ab = torch.concatenate([a, b], dim=-1)
        return self._module_c(ab)

    def training_step(self, batch, *_, **__):
        x, y, img = batch
        return self._evaluate(x, y, img, "train")

    def validation_step(self, batch, *_, **__):
        x, y, img = batch
        return self._evaluate(x, y, img, "val")

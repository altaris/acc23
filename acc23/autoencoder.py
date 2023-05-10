"""
Dataset's image autoencoder for latent representation
"""
__docformat__ = "google"

from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from transformers.activations import get_activation

from acc23.constants import IMAGE_RESIZE_TO, N_CHANNELS
from acc23.models.utils import resnet_decoder, resnet_encoder


class Autoencoder(pl.LightningModule):
    """Dataset's image autoencoder for latent representation"""

    _decoder_a: nn.Module
    _decoder_b: nn.Module
    _encoder: nn.Module

    _latent_image_shape: Tuple[int, int, int]

    def __init__(
        self,
        out_channels: List[int],
        latent_space_dim: int = 512,
        input_shape: Tuple[int, int, int] = (
            N_CHANNELS,
            IMAGE_RESIZE_TO,
            IMAGE_RESIZE_TO,
        ),
        n_blocks: int = 1,
        activation: str = "silu",
        last_decoder_activation: str = "sigmoid",
    ) -> None:
        # TODO: assert that the input shape is square
        super().__init__()
        self.save_hyperparameters()

        self._encoder, latent_image_dim = resnet_encoder(
            input_shape[0],
            out_channels,
            input_size=input_shape[1],
            n_blocks=n_blocks,
            activation=activation,
        )
        self._encoder.append(
            nn.Linear(latent_image_dim, latent_space_dim, bias=False)
        )
        a = out_channels[-1]
        b = int(input_shape[1] / (2 ** len(out_channels)))
        self._latent_image_shape = (a, b, b)

        self._decoder_a = nn.Sequential(
            nn.Linear(latent_space_dim, latent_image_dim),
            get_activation(activation),
        )
        c = list(reversed([input_shape[0], *out_channels]))
        self._decoder_b = resnet_decoder(
            c[0],
            c[1:],
            n_blocks=n_blocks,
            activation=activation,
            last_activation=last_decoder_activation,
        )

        self.example_input_array = torch.zeros((32, *input_shape))
        self.forward(self.example_input_array)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }

    def decode(self, z: Tensor) -> Tensor:
        """Decodes a latent vector into an image"""
        z = z.to(self.device)  # type: ignore
        z = self._decoder_a(z)
        z = torch.reshape(z, (-1, *self._latent_image_shape))
        z = self._decoder_b(z)
        return z

    def encode(self, x: Tensor) -> Tensor:
        """Encodes an image into a latent vector"""
        x = x.to(self.device)  # type: ignore
        x = self._encoder(x)
        return x

    def evaluate(self, x: Tensor, stage: Optional[str] = None) -> Tensor:
        """
        Encodes and decodes an input, and evaluates the reconstruction loss
        (bce). If `stage` is given, also logs the loss to tensorboard under
        `<stage>/loss`.
        """
        x_hat = self.forward(x)
        criterion = nn.functional.mse_loss
        loss = criterion(x_hat, x)
        if stage is not None:
            self.log(f"{stage}/loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def training_step(self, img: Tensor, *_, **__) -> Tensor:
        return self.evaluate(img, "train")

    def validation_step(self, img: Tensor, *_, **__) -> Tensor:
        return self.evaluate(img, "val")

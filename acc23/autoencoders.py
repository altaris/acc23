"""Variational autoencoder"""
__docformat__ = "google"

from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch
from diffusers.models.vae import DiagonalGaussianDistribution
from torch import Tensor, nn

from acc23.constants import IMAGE_RESIZE_TO, N_CHANNELS
from acc23.models.utils import ResNetEncoderLayer, resnet_decoder, resnet_encoder


class AE(pl.LightningModule):
    """Dataset's image autoencoder for latent representation"""

    decoder: nn.Module
    encoder: nn.Module

    _latent_image_shape: Tuple[int, int, int]

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (
            N_CHANNELS,
            IMAGE_RESIZE_TO,
            IMAGE_RESIZE_TO,
        ),
        latent_dim: int = 256,
        n_blocks: int = 1,
    ) -> None:
        # TODO: assert that the input shape is square
        super().__init__()
        self.save_hyperparameters()
        self.encoder, _ = resnet_encoder(
            in_channels=input_shape[0],
            out_channels=[
                8,  # IMAGE_RESIZE_TO = 512 -> 256
                8,  # -> 128
                16,  # -> 64
                16,  # -> 32
                32,  # -> 16
                64,  # -> 8
                128,  # -> 4
                latent_dim,  # -> 2
                latent_dim,  # -> 1
            ],
            n_blocks=n_blocks,
            input_size=input_shape[1],
        )
        self.decoder = resnet_decoder(
            in_channels=latent_dim,
            out_channels=[
                latent_dim,  # 1 -> 2
                128,  # -> 4
                64,  # -> 8
                32,  # -> 16
                16,  # -> 32
                16,  # -> 64
                8,  # -> 128
                8,  # -> 256
                input_shape[0],  # -> 512 = IMAGE_RESIZE_TO
            ],
            n_blocks=n_blocks,
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
        if z.ndim == 2:  # Batch of (flat) vectors
            latent_dim = self.hparams["latent_dim"]
            z = z.reshape(-1, latent_dim, 1, 1)
        z = z.to(self.device)  # type: ignore
        return self.decoder(z)

    def encode(self, x: Tensor) -> Tensor:
        """Encodes an image into a (flat) latent vector"""
        x = x.to(self.device)  # type: ignore
        return self.encoder(x)

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
        """Override"""
        return self.evaluate(img, "train")

    def validation_step(self, img: Tensor, *_, **__) -> Tensor:
        """Override"""
        return self.evaluate(img, "val")


class VAE(pl.LightningModule):
    """Variational autoencoder"""

    beta: float
    encoder: nn.Module
    decoder: nn.Module

    # pylint: disable=unused-argument
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (
            N_CHANNELS,
            IMAGE_RESIZE_TO,
            IMAGE_RESIZE_TO,
        ),
        latent_dim: int = 256,
        n_blocks: int = 1,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.beta = beta
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.MaxPool2d(7, 1, 3),
            ResNetEncoderLayer(N_CHANNELS, 8),  # IMAGE_RESIZE_TO = 512 -> 256
            nn.MaxPool2d(7, 1, 3),
            ResNetEncoderLayer(8, 8),  # -> 128
            nn.MaxPool2d(7, 1, 3),
            ResNetEncoderLayer(8, 16),  # -> 64
            ResNetEncoderLayer(16, 16),  # -> 32
            ResNetEncoderLayer(16, 32),  # -> 16
            ResNetEncoderLayer(32, 64),  # -> 8
            ResNetEncoderLayer(64, 128),  # -> 4
            ResNetEncoderLayer(128, latent_dim),  # -> 2
            ResNetEncoderLayer(latent_dim, 2 * latent_dim),  # -> 1
            nn.Flatten(),
        )
        # self.encoder, _ = resnet_encoder(
        #     in_channels=input_shape[0],
        #     out_channels=[
        #         8,  # IMAGE_RESIZE_TO = 512 -> 256
        #         8,  # -> 128
        #         16,  # -> 64
        #         16,  # -> 32
        #         32,  # -> 16
        #         64,  # -> 8
        #         128,  # -> 4
        #         latent_dim,  # -> 2
        #         2 * latent_dim,  # -> 1
        #     ],
        #     n_blocks=n_blocks,
        #     input_size=input_shape[1],
        # )
        self.decoder = resnet_decoder(
            in_channels=latent_dim,
            out_channels=[
                latent_dim,  # 1 -> 2
                128,  # -> 4
                64,  # -> 8
                32,  # -> 16
                16,  # -> 32
                16,  # -> 64
                8,  # -> 128
                8,  # -> 256
                input_shape[0],  # -> 512 = IMAGE_RESIZE_TO
            ],
            n_blocks=n_blocks,
        )
        self.example_input_array = torch.zeros([32, *input_shape])
        self.forward(self.example_input_array)

    def configure_optimizers(self):
        """Override"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes a set of points in the latent space."""
        if z.ndim == 2:  # Batch of (flat) vectors
            latent_dim = self.hparams["latent_dim"]
            z = z.reshape(-1, latent_dim, 1, 1)
        z = z.to(self.device)  # type: ignore
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encodes a set of inputs. Returns a `DiagonalGaussianDistribution` of
        shape `(N, latent_dim, 1, 1)`. The interesting properties of a
        `DiagonalGaussianDistribution` object `d` are:
        * `d.mean`, `d.var`, `d.logvar`, `d.std`: all of shape `(N,
            latent_dim, 1, 1)` with gradients;
        * `d.kl()` of shape `(N, 1)` with gradients;
        * `d.sample()` of shape `(N, latent_dim, 1, 1)` with gradients.
        """
        x = x.to(self.device)  # type: ignore
        z = self.encoder(x)  # Returns a (batch of) flat vector!!
        latent_dim = self.hparams["latent_dim"]
        z = z.reshape(-1, 2 * latent_dim, 1, 1)
        return DiagonalGaussianDistribution(z)

    def evaluate(
        self, x: torch.Tensor, stage: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the reconstruction loss, the KL-loss, and finally the total
        loss, which is

            loss_rec + beta * loss_kl

        The tensors are returned in this order:

            loss, loss_rec, loss_kl
        """
        q_z_x = self.encode(x)
        x_hat = self.decode(q_z_x.sample())
        p_x_z = DiagonalGaussianDistribution(
            torch.concatenate(
                [x_hat, torch.zeros_like(x_hat)],
                dim=1,
            )
        )
        loss_rec = p_x_z.nll(x).mean()
        loss_kl = q_z_x.kl().mean()
        loss = loss_rec + self.beta * loss_kl
        if stage:
            self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
            self.log_dict(
                {
                    f"{stage}/loss_kl": loss_kl,
                    f"{stage}/loss_rec": loss_rec,
                },
                sync_dist=True,
            )
        return loss, loss_rec, loss_kl

    def forward(self, x: Tensor, *_, **__) -> Tensor:
        """
        Encodes, samples, and decodes a batch of inputs. The output shape is
        thus `(N, L)`, where `L` is the dimension of the latent space.
        """
        z = self.encode(x).sample()
        x_hat = self.decode(z)
        return x_hat

    def training_step(self, x: Tensor, *_, **__) -> Tensor:
        """Override"""
        return self.evaluate(x, "train")[0]

    def validation_step(self, x: Tensor, *_, **__) -> Tensor:
        """Override"""
        return self.evaluate(x, "val")[0]

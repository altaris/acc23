"""Variational autoencoder"""
__docformat__ = "google"

from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from diffusers import AutoencoderKL
from diffusers.models.vae import DiagonalGaussianDistribution
from loguru import logger as logging

from acc23.constants import IMAGE_RESIZE_TO, N_CHANNELS


class VAE(pl.LightningModule):
    """Variational autoencoder"""

    beta: float
    latent_shape: List[int]
    vae: AutoencoderKL

    # pylint: disable=unused-argument
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (
            N_CHANNELS,
            IMAGE_RESIZE_TO,
            IMAGE_RESIZE_TO,
        ),
        hidden_channels: int = 32,
        n_blocks: int = 8,
        latent_channels: int = 256,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.beta = beta
        bocs = [hidden_channels] * n_blocks
        self.vae = AutoencoderKL(
            in_channels=input_shape[0],
            out_channels=input_shape[0],
            latent_channels=latent_channels,
            down_block_types=(["DownEncoderBlock2D"] * n_blocks),
            block_out_channels=bocs,
            up_block_types=(["UpDecoderBlock2D"] * n_blocks),
        )
        self.example_input_array = torch.zeros([32, *input_shape])
        zd = self.encode(self.example_input_array)
        self.latent_shape = list(zd.std.shape)[1:]
        logging.info("VAE latent shape: {}", self.latent_shape)

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
        return self.vae.decode(z).sample

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encodes a set of inputs. Returns a `DiagonalGaussianDistribution`. The
        interesting properties of a `DiagonalGaussianDistribution` object `d`
        are:
        * `d.mean`, `d.var`, `d.logvar`, `d.std`: all of shape `(N,
            latent_channels, H_latent, W_latent)` with gradients;
        * `d.kl()` of shape `(N, 1)` with gradients;
        * `d.sample()` of shape `(N, latent_channels, H_latent, W_latent)`
          without gradient of course.
        """
        return self.vae.encode(x).latent_dist

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
        Encodes, samples, and flattens a batch of inputs. The output shape is
        thus `(N, L)`, where `L` is the dimension of the latent space.
        """
        return self.encode(x).sample().flatten(1)

    def training_step(self, x: Tensor, *_, **__) -> Tensor:
        """Override"""
        return self.evaluate(x, "train")[0]

    def validation_step(self, x: Tensor, *_, **__) -> Tensor:
        """Override"""
        return self.evaluate(x, "val")[0]

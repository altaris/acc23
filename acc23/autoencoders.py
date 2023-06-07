"""Variational autoencoder"""
__docformat__ = "google"

from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torchvision
from diffusers.models.vae import DiagonalGaussianDistribution
from torch import Tensor, nn
from transformers.models.resnet.modeling_resnet import ResNetBasicLayer

from acc23.constants import IMAGE_SIZE, N_CHANNELS


class GenerateCallback(pl.Callback):
    """
    A callback to reconstruct images and log them to tensorboard during
    training.
    """

    every_n_epochs: int
    imgs: Tensor

    def __init__(self, imgs: Tensor, every_n_epochs: int = 10):
        """
        Args:
            imgs (Tensor): Batch of images to reconstruct during training
            every_n_epochs (int):
        """
        super().__init__()
        self.imgs = imgs.cpu()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if not (
            trainer.current_epoch > 0
            and trainer.current_epoch % self.every_n_epochs == 0
        ):
            return
        with torch.no_grad():
            rimgs = pl_module(self.imgs).detach().cpu()
        imgs = torch.stack([self.imgs, rimgs], dim=1)
        imgs = imgs.flatten(0, 1)
        grid = torchvision.utils.make_grid(imgs, nrow=2, pad_value=1)
        trainer.logger.experiment.add_image(  # type: ignore
            f"recons/{pl_module.__class__.__name__.lower()}",
            grid,
            global_step=trainer.global_step,
        )


class AE(pl.LightningModule):
    """Dataset's image autoencoder for latent representation"""

    decoder_b: nn.Module
    encoder: nn.Module
    encoded_size: int

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (
            N_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        latent_dim: int = 32,
    ) -> None:
        # TODO: assert that the input shape is square
        super().__init__()
        self.save_hyperparameters()
        bnhc = 8
        cs = [
            input_shape[0],
            bnhc,  # 512 -> 256
            bnhc,  # -> 128
            2 * bnhc,  # -> 64
            2 * bnhc,  # -> 32
            4 * bnhc,  # -> 16
            4 * bnhc,  # -> 8
            # 8 * bnhc,  # -> 4
            # 8 * bnhc,  # -> 2
        ]
        self.encoded_size = input_shape[1] // (2 ** (len(cs) - 1))
        encoder_layers: List[nn.Module] = []
        decoder_layers: List[nn.Module] = []
        for i in range(len(cs) - 1):
            encoder_layers += [
                # nn.Conv2d(
                #     cs[i],
                #     cs[i + 1],
                #     kernel_size=3,
                #     stride=2,
                #     padding=1,
                # ),
                # nn.GELU(),
                ResNetBasicLayer(cs[i], cs[i + 1], stride=2, activation="gelu")
            ]
            decoder_layers = [
                nn.ConvTranspose2d(
                    cs[i + 1],
                    cs[i],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.GELU() if i > 0 else nn.Sigmoid(),
            ] + decoder_layers
        encoder_layers += [
            nn.Flatten(),
            nn.Linear((self.encoded_size**2) * cs[-1], latent_dim),
        ]
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder_a = nn.Sequential(
            nn.Linear(latent_dim, (self.encoded_size**2) * cs[-1]),
            nn.GELU(),
        )
        self.decoder_b = nn.Sequential(*decoder_layers)
        self.example_input_array = torch.zeros((32, *input_shape))
        self.forward(self.example_input_array)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=5, min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }

    def decode(self, z: Tensor) -> Tensor:
        """Decodes a latent vector into an image"""
        z = z.to(self.device)  # type: ignore
        z = self.decoder_a(z)
        z = z.reshape(z.shape[0], -1, self.encoded_size, self.encoded_size)
        return self.decoder_b(z)

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

    # pylint: disable=missing-function-docstring
    def training_step(self, img: Tensor, *_, **__) -> Tensor:
        return self.evaluate(img, "train")

    # pylint: disable=missing-function-docstring
    def validation_step(self, img: Tensor, *_, **__) -> Tensor:
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
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        latent_dim: int = 256,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.beta = beta
        bnhc = 8
        cs = [
            input_shape[0],
            bnhc,  # 512 -> 256
            bnhc,  # -> 128
            2 * bnhc,  # -> 64
            2 * bnhc,  # -> 32
            4 * bnhc,  # -> 16
            4 * bnhc,  # -> 8
        ]
        encoder_layers, decoder_layers = [], []
        for i in range(len(cs) - 1):
            encoder_layers += [
                nn.Conv2d(
                    cs[i],
                    cs[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.GELU(),
            ]
            decoder_layers += [
                nn.GELU() if i > 0 else nn.Sigmoid(),
                nn.ConvTranspose2d(
                    cs[i + 1],
                    cs[i],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        encoder_layers += [
            nn.Flatten(),
            nn.Linear(8 * 8 * 4 * bnhc, 2 * latent_dim),
        ]
        decoder_layers = list(reversed(decoder_layers))
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.example_input_array = torch.zeros([32, *input_shape])
        self.forward(self.example_input_array)

    # pylint: disable=missing-function-docstring
    def configure_optimizers(self):
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
            z = z.reshape(z.shape[0], -1, 1, 1)
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

    # pylint: disable=missing-function-docstring
    def training_step(self, x: Tensor, *_, **__) -> Tensor:
        return self.evaluate(x, "train")[0]

    # pylint: disable=missing-function-docstring
    def validation_step(self, x: Tensor, *_, **__) -> Tensor:
        return self.evaluate(x, "val")[0]

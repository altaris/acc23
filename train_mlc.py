# pylint: disable=missing-function-docstring
# pylint: disable=unnecessary-lambda-assignment
"""Script to train acc23's current model implementation"""

from datetime import datetime
from functools import partial

from loguru import logger as logging
from torch import Tensor
import pytorch_lightning as pl

from acc23.autoencoders import AE, VAE
from acc23.dataset import ACCDataset
from acc23.models import Gordon as Model  # SET CORRECT MODEL CLASS HERE
from acc23.postprocessing import (
    evaluate_on_test_dataset,
    evaluate_on_train_dataset,
)
from acc23.utils import last_checkpoint_path, train_model


def main():

    pl.seed_everything(42, workers=True)

    name = Model.__name__.lower()
    logging.info("Training model '{}'", name)

    if name == "dexter":

        def _ae_encode(ae: AE, x: Tensor) -> Tensor:
            z = ae.encode(x.unsqueeze(0)).flatten()
            return z

        ckpt = last_checkpoint_path(
            "out/tb_logs/autoencoder/version_1/checkpoints/"
        )
        logging.info("Loading autoencoder from", ckpt)
        ae = AE.load_from_checkpoint(ckpt)
        ae.eval()
        ae.requires_grad_(False)
        image_transform = partial(_ae_encode, ae)
        model = Model(ae_latent_dim=ae.hparams["latent_dim"])

    elif name == "farzad":

        def _vae_encode(vae: VAE, x: Tensor) -> Tensor:
            z = vae.encode(x.unsqueeze(0)).sample().flatten()
            return z

        ckpt = last_checkpoint_path("out/tb_logs/vae/version_2/checkpoints/")
        logging.info("Loading vae from", ckpt)
        vae = VAE.load_from_checkpoint(ckpt)
        vae.eval()
        vae.requires_grad_(False)
        image_transform = partial(_vae_encode, vae)
        model = Model(vae_latent_dim=vae.hparams["latent_dim"])

    else:
        model, image_transform = Model(), None

    ds = ACCDataset(
        "data/train.pre.csv",
        "data/images",
        image_transform,
        load_csv_kwargs={"preprocess": False, "impute": False},
    )
    train, val = ds.train_test_split_dl(ratio=0.9)
    model = train_model(
        model,
        train,
        val,
        root_dir="out",
        name=name,
        early_stopping_kwargs={
            "check_finite": True,
            "mode": "max",
            "monitor": "val/f1",
            "patience": 25,
        },
    )

    dt = datetime.now().strftime("%Y-%m-%d-%H-%M")

    df = evaluate_on_test_dataset(
        model,
        "data/test.csv",
        "data/images",
        image_transform,
    )
    path = f"out/{dt}.{name}.test.csv"
    df.to_csv(path, index=False)
    logging.info("Saved test set prediction to '{}'", path)

    df, metrics = evaluate_on_train_dataset(
        model,
        "data/train.csv",
        "data/images",
        image_transform,
    )
    logging.debug(
        "Metrics on train set:\n{}",
        metrics,
    )
    path = f"out/{dt}.{name}.train.csv"
    df.to_csv(path, index=False)
    logging.info("Saved train set prediction to '{}'", path)
    path = f"out/{dt}.{name}.train.metrics.csv"
    metrics.to_csv(path, index=True)
    logging.info("Saved train set prediction metrics to '{}'", path)


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Oh no :(")

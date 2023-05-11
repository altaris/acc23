# pylint: disable=missing-function-docstring
# pylint: disable=unnecessary-lambda-assignment
"""Script to train acc23's current model implementation"""

from datetime import datetime
from functools import partial

import torch
from loguru import logger as logging
from torch import Tensor

from acc23.autoencoders import AE, VAE
from acc23.dataset import ACCDataset
from acc23.models import Farzad as Model  # SET CORRECT MODEL CLASS HERE
from acc23.postprocessing import evaluate_on_test_dataset
from acc23.utils import last_checkpoint_path, train_model


def main():
    name = Model.__name__.lower()

    if name == "dexter":

        def _ae_encode(ae: AE, x: Tensor) -> Tensor:
            z = ae.encode(x.unsqueeze(0)).flatten()
            return z

        ckpt = last_checkpoint_path(
            "out/tb_logs/autoencoder/version_1/checkpoints/"
        )
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
        vae = VAE.load_from_checkpoint(ckpt)
        vae.eval()
        vae.requires_grad_(False)
        image_transform = partial(_vae_encode, vae)
        model = Model(vae_latent_dim=vae.hparams["latent_dim"])

    else:
        model, image_transform = Model(), None

    ds = ACCDataset("data/train.csv", "data/images", image_transform)
    # ds = ACCDataset(
    #     "data/train.processed.csv",
    #     "data/images",
    #     image_transform,
    #     load_csv_kwargs={"preprocess": False, "impute": False},
    # )
    train, val = ds.test_train_split_dl()
    model = train_model(
        model,
        train,
        val,
        root_dir="out",
        name=name,
        early_stopping_kwargs={
            "monitor": "val/f1",
            "patience": 25,
            "mode": "max",
        },
    )

    df = evaluate_on_test_dataset(
        model, "data/test.csv", "data/images", image_transform
    )
    dt = datetime.now().strftime("%Y-%m-%d-%H-%M")
    path = f"out/{dt}.{name}.csv"
    df.to_csv(path, index=False)
    logging.info("Saved test set prediction to '{}'", path)


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Oh no :(")

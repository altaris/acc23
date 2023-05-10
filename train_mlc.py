# pylint: disable=missing-function-docstring
# pylint: disable=unnecessary-lambda-assignment
"""Script to train acc23's current model implementation"""

from datetime import datetime

from loguru import logger as logging

from acc23.ae import Autoencoder
from acc23.dataset import ACCDataset
from acc23.models import Farzad as Model  # SET CORRECT MODEL CLASS HERE
from acc23.postprocessing import evaluate_on_test_dataset
from acc23.utils import last_checkpoint_path, train_model
from acc23.vae import VAE


def main():
    name = Model.__name__.lower()
    if name == "dexter":
        ckpt = last_checkpoint_path(
            "out/tb_logs/autoencoder/version_1/checkpoints/"
        )
        ae = Autoencoder.load_from_checkpoint(ckpt)
        ae.eval()
        image_transform = lambda x: ae.encode(x.unsqueeze(0))[0]
        model = Model(ae_latent_dim=ae.hparams["latent_space_dim"])
    elif name == "farzad":
        ckpt = last_checkpoint_path("out/tb_logs/vae/version_0/checkpoints/")
        vae = VAE.load_from_checkpoint(ckpt)
        vae.eval()
        ls = vae.latent_shape
        ld = ls[0] * ls[1] * ls[2]
        image_transform = lambda x: vae(x.unsqueeze(0))[0]
        model = Model(vae_latent_dim=ld)
    else:
        model, image_transform = Model(), None
    ds = ACCDataset("data/train.csv", "data/images", image_transform)
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

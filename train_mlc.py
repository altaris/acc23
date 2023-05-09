# pylint: disable=missing-function-docstring
"""Script to train acc23's current model implementation"""

from datetime import datetime

from loguru import logger as logging

from acc23.dataset import ACCDataset
from acc23.models import Ampere as Model  # SET CORRECT MODEL CLASS HERE
from acc23.postprocessing import evaluate_on_test_dataset
from acc23.utils import last_checkpoint_path, train_model


def main():

    name = Model.__name__.lower()
    if name == "dexter":
        # Model requires autoencoder
        # TODO: Could read latent dim from hparams.yaml
        model = Model(ae_latent_dim=128)
        ae_ckpt = last_checkpoint_path(
            "out/tb_logs/autoencoder/version_1/checkpoints/"
        )
    else:
        model, ae_ckpt = Model(), None

    ds = ACCDataset("data/train.csv", "data/images", autoencoder_ckpt=ae_ckpt)
    train, val = ds.test_train_split_dl()
    model = train_model(
        model,
        train,
        val,
        root_dir="out",
        name=name,
        early_stopping_kwargs={
            "monitor": "val/f1",
            "patience": 20,
            "mode": "max",
        },
    )

    df = evaluate_on_test_dataset(model, "data/test.csv", "data/images", ae_ckpt)
    dt = datetime.now().strftime("%Y-%m-%d-%H-%M")
    path = f"out/{dt}.{name}.csv"
    df.to_csv(path, index=False)
    logging.info("Saved test set prediction to '{}'", path)


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Oh no :(")

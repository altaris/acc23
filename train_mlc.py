# pylint: disable=missing-function-docstring
"""Script to train acc23's current model implementation"""

from pathlib import Path
from loguru import logger as logging

from acc23.dataset import ACCDataset
from acc23.models import Dexter as Model
from acc23.utils import last_checkpoint_path, train_model


def main():
    ds = ACCDataset("data/train.csv", "data/images")
    train, val = ds.test_train_split_dl()
    # model = Model()
    model = Model(
        last_checkpoint_path(
            Path("out/tb_logs/autoencoder/version_1/checkpoints")
        )
    )
    name = model.__class__.__name__.lower()
    train_model(
        model,
        train,
        val,
        root_dir="out",
        name=name,
        early_stopping_kwargs={
            "monitor": "val/f1",
            "patience": 10,
            "mode": "max",
        },
    )


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Oh no :(")

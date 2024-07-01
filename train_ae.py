# pylint: disable=missing-function-docstring
"""Script to train acc23's autoencoder"""

from loguru import logger as logging

from acc23.autoencoders import AE, GenerateCallback
from acc23.utils import train_model
from acc23.dataset import ImageFolderDataset


def main():
    ds = ImageFolderDataset("data/images")
    train, val = ds.train_test_split_dl(
        ratio=0.9,
        # dataloader_kwargs={
        #     "batch_size": 256,
        #     "pin_memory": True,
        #     "num_workers": 32,
        # },
    )
    model = AE()
    train_model(
        model,
        train,
        val,
        root_dir="out",
        name=model.__class__.__name__.lower(),
        additional_callbacks=[GenerateCallback(ds.sample(4))],
        # early_stopping_kwargs={
        #     "check_finite": True,
        #     "mode": "min",
        #     "monitor": "val/loss",
        #     "patience": 20,
        # },
    )


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Oh no :(")

# pylint: disable=missing-function-docstring
"""Script to train acc23's autoencoder"""

from loguru import logger as logging

from acc23.ae import Autoencoder
from acc23.utils import train_model
from acc23.dataset import ImageFolderDataset


def main():
    ds = ImageFolderDataset("data/images")
    train, val = ds.test_train_split_dl()
    model = Autoencoder(
        out_channels=[
            8,  # IMAGE_RESIZE_TO = 512 -> 256
            8,  # -> 128
            16,  # -> 64
            16,  # -> 32
            32,  # -> 16
            32,  # -> 8
            64,  # -> 4
            64,  # -> 2
            128,  # -> 1
        ],
        n_blocks=1,
        latent_space_dim=128,
    )
    name = model.__class__.__name__.lower()
    train_model(model, train, val, root_dir="out", name=name)


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Oh no :(")

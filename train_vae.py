# pylint: disable=missing-function-docstring
"""Script to train acc23's variational autoencoder"""

from loguru import logger as logging

from acc23.autoencoders import VAE
from acc23.utils import train_model
from acc23.dataset import ImageFolderDataset


def main():
    ds = ImageFolderDataset("data/images")
    train, val = ds.test_train_split_dl()
    model = VAE()
    name = model.__class__.__name__.lower()
    train_model(model, train, val, root_dir="out", name=name)


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Oh no :(")

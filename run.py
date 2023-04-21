"""Script to train acc23's current model implementation"""

import sys

from loguru import logger as logging

from acc23.model import ACCModel
from acc23.dataset import ACCDataset
from acc23.utils import train_model

def main():
    name = sys.argv[1]
    ds = ACCDataset("data/train.csv", "data/images")
    train, val = ds.test_train_split_dl()
    model = ACCModel()
    train_model(model, train, val, root_dir="out", name=name)

if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Oh no :(")

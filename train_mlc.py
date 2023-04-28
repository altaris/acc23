# pylint: disable=missing-function-docstring
"""Script to train acc23's current model implementation"""

from datetime import datetime
from pathlib import Path
from loguru import logger as logging

from acc23.dataset import ACCDataset
from acc23.models import Ampere as Model
# from acc23.models import Dexter as Model
from acc23.utils import last_checkpoint_path, train_model
from acc23.postprocessing import evaluate_on_test_dataset


def main():
    ds = ACCDataset("data/train.csv", "data/images")
    train, val = ds.test_train_split_dl()
    model = Model()
    # model = Model(
    #     last_checkpoint_path(
    #         Path("out/tb_logs/autoencoder/version_6/checkpoints")
    #     )
    # )
    name = model.__class__.__name__.lower()
    model = train_model(
        model,
        train,
        val,
        root_dir="out",
        name=name,
        early_stopping_kwargs={
            "monitor": "val/f1",
            "patience": 15,
            "mode": "max",
        },
    )
    df = evaluate_on_test_dataset(model, "data/test.csv", "data/images")
    dt = datetime.now().strftime("%Y-%m-%d-%H-%M")
    path = f"out/{dt}.{name}.csv"
    df.to_csv(path, index=False)
    logging.info("Saved test set prediction to '{}'", path)


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Oh no :(")

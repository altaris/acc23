# pylint: disable=missing-function-docstring
# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import
"""Script to train acc23's current model implementation"""

from datetime import datetime
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from loguru import logger as logging
import torch
from acc23.constants import TARGETS

from acc23.dataset import ACCDataModule
from acc23.models import *
from acc23.postprocessing import output_to_dataframe
from acc23.utils import train_model


@rank_zero_only
def evaluate_model(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
) -> None:
    trainer = pl.Trainer(
        callbacks=[pl.callbacks.RichProgressBar()],
        devices=1,
        num_nodes=1,
    )
    trainer.test(model, datamodule=datamodule)
    results = trainer.predict(model, datamodule=datamodule)

    df = output_to_dataframe(torch.cat(results)).astype(int)
    raw = pd.read_csv("data/test.csv")
    df["trustii_id"] = raw["trustii_id"]
    df = df[["trustii_id"] + TARGETS]  # Columns order

    dt = datetime.now().strftime("%Y-%m-%d-%H-%M")
    name = model.__class__.__name__.lower()
    path = f"test.out/{dt}.{name}.test.csv"
    df.to_csv(path, index=False)
    logging.info("Saved test set prediction to '{}'", path)


def main():
    model = Norway()  # SET CORRECT MODEL CLASS HERE
    datamodule = ACCDataModule()
    model = train_model(
        model,
        datamodule,
        root_dir="out",
        early_stopping_kwargs={
            "check_finite": True,
            "mode": "min",
            "monitor": "val/loss",
            "patience": 20,
        },
        max_epochs=100,
    )
    evaluate_model(model, datamodule)

    # df, metrics = evaluate_on_train_dataset(
    #     model,
    #     "data/train.csv",
    #     "data/images",
    # )
    # logging.debug(
    #     "Metrics on train set:\n{}",
    #     metrics,
    # )
    # logging.debug(
    #     "Metrics on train set, basic statistics:\n{}",
    #     metrics.describe().drop(
    #         columns=["prev_true", "prev_pred"],
    #         errors="ignore",
    #     ),
    # )
    # path = f"out/{dt}.{name}.train.csv"
    # df.to_csv(path, index=False)
    # logging.info("Saved train set prediction to '{}'", path)
    # path = f"out/{dt}.{name}.train.metrics.csv"
    # metrics = metrics.reset_index(names=["target"])
    # metrics.to_csv(path, index=True)
    # logging.info("Saved train set prediction metrics to '{}'", path)


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    try:
        main()
    except:
        logging.exception("Oh no :(")

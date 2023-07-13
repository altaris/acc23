# pylint: disable=missing-function-docstring
# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import
"""Script to train acc23's current model implementation"""

from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
import pytorch_lightning as pl
from loguru import logger as logging
from pytorch_lightning.utilities import rank_zero_only

from acc23 import (
    ACCDataModule,
    eval_on_test_dataset,
    eval_on_train_dataset,
    models,
)
from acc23.utils import train_model


def _save_df(df: pd.DataFrame, path: Union[str, Path], name: str) -> None:
    """Convenience function to safe a dataframe and log an info message"""
    df.to_csv(path, index=False)
    logging.info("Saved {} to '{}'", name, path)


@rank_zero_only
def evaluate_model(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
) -> None:
    dt = datetime.now().strftime("%Y-%m-%d-%H-%M")
    name = model.__class__.__name__.lower()

    df = eval_on_test_dataset(model, datamodule, root_dir="out/eval/test")
    _save_df(df, f"out/{dt}.{name}.test.csv", "test set predictions")

    df, metrics = eval_on_train_dataset(model, datamodule, "out/eval/train")
    logging.debug(
        "Metrics on train set:\n{}",
        metrics,
    )
    logging.debug(
        "Metrics on train set, basic statistics:\n{}",
        metrics.describe().drop(
            columns=["prev_true", "prev_pred"],
            errors="ignore",
        ),
    )
    _save_df(df, f"out/{dt}.{name}.train.csv", "train set predictions")
    _save_df(
        metrics, f"out/{dt}.{name}.metrics.csv", "train set prediction metrics"
    )


def main():
    model = models.Orchid()  # SET CORRECT MODEL CLASS HERE
    datamodule = ACCDataModule()
    model = train_model(
        model,
        datamodule,
        root_dir="out",
        early_stopping_kwargs={
            "check_finite": True,
            "mode": "max",
            "monitor": "val/f1",
            "patience": 20,
            "min_delta": 1e-2,
        },
        max_epochs=100,
    )
    evaluate_model(model, datamodule)


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    try:
        main()
    except:
        logging.exception("Oh no :(")

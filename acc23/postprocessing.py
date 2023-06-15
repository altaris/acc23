"""
Everything related to post processing, i.e. going from the raw outputs of a
model to a submittable csv file
"""
__docformat__ = "google"

import json
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import requests
import torch
from loguru import logger as logging
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor

from acc23.constants import TARGETS, TRUE_TARGETS
from acc23.dataset import ACCDataModule
from acc23.preprocessing import reorder_columns, set_fake_targets


def eval_on_test_dataset(
    model: pl.LightningModule,
    dm: pl.LightningDataModule,
    root_dir: Union[str, Path],
    raw_test_csv_file_path: Union[str, Path] = "data/test.csv",
) -> pd.DataFrame:
    """
    Returns a trustii-submittable dataframe containing predictions on the
    predict dataset of the datamodule.

    Args:
        model (pl.LightningModule):
        dm (pl.LightningDataModule):
        root_dir (Union[str, Path]): For Pytorch Lightning use
        raw_test_csv_file_path (Union[str, Path]): Path to the competition
            `test.csv` (most likely `data/test.csv`). This is used to get the
            `trustii_id` colunm.
    """
    trainer = pl.Trainer(
        callbacks=[pl.callbacks.RichProgressBar()],
        devices=1,
        num_nodes=1,
        default_root_dir=root_dir,
    )
    raw = trainer.predict(model, datamodule=dm)
    df = output_to_dataframe(torch.cat(raw))  # type: ignore
    df["trustii_id"] = pd.read_csv(raw_test_csv_file_path)["trustii_id"]
    df = df[["trustii_id"] + TARGETS]  # Columns order
    df = df.astype(int)
    return df


def eval_on_train_dataset(
    model: pl.LightningModule,
    dm: ACCDataModule,
    root_dir: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns prediction on the training+validation dataset. Also returns a
    second dataframe containing various metrics.

    Args:
        model (pl.LightningModule):
        dm (pl.LightningDataModule):
        root_dir (Union[str, Path]): For Pytorch Lightning use
    """
    trainer = pl.Trainer(
        callbacks=[pl.callbacks.RichProgressBar()],
        devices=1,
        num_nodes=1,
        default_root_dir=root_dir,
    )
    # The test dataset is actually train.csv without oversampling
    dm.prepare_data()
    dm.setup("test")
    raw = trainer.predict(model, dm.test_dataloader())
    df_pred = output_to_dataframe(torch.cat(raw))  # type: ignore
    assert dm.ds_test is not None
    df_true = dm.ds_test.data[TARGETS]
    df_true = df_true.where(df_true.notna(), df_pred[TARGETS])
    n, kw = len(df_pred), {"zero_division": 0}
    metrics = pd.DataFrame(
        [
            [
                df_true[t].sum() / n,
                df_pred[t].sum() / n,
                f1_score(df_true[t], df_pred[t], **kw),
                precision_score(df_true[t], df_pred[t], **kw),
                recall_score(df_true[t], df_pred[t], **kw),
                ((1 - df_pred[t]) * (1 - df_true[t])).sum()
                / (1 - df_true[t]).sum(),
                accuracy_score(df_true[t], df_pred[t]),
            ]
            for t in TARGETS
        ],
        columns=["prev_true", "prev_pred", "f1", "prec", "rec", "spec", "acc"],
        index=TARGETS,
    )
    return df_pred, metrics


def output_to_dataframe(y: Tensor) -> pd.DataFrame:
    """
    Converts a raw model output logits, which is a `(N, TT)` tensor, with `TT`
    being the number of true targets (see `acc23.preprocessing.TRUE_TARGETS`),
    to a pandas dataframe. The 'fake' targets of `acc23.preprocessing.TARGETS`
    are recalculated here. The `trustii_id` and `Patient_ID` columns are not
    added.

    The order of the target columns is the same as the order in
    `acc23.preprocessing.TARGETS`.
    """
    arr = y.cpu().detach().numpy()
    arr = (arr > 0).astype(int)
    df = pd.DataFrame(data=arr, columns=TRUE_TARGETS)
    df = set_fake_targets(df)
    return reorder_columns(df).astype(int)


def submit_to_trustii(
    csv_file_path: Union[str, Path],
    ipynb_file_path: Union[str, Path],
    trustii_challenge_id: int,
    token: str,
) -> None:
    """
    Submits a CSV file and a IPYNB file to trustii. This code is essentially a
    copypaste from the trustii tutorial notebook.
    """
    endpoint_url = (
        "https://api.trustii.io/api/ds/notebook/datasets/"
        f"{trustii_challenge_id}/prediction"
    )
    with open(csv_file_path, "rb") as fp:
        csv_data = fp.read()
    with open(ipynb_file_path, "rb") as fp:
        ipynb_data = fp.read()
    headers = {"Trustii-Api-User-Token": token}
    data = {
        "csv_file": ("predictions.csv", csv_data),
        "ipynb_file": ("notebook.ipynb", ipynb_data),
    }
    response = requests.post(
        endpoint_url, headers=headers, files=data, timeout=100
    )
    if response.status_code == 200:
        logging.success("Predictions submitted to trustii")
        document = json.loads(response.text)
        error = document["data"].get("error")
        if error:
            logging.error("Submission rejected: {}", error)
        else:
            scores = document["data"]["publicScore"]
            for k in sorted(list(scores.keys())):
                logging.info("{}: {}", k, scores[k])
    else:
        logging.error(
            "Submission failed: {} {}", response.status_code, response.text
        )

"""
Everything related to post processing, i.e. going from the raw outputs of a
model to a submittable csv file
"""
__docformat__ = "google"

from collections import OrderedDict
import json
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import requests
import torch
from loguru import logger as logging
from rich.progress import track
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader

from acc23.constants import TARGETS, TRUE_TARGETS
from acc23.dataset import ACCDataset, Transform_t
from acc23.preprocessing import load_csv, set_fake_targets


def evaluate_on_dataset(
    model: nn.Module,
    data: Union[str, Path, pd.DataFrame],
    image_dir_path: Union[str, Path],
    image_transform: Optional[Transform_t] = None,
    load_csv_kwargs: Optional[dict] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Simply evaluates a model on a dataset"""
    ds = ACCDataset(data, image_dir_path, image_transform, load_csv_kwargs)
    dl = DataLoader(ds, batch_size=batch_size)
    with torch.no_grad():
        y = []
        for x, _, img in track(dl, "Evaluating..."):
            out = model(x, img)
            y.append(out[0] if isinstance(out, tuple) else out)
    return output_to_dataframe(torch.cat(y))


def evaluate_on_test_dataset(
    model: nn.Module,
    csv_file_path: Union[str, Path],
    image_dir_path: Union[str, Path],
    image_transform: Optional[Transform_t] = None,
    load_csv_kwargs: Optional[dict] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Evaluates a model on a test dataset, and returns a submittable dataframe
    (with all the target columns and the `trustii_id` column). The dataset at
    `csv_file_path` is assumed to have a `trustii_id` column.

    Make sure that the input csv file is **NOT** preprocessed.
    """
    df = evaluate_on_dataset(
        model,
        csv_file_path,
        image_dir_path,
        image_transform,
        load_csv_kwargs,
        batch_size,
    )
    raw = pd.read_csv(csv_file_path)
    raw[TARGETS] = df[TARGETS]
    return raw


def evaluate_on_train_dataset(
    model: nn.Module,
    csv_file_path: Union[str, Path],
    image_dir_path: Union[str, Path],
    image_transform: Optional[Transform_t] = None,
    load_csv_kwargs: Optional[dict] = None,
    batch_size: int = 32,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates a model on a training dataset. The dataset is assumed to have all
    target columns. Also returns a dataframe containing various performance
    metrics on true target predictions.
    """
    df_pred = evaluate_on_dataset(
        model,
        csv_file_path,
        image_dir_path,
        image_transform,
        load_csv_kwargs,
        batch_size,
    )
    targets = TARGETS  # Can also use TRUE_TARGETS
    df_true = load_csv(  # TODO: this is not pretty
        csv_file_path,
        preprocess=(load_csv_kwargs or {}).get("preprocess", True),
        impute=False,
    )
    df_true = df_true[targets]
    # Replace nan targets with predictions
    df_true = df_true.where(df_true.notna(), df_pred[targets])
    n = len(df_pred)
    kw = {"zero_division": 0}
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
            for t in targets
        ],
        columns=["prev_true", "prev_pred", "f1", "prec", "rec", "spec", "acc"],
        index=targets,
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
    data = OrderedDict.fromkeys(TARGETS)
    for i, t in enumerate(TRUE_TARGETS):
        data[t] = arr[:, i]
    df = pd.DataFrame(data=data)
    df = set_fake_targets(df)
    return df[TARGETS].astype(int)


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

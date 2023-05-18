"""
Everything related to post processing, i.e. going from the raw outputs of a
model to a submittable csv file
"""
__docformat__ = "google"

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

from acc23.constants import TARGETS
from acc23.dataset import ACCDataset, Transform_t


def evaluate_on_dataset(
    model: nn.Module,
    csv_file_path: Union[str, Path],
    image_dir_path: Union[str, Path],
    image_transform: Optional[Transform_t] = None,
    load_csv_kwargs: Optional[dict] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Simply evaluates a model on a dataset"""
    ds = ACCDataset(
        csv_file_path,
        image_dir_path,
        image_transform,
        load_csv_kwargs,
    )
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
    if "trustii_id" in raw.columns:
        ids = raw["trustii_id"]
    else:
        ids = pd.Series(range(len(df)), name="trustii_id")
    df = pd.concat([ids, df], axis=1)
    df.set_index("trustii_id")
    return df


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
    target columns. Also returns a metric dataframe.
    """
    df = evaluate_on_dataset(
        model,
        csv_file_path,
        image_dir_path,
        image_transform,
        load_csv_kwargs,
        batch_size,
    )
    tgt = pd.read_csv(csv_file_path)[TARGETS]
    n = len(df)
    metrics = pd.DataFrame(
        [
            [
                tgt[t].sum() / n,
                df[t].sum() / n,
                f1_score(tgt[t], df[t], zero_division=0),
                precision_score(tgt[t], df[t], zero_division=0),
                recall_score(tgt[t], df[t], zero_division=0),
                ((1 - df[t]) * (1 - tgt[t])).sum() / (1 - tgt[t]).sum(),
                accuracy_score(tgt[t], df[t]),
            ]
            for t in TARGETS
        ],
        columns=["prev_true", "prev_pred", "f1", "prec", "rec", "spec", "acc"],
        index=TARGETS,
    )
    return df, metrics


def output_to_dataframe(y: Tensor) -> pd.DataFrame:
    """
    Converts a raw model output logits, which is a `(N, T)` tensor, with `T`
    being the number of targets (see `acc23.preprocessing.TARGETS`), to a
    pandas dataframe. The 'fake' targets of `acc23.preprocessing.TARGETS` are
    recalculated here. The `trustii_id` and `Patient_ID` columns are not added.
    """

    arr = y.cpu().detach().numpy()
    arr = (arr > 0).astype(int)
    df = pd.DataFrame(data=arr, columns=TARGETS)
    # df["Allergy_Present"] = df.sum(axis=1).clip(0, 1)
    # df["Respiratory_Allergy"] = (
    #     df[
    #         [
    #             "Type_of_Respiratory_Allergy_ARIA",
    #             "Type_of_Respiratory_Allergy_CONJ",
    #             "Type_of_Respiratory_Allergy_GINA",
    #             "Type_of_Respiratory_Allergy_IGE_Pollen_Gram",
    #             "Type_of_Respiratory_Allergy_IGE_Pollen_Herb",
    #             "Type_of_Respiratory_Allergy_IGE_Pollen_Tree",
    #             "Type_of_Respiratory_Allergy_IGE_Dander_Animals",
    #             "Type_of_Respiratory_Allergy_IGE_Mite_Cockroach",
    #             "Type_of_Respiratory_Allergy_IGE_Molds_Yeast",
    #         ]
    #     ]
    #     .sum(axis=1)
    #     .clip(0, 1)
    # )
    # df["Food_Allergy"] = (
    #     df[
    #         [
    #             "Type_of_Food_Allergy_Aromatics",
    #             "Type_of_Food_Allergy_Other",
    #             "Type_of_Food_Allergy_Cereals_&_Seeds",
    #             "Type_of_Food_Allergy_Egg",
    #             "Type_of_Food_Allergy_Fish",
    #             "Type_of_Food_Allergy_Fruits_and_Vegetables",
    #             "Type_of_Food_Allergy_Mammalian_Milk",
    #             "Type_of_Food_Allergy_Oral_Syndrom",
    #             "Type_of_Food_Allergy_Other_Legumes",
    #             "Type_of_Food_Allergy_Peanut",
    #             "Type_of_Food_Allergy_Shellfish",
    #             "Type_of_Food_Allergy_TPO",
    #             "Type_of_Food_Allergy_Tree_Nuts",
    #         ]
    #     ]
    #     .sum(axis=1)
    #     .clip(0, 1)
    # )
    # df["Venom_Allergy"] = (
    #     df[
    #         [
    #             "Type_of_Venom_Allergy_ATCD_Venom",
    #             "Type_of_Venom_Allergy_IGE_Venom",
    #         ]
    #     ]
    #     .sum(axis=1)
    #     .clip(0, 1)
    # )
    return df


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
        logging.success("Predictions submitted")
        document = json.loads(response.text)
        scores = document["data"]["publicScore"]
        for k in sorted(list(scores.keys())):
            logging.info("{}: {}", k, scores[k])
    else:
        logging.error(
            "Submission failed: {} {}", response.status_code, response.text
        )

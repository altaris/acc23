"""
Everything related to post processing, i.e. going from the raw outputs of a
model to a submittable csv file
"""
__docformat__ = "google"

from pathlib import Path
from typing import Union
import requests

import pandas as pd
import torch
from rich.progress import track
from torch import Tensor, nn
from torch.utils.data import DataLoader

from loguru import logger as logging

from acc23.dataset import ACCDataset
from acc23.constants import TARGETS


def evaluate_on_test_dataset(
    model: nn.Module,
    csv_file_path: Union[str, Path],
    image_dir_path: Union[str, Path],
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Evaluates a model on a dataset, and returns a submittable dataframe (with
    all the target columns and the `trustii_id` column).
    """
    ds = ACCDataset(csv_file_path, image_dir_path)
    dl = DataLoader(ds, batch_size=batch_size)
    with torch.no_grad():
        y = [model(x, img) for x, _, img in track(dl, "Evaluating...")]
    df = output_to_dataframe(torch.cat(y))
    ids = pd.read_csv(csv_file_path)["trustii_id"]
    df = pd.concat([ids, df], axis=1)
    df.set_index("trustii_id")
    return df


def output_to_dataframe(y: Tensor) -> pd.DataFrame:
    """
    Converts a raw model output, which is a `(N, T)` tensor, with `T` being the
    number of targets (see `acc23.preprocessing.TARGETS`), to a pandas
    dataframe. The 'fake' targets of `acc23.preprocessing.TARGETS` are
    recalculated here. The `trustii_id` and `Patient_ID` columns are not added.
    """
    arr = y.cpu().detach().numpy()
    arr = (arr > 0.5).astype(int)
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
        logging.success("Submitted: {}", response.text)
    else:
        logging.error(
            "Submission failed: {} {}", response.status_code, response.text
        )

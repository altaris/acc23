"""
Everything related to post processing, i.e. going from the raw outputs of a
model to a submittable csv file
"""
__docformat__ = "google"

import json
from pathlib import Path
from typing import Union

import pandas as pd
import requests
from loguru import logger as logging
from torch import Tensor

from acc23.constants import TRUE_TARGETS
from acc23.preprocessing import reorder_columns, set_fake_targets


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

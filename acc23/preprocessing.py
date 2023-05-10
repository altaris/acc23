"""Preprocessing stuff"""

import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger as logging
from PIL import Image
from rich.progress import track
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    MultiLabelBinarizer,
    StandardScaler,
)
from sklearn_pandas import DataFrameMapper
from sklearn_pandas.pipeline import make_transformer_pipeline
from torchvision.transforms.functional import resize

from acc23.constants import (
    ALLERGENS,
    CLASSES,
    IMAGE_RESIZE_TO,
    N_CHANNELS,
    TARGETS,
)


class MultiLabelSplitBinarizer(TransformerMixin):
    """
    Essentially applies a MultiLabelBinarizer to column that contains
    comma-separated lists of labels. Plays nice with sklearn-pandas since the
    internal `MultiLabelBinarizer.classes_` is accessible. For some reason
    deriving this class from `MultiLabelBinarizer` directly doesn"t work as
    well...
    """

    _multilabel_binarizer: MultiLabelBinarizer
    _split_delimiters: str

    def __init__(
        self, classes: Optional[list] = None, split_delimiters: str = ","
    ):
        self._multilabel_binarizer = MultiLabelBinarizer(classes=classes)
        self._split_delimiters = split_delimiters

    @property
    def classes_(self) -> Iterable[str]:
        """
        Access the internal `MultiLabelBinarizer.classes_` directly to play
        nice with sklearn-pandas.
        """
        return self._multilabel_binarizer.classes_

    def fit(self, x: np.ndarray, *_, **__) -> "MultiLabelSplitBinarizer":
        """Fits the internal `MultiLabelBinarizer`"""
        s = map_split(x, self._split_delimiters)
        self._multilabel_binarizer.fit(s)
        return self

    def transform(self, x: np.ndarray, *_, **__) -> np.ndarray:
        """
        Transforms the input dataframe using `map_split` and the internal
        `MultiLabelBinarizer`.
        """
        s = map_split(x, self._split_delimiters)
        return self._multilabel_binarizer.transform(s)


def get_dtypes() -> dict:
    """Gets the types the columns of `train.csv` and `test.csv` should have."""
    a = {
        "Patient_ID": str,
        "Chip_Code": str,
        "Chip_Type": str,
        "Chip_Image_Name": str,
        "Age": np.float32,  # spec says it can be nan
        "Gender": np.float32,  # has nan in practice
        "Blood_Month_sample": np.float32,  # spec says it can be nan
        "French_Residence_Department": str,
        "French_Region": str,
        "Rural_or_urban_area": np.float32,  # spec says it can be nan
        "Sensitization": np.uint8,
        "Food_Type_0": str,
        "Food_Type_2": str,  # In the spec but not in the csv files?
        "Treatment_of_rhinitis": str,  # Comma-sep lst of codes
        "Treatment_of_athsma": str,  # Comma-sep lst of codes
        "Age_of_onsets": str,  # Comma-sep lst of age codes
        "Skin_Symptoms": str,  # Will be categorized
        "General_cofactors": str,  # Comma-sep lst of codes
        "Treatment_of_atopic_dematitis": str,  # Comma-sep lst of treatment codes
    }
    b = {allergen: np.float32 for allergen in ALLERGENS}
    c = {target: np.uint8 for target in TARGETS}
    return {**a, **b, **c}


def bruteforce_test_dtypes(csv_file_path: Union[str, Path]):
    """
    Neanderthal-grade method that checks that a CSV file can be loaded with the
    dtypes of `acc23.dtype.DTYPES` by repeatedly loading it with larger and
    larger `acc23.dtype.DTYPES` subdictionaries. Reports when a columns fails
    conversion.
    """
    dtypes = get_dtypes()
    dtypes_items = list(dtypes.items())
    for i in range(1, len(dtypes)):
        try:
            partial_dtypes = dict(dtypes_items[: i + 1])
            pd.read_csv(csv_file_path, dtype=partial_dtypes)
        except Exception as e:
            n, t = list(partial_dtypes.items())[-1]
            logging.error(
                "Column type error: idx={}, name={}, set dtype={}, "
                "error={}",
                i,
                n,
                t,
                e,
            )


def impute_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs various imputation tasks on the dataframe.

    TODO: list all of them
    """
    logging.debug("Imputing dataframe")

    # PMF imputation for allergen columns
    a = df[ALLERGENS].to_numpy()
    df[ALLERGENS] = pmf_impute(
        a,
        30,
        sigma_x=0.1,
        sigma_y=0.1,
        sigma_v=0.1,
        sigma_w=0.1,
    )

    # Simple imputations
    imputers = [
        (["Age"], SimpleImputer()),
        (["Gender"], SimpleImputer(strategy="most_frequent")),
        # (ALLERGENS, KNNImputer()),
    ]
    # Check that non-impute columns don't have nans
    impute_columns: List[str] = []
    for i in imputers:
        c: Union[str, List[str]] = i[0]
        impute_columns += c if isinstance(c, list) else [c]

    # Dummy imputers to retain all the columns
    non_impute_columns = [c for c in df.columns if c not in impute_columns]
    for c in non_impute_columns:
        a, b = df[c].count(), len(df)
        if a != b:
            raise RuntimeError(
                f"Columns '{c}' is marked for non-imputation but it has "
                f"{b - a} / {b} nan values ({(b - a) / b * 100} %)"
            )

    dummy_imputers = [(c, FunctionTransformer()) for c in non_impute_columns]
    # There's some issues if we ask the mapper to return a dataframe
    # (df_out=True): All allergen columns get merged :/ Instead we get an array
    # and reconstruct the dataframe around it being careful with the order of
    # the column names
    mapper = DataFrameMapper(imputers + dummy_imputers)
    x = mapper.fit_transform(df)
    return pd.DataFrame(data=x, columns=impute_columns + non_impute_columns)


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Opens a csv dataframe (presumable `data/train.csv` or `data/test.csv`),
    enforces adequate column types (see `get_dtypes`), and applies some
    preprocessing transforms (see `preprocess_dataframe`).
    """
    logging.debug("Loading dataframe {}", path)
    dtypes = get_dtypes()
    df = pd.read_csv(path, dtype=dtypes)
    # Apparently typing 1 time isn't enough
    df = df.astype({c: t for c, t in dtypes.items() if c in df.columns})
    df = preprocess_dataframe(df)
    df = impute_dataframe(df)
    return df


def load_image(path: Union[str, Path]) -> torch.Tensor:
    """
    Convenience function to load an PNG or BMP image. The returned image tensor
    has shape `(C, H, W)` (the torch/torchvision convention), values from `0`
    to `1`, and dtype `float32`. Here, `C = constants.N_CHANNELS`, and `H = W =
    constants.IMAGE_RESIZE_TO`. In particular, the image is transposed since
    Pillow uses a `(W, H, C)` convention.
    """
    # See https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    fmts = {1: "L", 3: "RGB", 4: "RGBA"}
    if N_CHANNELS not in fmts:
        raise RuntimeError(
            "Invalid constant acc23.constants.N_CHANNELS. Supported values "
            "are 1 (to preprocess raw images into greyscale), 3 (RGB), and "
            "4 (RGBA)"
        )
    img = Image.open(Path(path)).convert(fmts[N_CHANNELS])
    arr = torch.Tensor(np.asarray(img))  # (W, H, C)
    arr = arr.permute(2, 1, 0)  # (C, H, W)
    arr = arr.to(torch.float32)
    arr -= arr.min()
    arr /= arr.max()
    arr = resize(arr, (IMAGE_RESIZE_TO, IMAGE_RESIZE_TO), antialias=True)
    return arr


def map_replace(x: np.ndarray, val: Any, rep: Any) -> np.ndarray:
    """Splits entries of an array of strings."""
    return np.where(x == val, rep, x)


def map_split(x: np.ndarray, delimiters: str = ",") -> Iterable[str]:
    """
    Splits entries of an array of strings. Also removed unnecessary spaces.
    """
    values_to_ignore = ["", "nan"]

    def f(s: str) -> Iterable[str]:
        a = re.split(f"\\s*[{delimiters}]\\s*", s.strip())
        return [b for b in a if b not in values_to_ignore]

    return map(f, x)  # type: ignore


def pmf_impute(
    x: np.ndarray,
    latent_dim: int = 10,
    n_iter: int = 2000,
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
    sigma_v: float = 1.0,
    sigma_w: float = 1.0,
    learning_rate: float = 0.1,
) -> np.ndarray:
    """
    Probabilistic matrix factorization imputation method. Assumes that x is
    zero-mean.

    See also:
        https://github.com/mcleonard/pmf-pytorch/blob/master/Probability%20Matrix%20Factorization.ipynb
    """
    n_samples, n_features = x.shape
    x_true = torch.Tensor(x)
    mask = 1 - torch.isnan(x_true).to(torch.float32)
    x_true = torch.where(mask == 0, 0, x_true)
    y = torch.normal(0, sigma_y, size=(latent_dim, n_samples))
    v = torch.normal(0, sigma_v, size=(latent_dim, n_features))
    w = torch.normal(0, sigma_w, size=(latent_dim, n_features))
    y.requires_grad, v.requires_grad, w.requires_grad = True, True, True
    lambda_y = (sigma_x / sigma_y) ** 2
    lambda_v = (sigma_x / sigma_v) ** 2
    lambda_w = (sigma_x / sigma_w) ** 2
    optimizer = torch.optim.Adam([y, v, w], lr=learning_rate)
    for _ in track(range(n_iter), "PMF imputation"):
        iw = (w @ mask.t()) / mask.sum(-1)
        u = y + iw
        x_pred = u.t() @ v  # No sigmoid (for empirical reasons :P)
        l = torch.sum(mask * (x_true - x_pred) ** 2)
        ry = lambda_y * torch.norm(y, dim=1).sum()
        rv = lambda_v * torch.norm(v, dim=1).sum()
        rw = lambda_w * torch.norm(w, dim=1).sum()
        loss = l + ry + rv + rw
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    x_pred = (u.t() @ v).detach()
    x_true = mask * x_true + (1 - mask) * x_pred
    return x_true.numpy()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all manners of preprocessing transformers to the dataframe.

    TODO: List all of them.
    """
    logging.debug("Preprocessing dataframe")
    general_transforms = [
        (
            ["Chip_Type"],
            MultiLabelBinarizer(classes=CLASSES["Chip_Type"]),
        ),
        ("Chip_Image_Name", FunctionTransformer()),  # identity
        (["Age"], MinMaxScaler()),
        ("Gender", FunctionTransformer()),  # identity
        # ("Blood_Month_sample", LabelBinarizer()),
        (
            ["French_Residence_Department"],
            MultiLabelBinarizer(
                classes=CLASSES["French_Residence_Department"]
            ),
        ),
        (
            ["French_Region"],
            MultiLabelBinarizer(classes=CLASSES["French_Region"]),
        ),
        ("Rural_or_urban_area", FunctionTransformer()),  # identity
        ("Sensitization", FunctionTransformer()),  # identity
        (
            "Food_Type_0",
            MultiLabelSplitBinarizer(classes=CLASSES["Food_Type_0"]),
        ),
        # In the spec but not in the csv files
        # ("Food_Type_2", MultiLabelSplitBinarizer()),
        (
            "Treatment_of_rhinitis",
            make_transformer_pipeline(
                FunctionTransformer(
                    map_replace, kw_args={"val": "0.0", "rep": ""}
                ),
                FunctionTransformer(
                    map_replace, kw_args={"val": "1.0", "rep": "1"}
                ),
                FunctionTransformer(
                    map_replace, kw_args={"val": "2.0", "rep": "2"}
                ),
                FunctionTransformer(
                    map_replace, kw_args={"val": "3.0", "rep": "3"}
                ),
                FunctionTransformer(
                    map_replace, kw_args={"val": "4.0", "rep": "4"}
                ),
                FunctionTransformer(
                    map_replace, kw_args={"val": "5.0", "rep": "5"}
                ),
                FunctionTransformer(
                    map_replace, kw_args={"val": "9.0", "rep": "9"}
                ),
                MultiLabelSplitBinarizer(
                    classes=CLASSES["Treatment_of_rhinitis"],
                    split_delimiters=",. ",
                ),
            ),
        ),
        (
            "Treatment_of_athsma",
            make_transformer_pipeline(
                FunctionTransformer(
                    map_replace, kw_args={"val": "0", "rep": ""}
                ),
                MultiLabelSplitBinarizer(
                    classes=CLASSES["Treatment_of_athsma"],
                    split_delimiters=",. ",
                ),
            ),
        ),
        (
            "Age_of_onsets",
            make_transformer_pipeline(
                FunctionTransformer(
                    map_replace, kw_args={"val": "0", "rep": ""}
                ),
                MultiLabelSplitBinarizer(classes=CLASSES["Age_of_onsets"]),
            ),
        ),
        (
            "Skin_Symptoms",
            MultiLabelSplitBinarizer(classes=CLASSES["Skin_Symptoms"]),
        ),
        (
            "General_cofactors",
            make_transformer_pipeline(
                FunctionTransformer(
                    map_replace, kw_args={"val": "0", "rep": ""}
                ),
                MultiLabelSplitBinarizer(
                    classes=CLASSES["General_cofactors"],
                    split_delimiters=",. ",
                ),
            ),
        ),
        (
            "Treatment_of_atopic_dematitis",
            make_transformer_pipeline(
                FunctionTransformer(
                    map_replace, kw_args={"val": "0", "rep": ""}
                ),
                MultiLabelSplitBinarizer(
                    classes=CLASSES["Treatment_of_atopic_dematitis"],
                    split_delimiters=",. ",
                ),
            ),
        ),
    ]
    allergen_trasforms = [
        ([allergen], StandardScaler()) for allergen in ALLERGENS
    ]
    target_transforms = [
        ([target], FunctionTransformer())
        for target in TARGETS
        if target in df.columns
    ]
    mapper = DataFrameMapper(
        general_transforms + allergen_trasforms + target_transforms,
        df_out=True,
    )
    return mapper.fit_transform(df)

"""Preprocessing stuff"""

import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger as logging
from PIL import Image
from sklearn.base import TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    MultiLabelBinarizer,
)
from sklearn_pandas import DataFrameMapper
from sklearn_pandas.pipeline import make_transformer_pipeline
from torchvision.transforms.functional import resize

from acc23.constants import (
    IMAGE_RESIZE_TO,
    N_CHANNELS,
    TARGETS,
    ALLERGENS,
    CLASSES,
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
    imputers = [
        (["Age"], SimpleImputer()),
        (["Gender"], SimpleImputer(strategy="most_frequent")),
        (ALLERGENS, KNNImputer()),
    ]
    # Check that non-impute columns don't have nans
    impute_columns: List[str] = []
    for i in imputers:
        c: Union[str, List[str]] = i[0]
        impute_columns += c if isinstance(c, list) else [c]
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
        ([allergen], MinMaxScaler()) for allergen in ALLERGENS
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

"""Preprocessing stuff"""

import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger as logging
from PIL import Image, ImageFile
from rich.progress import track
from sklearn.base import TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    MultiLabelBinarizer,
    StandardScaler,
)
from sklearn_pandas import DataFrameMapper
from sklearn_pandas.pipeline import make_transformer_pipeline
from torchvision.transforms.functional import pad, resize

from acc23.constants import (
    CLASSES,
    FEATURES,
    IGES,
    IMAGE_SIZE,
    N_CHANNELS,
    TARGETS,
    TRUE_TARGETS,
)
from acc23.mlsmote import mlsmote

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Some images are buggy yay:
# CY60527_4_190006236104_2022_12_22_12_11_20.bmp


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
    _last_class_is_nan: bool

    def __init__(
        self,
        classes: Optional[list] = None,
        split_delimiters: str = ",",
        last_class_is_nan: bool = False,
    ):
        """
        Args:
            classes (Optional[list]): Leave to `None` to automatically infer
                the classes
            split_delimiters (str): Splitting delimiters, defaults to only `,`
                (so that it deals with comma-separated lists)
            last_class_is_nan (bool): If set to `True`, then the last class
                will treated as the "unknown" class. Here's an example:

                >>> x = ["a,b,c", "a,b", "d", "a,d"]
                >>> f = MultiLabelSplitBinarizer(classes=["a", "b", "c", "d"])
                >>> f.fit_transform(x)
                array([[1, 1, 1, 0],
                       [1, 1, 0, 0],
                       [0, 0, 0, 1],
                       [1, 0, 0, 1]])
                >>> g = MultiLabelSplitBinarizer(classes=["a", "b", "c", "d"], last_class_is_nan=True)
                >>> g.fit_transform(x)
                array([[ 1.,  1.,  1.],
                       [ 1.,  1.,  0.],
                       [nan, nan, nan],
                       [nan, nan, nan]])

                Note that `g.fit_transform(x)` only has 3 columns.
        """
        self._multilabel_binarizer = MultiLabelBinarizer(classes=classes)
        self._split_delimiters = split_delimiters
        self._last_class_is_nan = last_class_is_nan

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
        y = self._multilabel_binarizer.transform(s)
        if self._last_class_is_nan:
            b = y[:, -1] == 1  # Whether row has last class
            b = np.stack([b] * len(list(self.classes_)), axis=-1)
            # b now has the same shape as y, and b[i,j] is True iff x[i] has
            # the last class, i.e. y[i,-1] == 1
            y = np.where(b, np.NaN, y)
            y = y[:, :-1]  # Drop last column
        return y


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
        "Sensitization": np.int64,
        "Food_Type_0": str,
        "Food_Type_2": str,  # In the spec but not in the csv files?
        "Treatment_of_rhinitis": str,  # Comma-sep lst of codes
        "Treatment_of_athsma": str,  # Comma-sep lst of codes
        "Age_of_onsets": str,  # Comma-sep lst of age codes
        "Skin_Symptoms": str,  # Will be categorized
        "General_cofactors": str,  # Comma-sep lst of codes
        "Treatment_of_atopic_dematitis": str,  # Comma-sep lst of treatment codes
    }
    b = {ige: np.float32 for ige in IGES}
    c = {target: np.float32 for target in TARGETS}  # Some targets have nans
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

    Warning:
        In the case of KNN imputation, the order of the columns is changed!
    """
    logging.debug("Imputing dataframe")

    # PMF imputation for IgE columns
    # df[IGES] = pmf_impute(
    #     df[IGES].to_numpy(),
    #     latent_dim=512,
    #     sigma_x=0.1,
    #     sigma_y=0.1,
    #     sigma_v=0.1,
    #     sigma_w=0.1,
    # )

    # Simple imputations
    imputers = [
        (["Age"], SimpleImputer()),
        (["Gender"], SimpleImputer(strategy="most_frequent")),
        # Columns with NaN classes, aka "unknown" classes, aka 9.
        # (
        #     None,
        #     KNNImputer(),
        # ),
        (IGES, KNNImputer()),
    ]
    # Some targets have NaNs, which is absurd. Fortunately, most true
    # targets don't, except for Severe_Allergy (1323/2989 NaNs, 44.26%)
    # if TARGETS[0] in df.columns:
    #     imputers.append((TARGETS, SimpleImputer(strategy="most_frequent")))
    impute_columns: List[str] = []
    for i in imputers:
        c: Union[str, List[str]] = i[0]
        impute_columns += c if isinstance(c, list) else [c]

    # Check that non-impute columns don't have NaNs
    non_impute_columns = [c for c in df.columns if c not in impute_columns]
    for c in non_impute_columns:
        a, b = df[c].count(), len(df)
        if a != b:
            logging.warning(
                "Columns '{}' is marked for non-imputation but it has "
                "{} / {} NaN values ({} %)",
                c,
                b - a,
                b,
                round((b - a) / b * 100, 3),
            )

    # Dummy imputers to retain all the columns
    dummy_imputers = [(c, FunctionTransformer()) for c in non_impute_columns]

    # There's some issues if we ask the mapper to return a dataframe
    # (df_out=True): All IgE columns get merged :/ Instead we get an array and
    # reconstruct the dataframe around it being careful with the order of the
    # column names
    mapper = DataFrameMapper(imputers + dummy_imputers)
    x = mapper.fit_transform(df)
    df = pd.DataFrame(data=x, columns=impute_columns + non_impute_columns)
    df = df.infer_objects()
    return reorder_columns(df)


def load_csv(
    path: Union[str, Path],
    preprocess: bool = True,
    impute: bool = True,
    oversample: bool = False,
    n_oversample_rounds: int = 1,
) -> pd.DataFrame:
    """
    Opens a csv dataframe (presumable `data/train.csv` or `data/test.csv`),
    enforces adequate column types (see `acc23.preprocessing.get_dtypes`), and
    applies some preprocessing transforms (see
    `acc23.preprocessing.preprocess_dataframe`).

    Args:
        path (Union[str, Path]): Path of the csv file.
        preprocess (bool): Whether the dataframe should go through
            `acc23.preprocessing.preprocess_dataframe`.
        impute (bool): Whether the dataframe should be imputed (see
            `acc23.preprocessing.impute_dataframe`). Note that imputation is
            not performed if `preprocess=False`.
        oversample (bool): Whether should be oversampled to try and fix class
            imbalance (see `acc23.mlsmote.mlsmote`). Note that oversampling is
            not performed if `preprocess=False` or if `impute=False`.
        n_oversample_rounds (int): Numver of MLSMOTE rounds, i.e. number of
            times MLSMOTE should be applied to the dataset. Defaults to 1.
    """
    logging.debug("Loading dataframe {}", path)
    dtypes = get_dtypes()
    df = pd.read_csv(path)
    # Apparently typing only once isn't enough
    df = df.astype({c: t for c, t in dtypes.items() if c in df.columns})
    if preprocess:
        df = preprocess_dataframe(df)
    else:
        logging.debug("Skipped preprocessing")
    if impute and preprocess:
        df = impute_dataframe(df)
    else:
        logging.debug("Skipped imputation")
    if oversample and impute and preprocess:
        df = mlsmote(df, TRUE_TARGETS, n_rounds=n_oversample_rounds)
    else:
        logging.debug("Skipped oversampling")
    return reorder_columns(df)


def load_image(path: Union[str, Path]) -> torch.Tensor:
    """
    Convenience function to load an PNG or BMP image. The returned image tensor
    has shape `(C, H, W)` (the torch/torchvision convention) and dtype
    `float32`. Here, `C = constants.N_CHANNELS`, and `H = W =
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
    arr /= 255
    s = 1400
    _, h, w = arr.shape
    arr = pad(arr, (0, 0, s - w, s - h), float(arr.mean()))
    arr = resize(arr, (IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    # arr = normalize(arr, [0] * N_CHANNELS, [1] * N_CHANNELS)
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


def preprocess_dataframe(
    df: pd.DataFrame,
    drop_nan_targets: bool = True,
) -> pd.DataFrame:
    """
    Applies all manners of preprocessing transformers to the dataframe.

    Args:
        df (DataFrame):
        drop_nan_targets (bool): Drop the rows where at least one true target
            is NaN or 9.

    TODO: List all transformers
    """
    logging.debug("Preprocessing dataframe")

    # General preprocessing
    general_transforms = [
        (
            ["Chip_Type"],
            MultiLabelBinarizer(classes=CLASSES["Chip_Type"]),
        ),
        (
            "Chip_Image_Name",
            make_transformer_pipeline(
                FunctionTransformer(
                    map_replace, kw_args={"val": "nan", "rep": None}
                ),
                FunctionTransformer(
                    map_replace, kw_args={"val": "NaN", "rep": None}
                ),
                FunctionTransformer(
                    map_replace, kw_args={"val": "", "rep": None}
                ),
            ),
        ),
        (["Age"], MinMaxScaler(feature_range=(0, 150))),
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
        (
            "Rural_or_urban_area",
            FunctionTransformer(
                # map_replace, kw_args={"val": 9, "rep": np.NaN}
            ),
        ),
        ("Sensitization", FunctionTransformer()),  # identity
        (
            "Food_Type_0",
            MultiLabelSplitBinarizer(classes=CLASSES["Food_Type_0"]),
        ),
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
                    map_replace, kw_args={"val": "9.0", "rep": "9"}
                ),
                MultiLabelSplitBinarizer(
                    classes=CLASSES["Treatment_of_rhinitis"],
                    split_delimiters=",. ",
                    # last_class_is_nan=True,
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
                    # last_class_is_nan=True,
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
            MultiLabelSplitBinarizer(
                classes=CLASSES["Skin_Symptoms"],
                # last_class_is_nan=True,
            ),
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
                    # last_class_is_nan=True,
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
                    # last_class_is_nan=True,
                ),
            ),
        ),
    ]
    iges_trasforms = [([ige], StandardScaler()) for ige in IGES]
    target_transforms = [
        (
            [target],
            FunctionTransformer(
                map_replace, kw_args={"val": 9, "rep": np.NaN}
            ),
        )
        for target in TARGETS
        if target in df.columns
    ]
    mapper = DataFrameMapper(
        general_transforms + iges_trasforms + target_transforms,
        df_out=True,
    )
    df = mapper.fit_transform(df)

    if drop_nan_targets:
        if all(t in df.columns for t in TRUE_TARGETS):
            no_nan_tgt = df[TRUE_TARGETS].notna().prod(axis=1) == 1
            a, b = no_nan_tgt.sum(), len(df)
            logging.debug(
                "Dropping rows with at least one NaN true target "
                "({} / {} rows)",
                a,
                b,
                round(a / b * 100, 3),
            )
            df = df[no_nan_tgt].reset_index(drop=True)
        else:
            logging.warning(
                "drop_nan_targets set to True, but dataframe does not have "
                "all true target columns. Skipping drops"
            )

    df = df.infer_objects()
    return reorder_columns(df)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorders the columns of the dataframe to `FEATURES + TARGETS`, where
    `FEATURES` and `TARGETS` are defined in `acc23.constants`. The input
    dataframe is allowed to have missing columns.

    The goal of this method is to make dataframe's columns order predictable,
    which is important because models may convert input dataframes to arrays
    (essentially removing the columns names).
    """
    return df[[c for c in FEATURES + TARGETS if c in df.columns]]


def set_fake_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the 'fake' targets from the true targets. The true target columns
    cannot have nans.
    """

    # Some targets have 0 prevalence lôl
    for c in [
        "Type_of_Food_Allergy_Other",
        "Type_of_Food_Allergy_Cereals_&_Seeds",
    ]:
        df[c] = df.get(c, 0)
    df["Allergy_Present"] = df.sum(axis=1).clip(0, 1)
    df["Respiratory_Allergy"] = df.get("Respiratory_Allergy", 0) + df[
        [
            "Type_of_Respiratory_Allergy_ARIA",
            "Type_of_Respiratory_Allergy_CONJ",
            "Type_of_Respiratory_Allergy_GINA",
            "Type_of_Respiratory_Allergy_IGE_Pollen_Gram",
            "Type_of_Respiratory_Allergy_IGE_Pollen_Herb",
            "Type_of_Respiratory_Allergy_IGE_Pollen_Tree",
            "Type_of_Respiratory_Allergy_IGE_Dander_Animals",
            "Type_of_Respiratory_Allergy_IGE_Mite_Cockroach",
            "Type_of_Respiratory_Allergy_IGE_Molds_Yeast",
        ]
    ].sum(axis=1).clip(0, 1)
    df["Food_Allergy"] = df.get("Food_Allergy", 0) + df[
        [
            "Type_of_Food_Allergy_Aromatics",
            "Type_of_Food_Allergy_Other",
            "Type_of_Food_Allergy_Cereals_&_Seeds",
            "Type_of_Food_Allergy_Egg",
            "Type_of_Food_Allergy_Fish",
            "Type_of_Food_Allergy_Fruits_and_Vegetables",
            "Type_of_Food_Allergy_Mammalian_Milk",
            "Type_of_Food_Allergy_Oral_Syndrom",
            "Type_of_Food_Allergy_Other_Legumes",
            "Type_of_Food_Allergy_Peanut",
            "Type_of_Food_Allergy_Shellfish",
            "Type_of_Food_Allergy_TPO",
            "Type_of_Food_Allergy_Tree_Nuts",
        ]
    ].sum(axis=1).clip(0, 1)
    df["Venom_Allergy"] = df.get("Venom_Allergy", 0) + df[
        [
            "Type_of_Venom_Allergy_ATCD_Venom",
            "Type_of_Venom_Allergy_IGE_Venom",
        ]
    ].sum(axis=1).clip(0, 1)
    return reorder_columns(df)

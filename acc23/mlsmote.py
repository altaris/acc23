"""
MLSMOTE oversampling algorithm from

    F. Charte et al., MLSMOTE: Approaching imbalanced multilabel learning
    through synthetic instance generation, Knowl. Based Syst. (2015),
    http://dx.doi.org/10.1016/j.knosys.2015.07.019

"""
__docformat__ = "google"
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from loguru import logger as logging
from rich.progress import track
from sklearn.neighbors import NearestNeighbors

from acc23.constants import TARGETS


def _mlsmote(
    df: pd.DataFrame,
    targets: List[str],
    n_neighbors: int = 10,
    sampling_factor: Union[int, Literal["irlbl"]] = 10,
) -> pd.DataFrame:
    """
    One round MLSMOTE oversampling.

    Args:
        df (pd.DataFrame):
        targets (List[str]):
        n_neighbors (int):
        sampling_factor (int): See `mlsmote`

    Warning:
        Assumes that the whole dataframe is numerical and has no `NaN`s.
    """
    columns, dtypes = df.columns, dict(df.dtypes)
    mean_ir = irlbl(df[targets]).mean()
    for tgt in track(targets, "Applying MLSMOTE..."):
        ir = pd.Series(irlbl(df[targets]), index=targets)
        if ir[tgt] <= mean_ir:
            continue
        min_bag, synth_smpls = df.loc[df[tgt] == 1], []
        n = len(min_bag)
        m = n * (
            sampling_factor
            if isinstance(sampling_factor, int)
            else int(ir[tgt])
        )
        # logging.debug("Minority target '{}', bag size = {}, irlbl = {}, new samples = {}", tgt, n, ir[tgt], m)
        knn = NearestNeighbors(n_neighbors=min(n_neighbors, n))
        knn.fit(min_bag.drop(columns=TARGETS).to_numpy())
        for i in np.random.choice(n, m):
            sample = min_bag.iloc[i]
            idx = knn.kneighbors(
                [sample.drop(TARGETS)], return_distance=False
            )[0]
            synth_smpl = _new_sample(
                sample=sample,
                ref_neigh=min_bag.iloc[np.random.choice(idx)],
                neighbors=min_bag.iloc[idx],
                targets=targets,
            )
            synth_smpls.append(synth_smpl)
        df = pd.concat([df, pd.DataFrame(synth_smpls)], ignore_index=True)
    return df[columns].astype(dtypes)  # Restore column order and dtypes


def _new_sample(
    sample: pd.Series,
    ref_neigh: pd.Series,
    neighbors: pd.DataFrame,
    targets: List[str],
) -> pd.Series:
    """Creates a synthetic sample"""
    synth_smpl = {}
    for c, d in neighbors.dtypes.items():
        if c in targets:
            h = (len(neighbors) + (1 if sample[c] == 1 else -1)) / 2
            synth_smpl[c] = int(neighbors[c].sum() > h)
        elif str(d).startswith("int"):  # c is a categorical feature
            synth_smpl[c] = neighbors[c].value_counts().index[0]
        else:  # c is a numerical feature
            u, v, r = ref_neigh[c], sample[c], np.random.rand()
            synth_smpl[c] = v + r * (u - v)
    return pd.Series(synth_smpl)


def _prevalence_summary(
    df_old: pd.DataFrame, df_new: pd.DataFrame, targets: List[str]
) -> None:
    """
    Logs a summary of the target prevalence changes between two dataframes
    """
    kw = {"axis": 0, "skipna": True}
    p_o = pd.Series(df_old[targets].mean(**kw), name="prev_old")
    p_n = pd.Series(df_new[targets].mean(**kw), name="prev_new")
    p_d = pd.Series((p_n - p_o) / p_o * 100, name="prev_diff (%)").round(3)
    n_o = pd.Series(df_old[targets].sum(**kw), name="n_old", dtype=int)
    n_n = pd.Series(df_new[targets].sum(**kw), name="n_new", dtype=int)
    n_d = pd.Series((n_n - n_o) / n_o * 100, name="n_diff (%)").round(3)
    s = pd.concat([p_o, p_n, p_d, n_o, n_n, n_d], axis=1)
    logging.debug("Prevalence summary:\n{}", s)


def irlbl(targets: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    IRLbl imbalance score.

    Args:
        df (Union[pd.DataFrame, np.ndarray]): Target / labels dataframe

    Returns:
        A numpy array of shape `(n_targets,)`

    Warning:
        All targets are expected to be represented at least once. In other
        words, every target should have a strictly positive prevalence.
    """
    if isinstance(targets, pd.DataFrame):
        targets = targets.to_numpy()
    s = targets.sum(axis=0)
    s = s.max() / (s + 1e-5)
    return s


def mlsmote(
    df: pd.DataFrame,
    targets: List[str],
    n_neighbors: int = 10,
    sampling_factor: Union[int, Literal["irlbl"]] = 10,
    apply_remedial: bool = False,
    remedial_threshold: Union[int, Literal["mean"]] = "mean",
) -> pd.DataFrame:
    """
    REMEDIAL-MLSMOTE oversampling. Note that object-type features are ignored
    in calculations, and synthetic samples will have these features set to
    `None`.

    Args:
        df (pd.DataFrame):
        targets (List[str]):
        n_neighbors (int):
        sampling_factor (Union[int, Literal["irlbl"]]): In the MLSMOTE stage,
            for each minority bag `b`, will generate `sampling_factor * len(b)`
            synthetic samples. If `sampling_factor` is `irlbl`, the (rounded)
            IRLbl score will be used as factor.
        apply_remedial (bool): Whether to apply REMEDIAL before oversampling
        remedial_threshold (Union[int, Literal["mean"]]): See `remedial`

    Warning:
        Numerical columns must not contain `NaN`s
    """
    df_old = df.copy()
    if apply_remedial:
        df = remedial(df, targets, remedial_threshold)
    obj_columns = [c for c, d in df.dtypes.items() if str(d) == "object"]
    if obj_columns:
        logging.debug(
            "Setting {} object columns aside: {}",
            len(obj_columns),
            obj_columns,
        )
        df, df_objs = df.drop(columns=obj_columns), df[obj_columns]
    df = _mlsmote(df, targets, n_neighbors, sampling_factor)
    if obj_columns:
        df_objs = pd.concat(
            [
                df_objs,
                pd.DataFrame(
                    [[None] * len(obj_columns)] * (len(df) - len(df_objs)),
                    columns=obj_columns,
                ),
            ],
            ignore_index=True,
        )
        df = pd.concat([df, df_objs], axis=1)
    df[targets] = df[targets].astype(int)
    _prevalence_summary(df_old, df, targets)
    return df


def remedial(
    df: pd.DataFrame,
    targets: List[str],
    threshold: Union[int, Literal["mean"]] = "mean",
) -> pd.DataFrame:
    """
    REMEDIAL decoupling algorithm of

        Charte, F., Rivera, A., del Jesus, M.J., Herrera, F. (2015). Resampling
        Multilabel Datasets by Decoupling Highly Imbalanced Labels. In: Onieva,
        E., Santos, I., Osaba, E., Quintián, H., Corchado, E. (eds) Hybrid
        Artificial Intelligent Systems. HAIS 2015. Lecture Notes in Computer
        Science(), vol 9121. Springer, Cham.
        https://doi.org/10.1007/978-3-319-19644-2_41

    Args:
        df (pd.DataFrame):
        targets (List[str]):
        threshold (Union[int, Literal["mean"]]): Either a percentile (between 0
            and 100) or the literal `"mean"`.
    """
    df, ir, sc = df.copy(), irlbl(df[targets]), scumble(df[targets])
    ir_mean = ir.mean()
    thresh = sc.mean() if threshold == "mean" else np.percentile(sc, threshold)
    new_rows = []
    for i, j in track(enumerate(df.index), "Applying REMEDIAL..."):
        if sc[i] <= thresh:
            continue
        new = df.loc[j].copy()
        df.loc[j, np.array(targets)[ir <= ir_mean]] = 0
        new[np.array(targets)[ir > ir_mean]] = 0
        new_rows.append(new)
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


def scumble(targets: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    SCUMBLE label concurrence score of

        Charte, Francisco, et al. "REMEDIAL-HwR: Tackling multilabel imbalance
        through label decoupling and data resampling hybridization."
        Neurocomputing 326 (2019): 110-122.

    The returned array contains the score of each individual sample. The score
    of a sample is contained in $[0, 1]$. A higher score means that the sample
    containes both majority and minority labels. To obtain the SCRUMBLE score
    of the while dataset, simply take the mean of the returned array.

    Args:
        targets (Union[pd.DataFrame, np.ndarray]): Target / labels dataframe.

    Returns:
        A numpy array of shape `(n_samples,)`

    Warning:
        The paper's SCUMBLE formula is wrong... This implementation is base on
        the R implementation
        https://github.com/fcharte/mldr/blob/master/R/measures.R#L67
        Incidentally, all papers I could find using SCUMBLE and citing the
        paper above also have the formula wrong. But don't forget, pEEr ReVieW
        ENsuuuUUUUuREs PapER QuALIty.
    """

    def div(
        a: Union[int, float, np.ndarray], b: np.ndarray, default: float = 0.0
    ) -> np.ndarray:
        """
        Convenience function compute `a / b`, but the result is `default` where
        `b == 0`.
        """
        c = a / np.where(b == 0, 1, b)
        return np.where(b == 0, default, c)

    if isinstance(targets, pd.DataFrame):
        targets = targets.to_numpy()
    ir = irlbl(targets)
    ir_grid = targets * ir  # (i,j) -> irlbl or tgt j if i has j, or 0
    ir_prod = np.where(ir_grid == 0, 1, ir_grid).prod(axis=1)
    lbl_cnt = targets.sum(axis=1)  # i -> num of labels of sample i
    ir_bar = div(ir_grid.sum(axis=1), lbl_cnt)
    return 1 - div(np.power(ir_prod, div(1, lbl_cnt, 1)), ir_bar)

"""
MLSMOTE oversampling algorithm from

    F. Charte et al., MLSMOTE: Approaching imbalanced multilabel learning
    through synthetic instance generation, Knowl. Based Syst. (2015),
    http://dx.doi.org/10.1016/j.knosys.2015.07.019

"""
__docformat__ = "google"

from typing import List

import numpy as np
import pandas as pd
from loguru import logger as logging
from rich.progress import track
from sklearn.neighbors import NearestNeighbors


def _mlsmote(
    df: pd.DataFrame, targets: List[str], n_neighbors: int = 5
) -> pd.DataFrame:
    """
    One round MLSMOTE oversampling.

    Warning:
        Assumes that the whole dataframe is numerical and has no `NaN`s.
    """
    mir = _irlbl(df[targets]).mean()
    for tgt in targets:
        ir = _irlbl(df[targets])  # Must be done at every iter.
        if ir[tgt] <= mir:
            continue
        # logging.debug("Minority target '{}'", tgt)
        min_bag, synth_smpls = df.loc[df[tgt] == 1], []
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(min_bag.to_numpy())
        for i in range(len(min_bag)):
            sample = min_bag.iloc[i]
            idx = knn.kneighbors([sample], return_distance=False)[0]
            neighbors = min_bag.iloc[idx]
            ref_neigh = min_bag.iloc[np.random.choice(idx)]
            synth_smpl = _new_sample(sample, ref_neigh, neighbors, targets)
            synth_smpls.append(synth_smpl)
        df = pd.concat([df, pd.DataFrame(synth_smpls)], ignore_index=True)
    return df


def _irlbl(df: pd.DataFrame) -> pd.DataFrame:
    """
    IRLbl imbalance score.

    Args:
        df (DataFrame): Target / labels dataframe

    Warning:
        All targets are expected to be represented at least once. In other
        words, every target should have a strictly positive prevalence.
    """
    s = df.sum(axis=0)
    s = s.max() / (s + 1e-5)
    return s


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


def mlsmote(
    df: pd.DataFrame,
    targets: List[str],
    n_neighbors: int = 5,
    n_rounds: int = 1,
) -> pd.DataFrame:
    """
    Multi-round MLSMOTE oversampling. Note that object-type features are
    ignored in calculations, and synthetic samples will have these features set
    to `None`.

    Warning:
        Numerical columns must not contain `NaN`s
    """
    columns = df.columns
    obj_columns = [c for c, d in df.dtypes.items() if str(d) == "object"]
    if obj_columns:
        logging.debug(
            "Setting {} object columns aside: {}",
            len(obj_columns),
            obj_columns,
        )
        df, df_objs = df.drop(columns=obj_columns), df[obj_columns]
    for _ in track(range(n_rounds), "MLSMOTE oversampling..."):
        df = _mlsmote(df, targets, n_neighbors)
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
    return df[columns]

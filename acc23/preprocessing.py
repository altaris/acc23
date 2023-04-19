"""Preprocessing stuff"""

import re
from math import nan
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger as logging
from PIL import Image
from sklearn.base import TransformerMixin
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelBinarizer,
    MinMaxScaler,
    MultiLabelBinarizer,
)
from sklearn_pandas import DataFrameMapper
from sklearn_pandas.pipeline import make_transformer_pipeline

ALLERGENS = [
    "Act_d_1",
    "Act_d_2",
    "Act_d_5",
    "Act_d_8",
    "Aln_g_1",
    "Alt_a_1",
    "Alt_a_6",
    "Amb_a_1",
    "Ana_o_2",
    "Ani_s_1",
    "Ani_s_3",
    "Api_g_1",
    "Api_m_1",
    "Api_m_4",
    "Ara_h_1",
    "Ara_h_2",
    "Ara_h_3",
    "Ara_h_6",
    "Ara_h_8",
    "Ara_h_9",
    "Art_v_1",
    "Art_v_3",
    "Asp_f_1",
    "Asp_f_3",
    "Asp_f_6",
    "Ber_e_1",
    "Bet_v_1",
    "Bet_v_2",
    "Bet_v_4",
    "Bla_g_1",
    "Bla_g_2",
    "Bla_g_5",
    "Bla_g_7",
    "Blo_t_5",
    "Bos_d_4",
    "Bos_d_5",
    "Bos_d_6",
    "Bos_d_8",
    "Bos_d_Lactoferrin",
    "Can_f_1",
    "Can_f_2",
    "Can_f_3",
    "Can_f_5",
    "Che_a_1",
    "Cla_h_8",
    "Cor_a_1.0101",
    "Cor_a_1.0401",
    "Cor_a_8",
    "Cor_a_9",
    "Cry_j_1",
    "Cup_a_1",
    "Cyn_d_1",
    "Der_f_1",
    "Der_f_2",
    "Der_p_1",
    "Der_p_10",
    "Der_p_2",
    "Equ_c_1",
    "Equ_c_3",
    "Fag_e_2",
    "Fel_d_1",
    "Fel_d_2",
    "Fel_d_4",
    "Gad_c_1",
    "Gal_d_1",
    "Gal_d_2",
    "Gal_d_3",
    "Gal_d_5",
    "Gly_m_4",
    "Gly_m_5",
    "Gly_m_6",
    "Hev_b_1",
    "Hev_b_3",
    "Hev_b_5",
    "Hev_b_6.01",
    "Hev_b_8",
    "Jug_r_1",
    "Jug_r_2",
    "Jug_r_3",
    "Lep_d_2",
    "Mal_d_1",
    "Mer_a_1",
    "Mus_m_1",
    "MUXF3",
    "Ole_e_1",
    "Ole_e_7",
    "Ole_e_9",
    "Par_j_2",
    "Pen_m_1",
    "Pen_m_2",
    "Pen_m_4",
    "Phl_p_1",
    "Phl_p_11",
    "Phl_p_12",
    "Phl_p_2",
    "Phl_p_4",
    "Phl_p_5",
    "Phl_p_6",
    "Phl_p_7",
    "Pla_a_1",
    "Pla_a_2",
    "Pla_a_3",
    "Pla_l_1",
    "Pol_d_5",
    "Pru_p_1",
    "Pru_p_3",
    "Sal_k_1",
    "Ses_i_1",
    "Tri_a_14",
    "Tri_a_19.0101",
    "Tri_a_aA_TI",
    "Ves_v_5",
    "Cor_a_14",
    "Can_f_4",
    "Hev_b_6",
    "Der_p_23",
    "Alpha-Gal",
    "Ana_o_3",
    "Can_f_6",
    "Aca_m",
    "Aca_s",
    "Ach_d",
    "Act_d_10",
    "Ail_a",
    "All_c",
    "All_s",
    "Aln_g_4",
    "Ama_r",
    "Amb_a",
    "Amb_a_4",
    "Ana_o",
    "Api_g_2",
    "Api_g_6",
    "Api_m",
    "Api_m_10",
    "Ara_h_15",
    "Arg_r_1",
    "Art_v",
    "Asp_f_4",
    "Ave_s",
    "Ber_e",
    "Bet_v_6",
    "Bla_g_4",
    "Bla_g_9",
    "Blo_t_10",
    "Blo_t_21",
    "Bos_d_2",
    "Bos_d_meat",
    "Bos_d_milk",
    "Bro_pa",
    "Cam_d",
    "Can_f_Fd1",
    "Can_f_male_urine",
    "Can_s",
    "Can_s_3",
    "Cap_a",
    "Cap_h_epithelia",
    "Cap_h_milk",
    "Car_c",
    "Car_i",
    "Car_p",
    "Cav_p_1",
    "Che_a",
    "Che_q",
    "Chi_spp.",
    "Cic_a",
    "Cit_s",
    "Cla_h",
    "Clu_h",
    "Clu_h_1",
    "Cor_a_1.0103",
    "Cor_a_11",
    "Cor_a_12_RUO",
    "Cor_a_pollen",
    "Cra_c_6",
    "Cuc_m_2",
    "Cuc_p",
    "Cup_s",
    "Cyn_d",
    "Cyp_c_1",
    "Dau_c",
    "Dau_c_1",
    "Der_p_11",
    "Der_p_20",
    "Der_p_21",
    "Der_p_5",
    "Der_p_7",
    "Dol_spp",
    "Equ_c_4",
    "Equ_c_meat",
    "Equ_c_milk",
    "Fag_e",
    "Fag_s_1",
    "Fel_d_7",
    "Fic_b",
    "Fic_c",
    "Fra_a_1+3",
    "Fra_e",
    "Fra_e_1",
    "Gad_m",
    "Gad_m_1",
    "Gad_m_2+3",
    "Gal_d_4",
    "Gal_d_meat",
    "Gal_d_white",
    "Gal_d_yolk",
    "Gly_d_2",
    "Gly_m_8",
    "Hel_a",
    "Hev_b_11",
    "Hev_b_6.02",
    "Hom_g",
    "Hom_s_LF",
    "Hor_v",
    "Jug_r_4",
    "Jug_r_6",
    "Jug_r_pollen",
    "Jun_a",
    "Len_c",
    "Lit_s",
    "Loc_m",
    "Lol_p_1",
    "Lol_spp.",
    "Lup_a",
    "Mac_i_2S_Albumin",
    "Mac_inte",
    "Mal_d_2",
    "Mal_d_3",
    "Mala_s_11",
    "Mala_s_5",
    "Mala_s_6",
    "Man_i",
    "Mel_g",
    "Mes_a_1_RUO",
    "Mor_r",
    "Mus_a",
    "Myt_e",
    "Ole_e_7_RUO",
    "Ori_v",
    "Ory_c_1",
    "Ory_c_2",
    "Ory_c_3",
    "Ory_s",
    "Ory_meat",
    "Ost_e",
    "Ovi_a_epithelia",
    "Ovi_a_meat",
    "Ovi_a_milk",
    "Pan_b",
    "Pan_m",
    "Pap_s",
    "Pap_s_2S_Albumin",
    "Par_j",
    "Pas_n",
    "Pec_spp.",
    "Pen_ch",
    "Pen_m_3",
    "Per_a",
    "Per_a_7",
    "Pers_a",
    "Pet_c",
    "Pha_v",
    "Phl_p_5.0101",
    "Pho_d_2",
    "Phod_s_1",
    "Phr_c",
    "Pim_a",
    "Pis_s",
    "Pis_v_1",
    "Pis_v_2",
    "Pis_v_3",
    "Pis_v_4_RUO",
    "Pla_l",
    "Pol_d",
    "Pop_n",
    "Pru_av",
    "Pru_du",
    "Pru_p_7_RUO",
    "Pyr_c",
    "Raj_c",
    "Raj_c_Parvalbumin",
    "Rat_n",
    "Rud_spp.",
    "Sac_c",
    "Sal_k",
    "Sal_s",
    "Sal_s_1",
    "Sco_s",
    "Sco_s_1",
    "Sec_c_flour",
    "Sec_c_pollen",
    "Ses_i",
    "Sin",
    "Sin_a_1",
    "Sol_spp.",
    "Sol_t",
    "Sola_l",
    "Sola_l_6",
    "Sus_d_1",
    "Sus_d_epithelia",
    "Sus_d_meat",
    "Ten_m",
    "Thu_a",
    "Thu_a_1",
    "Tri_a_19",
    "Tri_fo",
    "Tri_s",
    "Tyr_p",
    "Tyr_p_2",
    "Ulm_c",
    "Urt_d",
    "Vac_m",
    "Ves_v",
    "Ves_v_1",
    "Vit_v_1",
    "Xip_g_1",
    "Zea_m",
    "Zea_m_14",
]


class MultiLabelSplitBinarizer(TransformerMixin):
    """
    Essentially applies a MultiLabelBinarizer to column that contains
    comma-separated lists of labels. Plays nice with sklearn-pandas since the
    internal `MultiLabelBinarizer.classes_` is accessible. For some reason
    deriving this class from `MultiLabelBinarizer` directly doesn't work as
    well...
    """

    _multilabel_binarizer: MultiLabelBinarizer
    _split_delimiters: str

    def __init__(self, split_delimiters: str = ","):
        self._multilabel_binarizer = MultiLabelBinarizer()
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
        "Skin_Symptoms": np.uint8,
        "General_cofactors": str,  # Comma-sep lst of codes
        "Treatment_of_atopic_dematitis": str,  # Comma-sep lst of treatment codes
        # LABELS ------------------------------------
        "Allergy_Present": np.uint8,  # no nans
        "Severe_Allergy": np.uint8,  # no nans
        "Respiratory_Allergy": np.uint8,  # no nans
        "Food_Allergy": np.uint8,  # no nans
        "Venom_Allergy": np.uint8,  # no nans
        "Type_of_Respiratory_Allergy_ARIA": np.uint8,  # no nans
        "Type_of_Respiratory_Allergy_CONJ": np.uint8,  # no nans
        "Type_of_Respiratory_Allergy_GINA": np.uint8,  # no nans
        "Type_of_Respiratory_Allergy_IGE_Pollen_Gram": np.uint8,  # no nans
        "Type_of_Respiratory_Allergy_IGE_Pollen_Herb": np.uint8,  # no nans
        "Type_of_Respiratory_Allergy_IGE_Pollen_Tree": np.uint8,  # no nans
        "Type_of_Respiratory_Allergy_IGE_Dander_Animals": np.uint8,  # no nans
        "Type_of_Respiratory_Allergy_IGE_Mite_Cockroach": np.uint8,  # no nans
        "Type_of_Respiratory_Allergy_IGE_Molds_Yeast": np.uint8,  # no nans
        "Type_of_Food_Allergy_Aromatics": np.uint8,  # no nans
        "Type_of_Food_Allergy_Other": np.uint8,  # no nans
        "Type_of_Food_Allergy_Cereals_&_Seeds": np.uint8,  # no nans
        "Type_of_Food_Allergy_Egg": np.uint8,  # no nans
        "Type_of_Food_Allergy_Fish": np.uint8,  # no nans
        "Type_of_Food_Allergy_Fruits_and_Vegetables": np.uint8,  # no nans
        "Type_of_Food_Allergy_Mammalian_Milk": np.uint8,  # no nans
        "Type_of_Food_Allergy_Oral_Syndrom": np.uint8,  # no nans
        "Type_of_Food_Allergy_Other_Legumes": np.uint8,  # no nans
        "Type_of_Food_Allergy_Peanut": np.uint8,  # no nans
        "Type_of_Food_Allergy_Shellfish": np.uint8,  # no nans
        "Type_of_Food_Allergy_TPO": np.uint8,  # no nans
        "Type_of_Food_Allergy_Tree_Nuts": np.uint8,  # no nans
        "Type_of_Venom_Allergy_ATCD_Venom": np.uint8,  # no nans
        "Type_of_Venom_Allergy_IGE_Venom": np.uint8,  # no nans
    }

    b = {allergen: np.float32 for allergen in ALLERGENS}

    return {**a, **b}


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
                "Column type error: idx={}, name='{}', set dtype={}, "
                "error='{}'",
                i,
                n,
                t,
                e,
            )


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Opens a csv dataframe (presumable `data/train.csv` or `data/test.csv`),
    enforces adequate column types (see `get_dtypes`), and applies some
    preprocessing transforms (see `preprocess_dataframe`).
    """
    dtypes = get_dtypes()
    df = pd.read_csv(path, dtype=dtypes)
    # Apparently 1 time isn't enough
    df = df.astype({c: t for c, t in dtypes.items() if c in df.columns})
    return preprocess_dataframe(df)


def load_image(path: Union[str, Path]) -> torch.Tensor:
    """
    Convenience function to load an PNG or BMP image. The returned image tensor
    has shape `(C, H, W)`, values from `0` to `1`, and dtype `float32`.
    """
    path = Path(path)
    img = torch.Tensor(np.asarray(Image.open(path)))  # img is channel-last
    if img.ndim == 2:
        img = torch.concat([img.unsqueeze(-1)] * 3, dim=-1)  # Add channel axis
    elif img.shape[-1] >= 4:
        img = img[:, :, :3]  # Remove alpha channel
    img = img.permute(2, 0, 1)  # img is now channel-first
    img = img.to(torch.float32)
    img -= img.min()
    img /= img.max()
    return img


def map_split(x: np.ndarray, delimiters: str = ","):  # TODO type output
    """
    Splits entries of an array of strings. Also removed unnecessary spaces.
    """
    return map(lambda a: re.split(f"\\s*[{delimiters}]\\s*", a.strip()), x)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all manners of preprocessing transformers to the dataframe.

    TODO: List all of them.
    """
    mapper = DataFrameMapper(
        [
            ("Chip_Type", LabelBinarizer()),
            ("Chip_Image_Name", FunctionTransformer()),  # identity
            (["Age"], MinMaxScaler()),
            ("Gender", FunctionTransformer()),  # identity
            # ("Blood_Mounth_sample", FunctionTransformer()),  # TODO
            ("French_Residence_Department", LabelBinarizer()),
            ("French_Region", LabelBinarizer()),
            ("Rural_or_urban_area", FunctionTransformer(replace_9_by_nan)),
            ("Sensitization", FunctionTransformer()),  # identity
            ("Food_Type_0", MultiLabelSplitBinarizer()),
            # ("Food_Type_2", MultiLabelSplitBinarizer()),
            (
                "Treatment_of_rhinitis",
                make_transformer_pipeline(
                    FunctionTransformer(replace_9_by_nan),
                    MultiLabelSplitBinarizer(),
                ),
            ),
            (
                "Treatment_of_athsma",
                make_transformer_pipeline(
                    FunctionTransformer(replace_9_by_nan),
                    MultiLabelSplitBinarizer(),
                ),
            ),
            (
                "Age_of_onsets",
                make_transformer_pipeline(
                    FunctionTransformer(replace_9_by_nan),
                    MultiLabelSplitBinarizer(),
                ),
            ),
            ("Skin_Symptoms", FunctionTransformer(replace_9_by_nan)),
            (
                # TODO: 9 can appear in multilabel cases (e.g. 2262)
                "General_cofactors",
                make_transformer_pipeline(
                    FunctionTransformer(replace_9_by_nan),
                    MultiLabelSplitBinarizer(split_delimiters=",. "),
                ),
            ),
            (
                "Treatment_of_atopic_dematitis",
                make_transformer_pipeline(
                    FunctionTransformer(replace_9_by_nan),
                    MultiLabelSplitBinarizer(split_delimiters=",. "),
                ),
            ),
        ]
        + [([allergen], MinMaxScaler()) for allergen in ALLERGENS],
        df_out=True,
    )
    return mapper.fit_transform(df)


# pylint: disable=unnecessary-lambda-assignment
def replace_9_by_nan(x: np.ndarray) -> np.ndarray:
    """Self explanatory."""  # TODO: maybe explain some more
    n = "nan" if x.dtype == "object" else nan
    predicate = lambda a: a in [9, "9", "9.0"]
    return np.where(list(map(predicate, x)), n, x)  # type: ignore

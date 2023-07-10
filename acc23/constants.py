# pylint: disable=too-many-lines
"""All the constants"""


TARGETS = [
    "Allergy_Present",
    "Severe_Allergy",
    "Respiratory_Allergy",
    "Food_Allergy",
    "Venom_Allergy",
    "Type_of_Respiratory_Allergy_ARIA",
    "Type_of_Respiratory_Allergy_CONJ",
    "Type_of_Respiratory_Allergy_GINA",
    "Type_of_Respiratory_Allergy_IGE_Pollen_Gram",
    "Type_of_Respiratory_Allergy_IGE_Pollen_Herb",
    "Type_of_Respiratory_Allergy_IGE_Pollen_Tree",
    "Type_of_Respiratory_Allergy_IGE_Dander_Animals",
    "Type_of_Respiratory_Allergy_IGE_Mite_Cockroach",
    "Type_of_Respiratory_Allergy_IGE_Molds_Yeast",
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
    "Type_of_Venom_Allergy_ATCD_Venom",
    "Type_of_Venom_Allergy_IGE_Venom",
]
"""
Name of the targets/labels. The order should be the same as in the dataset CSV
files that are provided by the organizers.
"""
N_TARGETS = len(TARGETS)
"""Number of targets"""


TRUE_TARGETS = [
    "Allergy_Present",
    "Severe_Allergy",
    "Respiratory_Allergy",
    "Food_Allergy",
    "Venom_Allergy",
    "Type_of_Respiratory_Allergy_ARIA",
    "Type_of_Respiratory_Allergy_CONJ",
    "Type_of_Respiratory_Allergy_GINA",
    "Type_of_Respiratory_Allergy_IGE_Pollen_Gram",
    "Type_of_Respiratory_Allergy_IGE_Pollen_Herb",
    "Type_of_Respiratory_Allergy_IGE_Pollen_Tree",
    "Type_of_Respiratory_Allergy_IGE_Dander_Animals",
    "Type_of_Respiratory_Allergy_IGE_Mite_Cockroach",
    "Type_of_Respiratory_Allergy_IGE_Molds_Yeast",
    "Type_of_Food_Allergy_Aromatics",
    # "Type_of_Food_Allergy_Other",  # Unrepresented
    # "Type_of_Food_Allergy_Cereals_&_Seeds",  # Unrepresented
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
    "Type_of_Venom_Allergy_ATCD_Venom",
    "Type_of_Venom_Allergy_IGE_Venom",
]
"""
The *true* targets, which exclude `Type_of_Food_Allergy_Other` and
`Type_of_Food_Allergy_Cereals_&_Seeds` since they have 0 prevalence (i.e. no
sample with these labels).
"""
N_TRUE_TARGETS = len(TRUE_TARGETS)
"""Number of true targets"""

TRUE_TARGETS_COUNT = [
    1121,
    938,
    1024,
    601,
    23,
    744,
    373,
    693,
    629,
    418,
    619,
    584,
    596,
    302,
    34,
    42,
    38,
    95,
    25,
    86,
    49,
    108,
    37,
    46,
    142,
    13,
    17,
]
"""
Number of rows that have that label, e.g. `TRUE_TARGETS_COUNT[0]` is the number
of samples that have `Allergy_Present`. This can be obtained by running

```py
import pandas as pd
from acc23.constants import TRUE_TARGETS
from acc23.preprocessing import load_csv
df = load_csv("data/train.csv", impute=False)
df = df[TRUE_TARGETS]
df = df.where(df.notna(), 0)
t = df.to_numpy()
t.sum(axis=0)
```

TODO:
    Don't hardcode this :/
"""

TRUE_TARGETS_IRLBL = [
    0.99999999,
    1.19509594,
    1.09472655,
    1.86522459,
    48.73910924,
    1.50672041,
    3.00536185,
    1.61760459,
    1.78219393,
    2.68181812,
    1.81098543,
    1.91952052,
    1.88087245,
    3.71192041,
    32.97057854,
    26.69046984,
    29.49999224,
    11.79999876,
    44.83998206,
    13.03488221,
    22.87754635,
    10.37962867,
    30.29728911,
    24.36955992,
    7.89436564,
    86.2307029,
    65.94113768,
]
"""
IRLbl score of the true targets. See `acc23.mlsmote.irlbl`. This can be
obtained by running

```py
from acc23 import load_csv, irlbl, TRUE_TARGETS
df = load_csv("data/train.csv", oversample=False)
irlbl(df[TRUE_TARGETS])
```

TODO:
    Don't hardcode this :/
"""

TRUE_TARGETS_PREVALENCE = [
    0.82669617,
    0.69174041,
    0.75516224,
    0.44321534,
    0.01696165,
    0.54867257,
    0.27507375,
    0.51106195,
    0.46386431,
    0.30825959,
    0.45648968,
    0.43067847,
    0.43952802,
    0.22271386,
    0.02507375,
    0.03097345,
    0.0280236,
    0.070059,
    0.01843658,
    0.06342183,
    0.03613569,
    0.07964602,
    0.02728614,
    0.0339233,
    0.10471976,
    0.00958702,
    0.01253687,
]
"""
Target prevalence, e.g. `TRUE_TARGETS_COUNT[0]` is the prevalence
of target `Allergy_Present`. This can be obtained by running

```py
import pandas as pd
from acc23.constants import TRUE_TARGETS
from acc23.preprocessing import load_csv
df = load_csv("data/train.csv", impute=False)
df = df[TRUE_TARGETS]
df = df.where(df.notna(), 0)
t = df.to_numpy()
t.sum(axis=0) / len(t)
```

TODO:
    Don't hardcode this :/
"""


FEATURES = [
    "Chip_Image_Name",
    "Age",
    "Gender",
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
    "Chip_Type_ALEX",
    "Chip_Type_ISAC_V1",
    "Chip_Type_ISAC_V2",
    "French_Residence_Department_deptA",
    "French_Residence_Department_deptAA",
    "French_Residence_Department_deptAAA",
    "French_Residence_Department_deptAAAA",
    "French_Residence_Department_deptB",
    "French_Residence_Department_deptBB",
    "French_Residence_Department_deptBBB",
    "French_Residence_Department_deptBBBB",
    "French_Residence_Department_deptC",
    "French_Residence_Department_deptCC",
    "French_Residence_Department_deptCCC",
    "French_Residence_Department_deptCCCC",
    "French_Residence_Department_deptD",
    "French_Residence_Department_deptDD",
    "French_Residence_Department_deptDDD",
    "French_Residence_Department_deptDDDDD",
    "French_Residence_Department_deptE",
    "French_Residence_Department_deptEE",
    "French_Residence_Department_deptEEE",
    "French_Residence_Department_deptF",
    "French_Residence_Department_deptFF",
    "French_Residence_Department_deptFFF",
    "French_Residence_Department_deptG",
    "French_Residence_Department_deptGG",
    "French_Residence_Department_deptGGG",
    "French_Residence_Department_deptH",
    "French_Residence_Department_deptHH",
    "French_Residence_Department_deptHHH",
    "French_Residence_Department_deptI",
    "French_Residence_Department_deptII",
    "French_Residence_Department_deptIII",
    "French_Residence_Department_deptJ",
    "French_Residence_Department_deptJJ",
    "French_Residence_Department_deptJJJ",
    "French_Residence_Department_deptK",
    "French_Residence_Department_deptKK",
    "French_Residence_Department_deptKKK",
    "French_Residence_Department_deptL",
    "French_Residence_Department_deptLL",
    "French_Residence_Department_deptLLL",
    "French_Residence_Department_deptM",
    "French_Residence_Department_deptMM",
    "French_Residence_Department_deptMMM",
    "French_Residence_Department_deptN",
    "French_Residence_Department_deptNN",
    "French_Residence_Department_deptNNN",
    "French_Residence_Department_deptO",
    "French_Residence_Department_deptOO",
    "French_Residence_Department_deptOOO",
    "French_Residence_Department_deptP",
    "French_Residence_Department_deptPP",
    "French_Residence_Department_deptPPP",
    "French_Residence_Department_deptQ",
    "French_Residence_Department_deptQQ",
    "French_Residence_Department_deptQQQ",
    "French_Residence_Department_deptR",
    "French_Residence_Department_deptRR",
    "French_Residence_Department_deptRRR",
    "French_Residence_Department_deptS",
    "French_Residence_Department_deptSS",
    "French_Residence_Department_deptSSS",
    "French_Residence_Department_deptT",
    "French_Residence_Department_deptTT",
    "French_Residence_Department_deptTTT",
    "French_Residence_Department_deptU",
    "French_Residence_Department_deptUU",
    "French_Residence_Department_deptUUU",
    "French_Residence_Department_deptV",
    "French_Residence_Department_deptVV",
    "French_Residence_Department_deptVVV",
    "French_Residence_Department_deptW",
    "French_Residence_Department_deptWW",
    "French_Residence_Department_deptWWW",
    "French_Residence_Department_deptX",
    "French_Residence_Department_deptXX",
    "French_Residence_Department_deptXXX",
    "French_Residence_Department_deptY",
    "French_Residence_Department_deptYY",
    "French_Residence_Department_deptYYY",
    "French_Residence_Department_deptZ",
    "French_Residence_Department_deptZZ",
    "French_Residence_Department_deptZZZ",
    "French_Region_regionA",
    "French_Region_regionB",
    "French_Region_regionC",
    "French_Region_regionD",
    "French_Region_regionE",
    "French_Region_regionF",
    "French_Region_regionG",
    "French_Region_regionH",
    "French_Region_regionI",
    "French_Region_regionJ",
    "French_Region_regionK",
    "French_Region_regionL",
    "French_Region_regionM",
    "French_Region_regionN",
    "French_Region_regionO",
    "Rural_or_urban_area",
    "Sensitization",
    "Food_Type_0_Egg",
    "Food_Type_0_Fish",
    "Food_Type_0_Mammalian Milk",
    "Food_Type_0_Other",
    "Food_Type_0_Peanut",
    "Food_Type_0_Tree Nuts",
    "Treatment_of_rhinitis_1",
    "Treatment_of_rhinitis_2",
    "Treatment_of_rhinitis_3",
    "Treatment_of_rhinitis_4",
    "Treatment_of_rhinitis_9",
    "Treatment_of_athsma_1",
    "Treatment_of_athsma_2",
    "Treatment_of_athsma_3",
    "Treatment_of_athsma_4",
    "Treatment_of_athsma_5",
    "Treatment_of_athsma_6",
    "Treatment_of_athsma_7",
    "Treatment_of_athsma_8",
    "Treatment_of_athsma_10",
    "Treatment_of_athsma_9",
    "Age_of_onsets_1",
    "Age_of_onsets_2",
    "Age_of_onsets_3",
    "Age_of_onsets_4",
    "Age_of_onsets_5",
    "Age_of_onsets_6",
    "Age_of_onsets_9",
    "Skin_Symptoms_0",
    "Skin_Symptoms_1",
    "Skin_Symptoms_9",
    "General_cofactors_1",
    "General_cofactors_2",
    "General_cofactors_3",
    "General_cofactors_4",
    "General_cofactors_5",
    "General_cofactors_6",
    "General_cofactors_7",
    "General_cofactors_8",
    "General_cofactors_10",
    "General_cofactors_11",
    "General_cofactors_12",
    "General_cofactors_9",
    "Treatment_of_atopic_dematitis_1",
    "Treatment_of_atopic_dematitis_2",
    "Treatment_of_atopic_dematitis_3",
    "Treatment_of_atopic_dematitis_4",
    "Treatment_of_atopic_dematitis_5",
    "Treatment_of_atopic_dematitis_6",
    "Treatment_of_atopic_dematitis_7",
    "Treatment_of_atopic_dematitis_9",
]
"""
Name of the tabular features *after* preprocessing. This list can be obtained
by running

```py
from acc23.preprocessing import load_csv
from acc23.constants import TARGETS
df = load_csv("data/train.csv")
df = df.drop(columns=TARGETS)
list(df.columns)
```
"""


N_FEATURES = len(FEATURES) - 1
"""Number of features **excluding `Chip_Image_Name`**."""

IGES = [
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
"""Features pertaining to Immunoglobulin E (IgE)"""

CLASSES = {
    "Chip_Type": ["ALEX", "ISAC_V1", "ISAC_V2"],
    "French_Residence_Department": [
        "deptA",
        "deptAA",
        "deptAAA",
        "deptAAAA",
        "deptB",
        "deptBB",
        "deptBBB",
        "deptBBBB",
        "deptC",
        "deptCC",
        "deptCCC",
        "deptCCCC",
        "deptD",
        "deptDD",
        "deptDDD",
        "deptDDDDD",
        "deptE",
        "deptEE",
        "deptEEE",
        "deptF",
        "deptFF",
        "deptFFF",
        "deptG",
        "deptGG",
        "deptGGG",
        "deptH",
        "deptHH",
        "deptHHH",
        "deptI",
        "deptII",
        "deptIII",
        "deptJ",
        "deptJJ",
        "deptJJJ",
        "deptK",
        "deptKK",
        "deptKKK",
        "deptL",
        "deptLL",
        "deptLLL",
        "deptM",
        "deptMM",
        "deptMMM",
        "deptN",
        "deptNN",
        "deptNNN",
        "deptO",
        "deptOO",
        "deptOOO",
        "deptP",
        "deptPP",
        "deptPPP",
        "deptQ",
        "deptQQ",
        "deptQQQ",
        "deptR",
        "deptRR",
        "deptRRR",
        "deptS",
        "deptSS",
        "deptSSS",
        "deptT",
        "deptTT",
        "deptTTT",
        "deptU",
        "deptUU",
        "deptUUU",
        "deptV",
        "deptVV",
        "deptVVV",
        "deptW",
        "deptWW",
        "deptWWW",
        "deptX",
        "deptXX",
        "deptXXX",
        "deptY",
        "deptYY",
        "deptYYY",
        "deptZ",
        "deptZZ",
        "deptZZZ",
    ],
    "French_Region": [
        "regionA",
        "regionB",
        "regionC",
        "regionD",
        "regionE",
        "regionF",
        "regionG",
        "regionH",
        "regionI",
        "regionJ",
        "regionK",
        "regionL",
        "regionM",
        "regionN",
        "regionO",
    ],
    "Food_Type_0": [
        "Egg",
        "Fish",
        "Mammalian Milk",
        "Other",
        "Peanut",
        "Tree Nuts",
    ],
    "Treatment_of_rhinitis": [
        "1",
        "2",
        "3",
        "4",
        # "5",  # not in the spec but see train.csv 2637
        "9",  # NaN class
    ],
    "Treatment_of_athsma": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "10",
        "9",  # NaN class
    ],
    "Age_of_onsets": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "9",  # NaN class
    ],
    "Skin_Symptoms": [
        "0",
        "1",
        "9",  # NaN class
    ],
    "General_cofactors": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "10",
        "11",
        "12",
        "9",  # NaN class
    ],
    "Treatment_of_atopic_dematitis": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "9",  # NaN class
    ],
}
"""
Dict mapping a categorical feature (e.g. `Chip_Type`) to the list of all its
possible classes (in this case `ALEX`, `ISAC_V1`, and `ISAC_V2`)
"""

IMAGE_SIZE = 256
"""
By default, images will be resized to `IMAGE_SIZE x IMAGE_SIZE`. See also
`acc23.preprocessing.load_image` and
[`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html).
"""

N_CHANNELS = 3
"""
Number of image channels after preprocessing, see
`acc23.preprocessing.load_image`. Even if this is 1, images will still be
presented as `(C, H, W)` tensors.
"""

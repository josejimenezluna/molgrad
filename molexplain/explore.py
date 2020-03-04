import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.inchi import MolToInchi
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from molexplain.utils import DATA_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH

RDLogger.DisableLog("rdApp.*")


def gen_guide(list_csvs):
    guide = {}
    for csv in tqdm(list_csvs):
        df = pd.read_csv(csv, header=0, sep=";")
        desc = pd.unique(df["Assay Description"])
        chembl_id = pd.unique(df["Assay ChEMBL ID"])

        assert len(desc) == 1 and len(chembl_id) == 1
        guide[chembl_id[0]] = desc[0]
    return guide


def clean_data(list_csvs):
    failed_d = {}

    for csv in list_csvs:
        print("Now processing dataset {}...".format(os.path.basename(csv)))
        df = pd.read_csv(csv, header=0, sep=";")
        chembl_id = pd.unique(df["Assay ChEMBL ID"])[0]

        inchis = []
        st_type = df["Standard Type"].to_numpy()
        st_value = df["Standard Value"].to_numpy()
        st_unit = df["Standard Units"].to_numpy()

        failed = []

        for idx, smiles in tqdm(enumerate(df["Smiles"].to_list()), total=len(df)):
            try:
                mol = MolFromSmiles(smiles)
                inchis.append(MolToInchi(mol))
            except:
                failed.append(idx)

        print("Failed to process {}/{} molecules".format(len(failed), len(df)))
        success = np.setdiff1d(np.arange(len(df)), np.array(failed))

        st_type, st_value, st_unit = (
            st_type[success].tolist(),
            st_value[success].tolist(),
            st_unit[success].tolist(),
        )

        df_new = pd.DataFrame(
            {
                "inchi": inchis,
                "st_type": st_type,
                "st_value": st_value,
                "st_unit": st_unit,
            }
        )

        df_new.to_csv(
            os.path.join(PROCESSED_DATA_PATH, "{}.csv".format(chembl_id)), index=None
        )
        failed_d[chembl_id] = failed

    with open(os.path.join(DATA_PATH, "failed.pt"), "wb") as handle:
        pickle.dump(failed_d, handle)


def check_overlap(list_csvs):
    n_csvs = len(list_csvs)
    overlap = np.zeros((n_csvs, n_csvs), dtype=np.uint8)

    for i in range(n_csvs):
        df_i = pd.read_csv(list_csvs[i])
        inchis_i = df_i['inchi'].to_numpy()
        for j in range(i + 1, n_csvs):
            df_j = pd.read_csv(list_csvs[j])
            inchis_j = df_j['inchi'].to_numpy()
            overlap[i, j] = len(np.intersect1d(inchis_i, inchis_j))
    overlap += overlap.T.copy()
    return overlap


if __name__ == "__main__":
    matplotlib.rcParams.update({'font.size': 8})

    # Generate guide description of datasets
    list_csvs = glob(os.path.join(RAW_DATA_PATH, '*.csv'))
    guide = gen_guide(list_csvs)

    with open(os.path.join(DATA_PATH, 'guide.pt'), 'wb') as handle:
        pickle.dump(guide, handle)

    # Convert to InChi for easy comparison & check overlap
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    clean_data(list_csvs)
    clean_csvs = glob(os.path.join(PROCESSED_DATA_PATH, '*.csv'))
    chembl_ids = [os.path.basename(p).split('.')[0] for p in clean_csvs]
    overlap = check_overlap(clean_csvs)
    overlap_df = pd.DataFrame(data=overlap, index=chembl_ids, columns=chembl_ids)

    mask = np.zeros_like(overlap)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(overlap_df, annot=True, square=True, mask=mask, cbar=False, fmt='d')
    plt.show()

import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.inchi import MolToInchi
from tqdm import tqdm

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

        missing_values = np.where(np.isnan(st_value))[0]
        non_missing_idx = np.setdiff1d(np.arange(len(df)), missing_values)
        df = df.iloc[non_missing_idx]

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

        if len(df_new) > 0:
            df_new.to_csv(
                os.path.join(PROCESSED_DATA_PATH, "{}.csv".format(chembl_id)),
                index=None,
            )
        failed_d[chembl_id] = np.unique(failed).tolist()

    with open(os.path.join(DATA_PATH, "failed.pt"), "wb") as handle:
        pickle.dump(failed_d, handle)


if __name__ == "__main__":
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # Generate guide description of datasets
    list_csvs = glob(os.path.join(RAW_DATA_PATH, "*.csv"))
    guide = gen_guide(list_csvs)

    with open(os.path.join(DATA_PATH, "guide.pt"), "wb") as handle:
        pickle.dump(guide, handle)

    # Convert to InChi for easy comparison
    clean_data(list_csvs)

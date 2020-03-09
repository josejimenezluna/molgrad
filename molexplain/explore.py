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

from molexplain.utils import DATA_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH, FIGURES_PATH

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
                os.path.join(PROCESSED_DATA_PATH, "{}.csv".format(chembl_id)), index=None
            )
        failed_d[chembl_id] = np.unique(failed).tolist()

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


def distribution_endpoints(list_csvs):
    ligand_d = {}
    
    for csv in list_csvs:
        df = pd.read_csv(csv)
        chembl_id = os.path.basename(csv).split('.')[0]
        print('Checking distribution for endpoint... {}'.format(chembl_id))

        for inchi in tqdm(df['inchi'], total=len(df)):
            ligand_d.setdefault(inchi, []).append(chembl_id)

    return ligand_d


def type_unit_value_exploration(list_csvs):
    type_d = {}
    value_d = {}

    for csv in list_csvs:
        df = pd.read_csv(csv)
        types = df['st_type'].to_list()
        units = df['st_unit'].to_list()
        chembl_id = os.path.basename(csv).split('.')[0]
        value_d[chembl_id] = df['st_value'].to_numpy()

        for t, u_ in zip(types, units):
            if t in type_d:
                if not isinstance(u_, str):
                    u_ = 'nan'
                if u_ in type_d[t]:
                    type_d[t][u_] += 1
                else:
                    type_d[t][u_] = 1
            else:
                type_d[t] = {}
    return type_d, value_d
        

if __name__ == "__main__":
    matplotlib.rcParams.update({'font.size': 8})
    matplotlib.use('Agg')
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # Generate guide description of datasets
    list_csvs = glob(os.path.join(RAW_DATA_PATH, '*.csv'))
    guide = gen_guide(list_csvs)

    with open(os.path.join(DATA_PATH, 'guide.pt'), 'wb') as handle:
        pickle.dump(guide, handle)

    # Convert to InChi for easy comparison & check overlap
    clean_data(list_csvs)
    clean_csvs = glob(os.path.join(PROCESSED_DATA_PATH, '*.csv'))
    chembl_ids = [os.path.basename(p).split('.')[0] for p in clean_csvs]
    overlap = check_overlap(clean_csvs)
    overlap_df = pd.DataFrame(data=overlap, index=chembl_ids, columns=chembl_ids)

    mask = np.zeros_like(overlap)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(9, 10))
    sns.heatmap(overlap_df, annot=True, square=True, mask=mask, cbar=False, fmt='d', ax=ax)
    plt.savefig(os.path.join(FIGURES_PATH, 'overlap.pdf'))
    plt.tight_layout()
    plt.close()

    # Check distribution of endpoints
    ligand_d = distribution_endpoints(clean_csvs)
    dist = [len(v) for v in ligand_d.values()]
    plt.hist(dist, bins=len(np.unique(dist)))
    plt.title('Distribution of endpoints per ligand')
    plt.savefig(os.path.join(FIGURES_PATH, 'endpoints_per_ligand.pdf'))
    plt.close()

    # Type & unit exploration
    type_d, value_d = type_unit_value_exploration(clean_csvs)

    f, ax = plt.subplots(nrows=5, ncols=3, figsize=(10, 16))
    row = 0
    col = 0

    for chembl_id, vals in value_d.items():
        ax[row, col].hist(vals)
        ax[row, col].set_title(chembl_id)
        col += 1
        if col > 2:
            row += 1
            col = 0
    # plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'endpoint_distribution.pdf'))
    plt.close()

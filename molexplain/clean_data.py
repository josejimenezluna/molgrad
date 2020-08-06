import os
import pickle
import requests
from glob import glob

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.inchi import MolToInchi, MolFromInchi
from tqdm import tqdm

from molexplain.utils import DATA_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH

RDLogger.DisableLog("rdApp.*")
REST_404 = "<h1>Page not found (404)</h1>\n"


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


# hERG public data
def process_herg(list_csvs):
    df = pd.read_csv(list_csvs[0], sep="\t")

    for idx, csv in enumerate(list_csvs):
        if idx > 0:
            df_next = pd.read_csv(csv, sep="\t")
            df = pd.concat([df, df_next])

    # filter only IC50, nM, = data.
    df = df.loc[
        (df.Value_type == "IC50") & (df.Unit == "nM") & (df.Relation == "="),
        ["Canonical_smiles", "Value"],
    ]

    df.Value = -np.log10(df.Value * 1e-9)  # pic50 conversion
    df.drop_duplicates(inplace=True)

    # average values with several measurements
    uq_smiles = pd.unique(df["Canonical_smiles"]).tolist()
    uq_values = []

    print("Averaging values with equal smiles...")
    for smi in tqdm(uq_smiles):
        df_uq = df.loc[df["Canonical_smiles"] == smi]
        uq_values.append(df_uq["Value"].mean())

    # drop faulty molecules
    print("Dropping faulty molecules...")
    inchis = []
    values = []

    for smi, val in tqdm(zip(uq_smiles, uq_values), total=len(uq_smiles)):
        try:
            mol = MolFromSmiles(smi)
            inchis.append(MolToInchi(mol))
            values.append(val)
        except:
            continue

    with open(os.path.join(DATA_PATH, "herg", "data_herg.pt"), "wb") as handle:
        pickle.dump([inchis, values], handle)


def process_caco2():
    # peerJ data
    df1 = pd.read_excel(os.path.join(DATA_PATH, "caco2", "peerj-03-1405-s001.xls"))
    df1 = df1.loc[:, ["InChI", "Caco-2 Papp * 10^6 cm/s"]]
    df1.dropna(inplace=True)
    df1["Value"] = -np.log10(df1["Caco-2 Papp * 10^6 cm/s"] * 1e-6)

    # plos one data
    df2 = pd.read_csv(os.path.join(DATA_PATH, "caco2", "caco2perm_pone.csv"))
    df2["Value"] = -np.log10(df2["Papp (Caco-2) [cm/s]"])
    df2 = df2.loc[:, ["name", "Value"]]
    df2.dropna(inplace=True)

    iupac_rest = "http://cactus.nci.nih.gov/chemical/structure/{}/inchi"

    print("Querying InchI strings from IUPAC names...")
    inchis = []
    values = []

    for mol_name, val in tqdm(zip(df2["name"], df2["Value"]), total=len(df2)):
        ans = requests.get(iupac_rest.format(mol_name))
        if ans.status_code == 200:
            inchi = ans.content.decode("utf8")
            inchis.append(inchi)
            values.append(val)

    inchis.extend(df1["InChI"].tolist())
    values.extend(df1["Value"].tolist())

    df = pd.DataFrame({"inchi": inchis, "values": values})
    uq_inchi = pd.unique(df["inchi"]).tolist()

    print("Averaging values and ensuring rdkit readability...")
    inchis = []
    values = []

    # Average values and make sure rdkit can read all inchis
    for inchi in tqdm(uq_inchi):
        mol = MolFromInchi(inchi)
        if mol is not None:
            df_uq = df.loc[df["inchi"] == inchi]
            inchis.append(inchi)
            values.append(df_uq["values"].mean())

    with open(os.path.join(DATA_PATH, "caco2", "data_caco2.pt"), "wb") as handle:
        pickle.dump([inchis, values], handle)


if __name__ == "__main__":
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # Generate guide description of datasets
    list_csvs = glob(os.path.join(RAW_DATA_PATH, "*.csv"))
    guide = gen_guide(list_csvs)

    with open(os.path.join(DATA_PATH, "guide.pt"), "wb") as handle:
        pickle.dump(guide, handle)

    # Convert to InChi for easy comparison
    clean_data(list_csvs)

    # hERG public data
    herg_list = glob(os.path.join(DATA_PATH, "herg", "part*.tsv"))
    process_herg(herg_list)

    # process caco2

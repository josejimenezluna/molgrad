import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
import requests
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.inchi import MolFromInchi, MolToInchi
from tqdm import tqdm

from molexplain.utils import DATA_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH

RDLogger.DisableLog("rdApp.*")
IUPAC_REST = "http://cactus.nci.nih.gov/chemical/structure/{}/inchi"


def smi_to_inchi_with_val(smiles, ovalues):
    inchis = []
    values = []

    for smi, val in zip(smiles, ovalues):
        mol = MolFromSmiles(smi)
        if mol is not None:
            try:
                inchi = MolToInchi(mol)
                m = MolFromInchi(inchi)
                if m is not None:  # ensure rdkit can read an inchi it just wrote...
                    inchis.append(inchi)
                    values.append(val)
            except:
                continue
    return inchis, values


def mean_by_key(df, key_col, val_col):
    uq_keys = pd.unique(df[key_col]).tolist()
    uq_values = []

    for key in uq_keys:
        df_uq = df.loc[df[key_col] == key]
        uq_values.append(df_uq[val_col].mean())
    return uq_keys, uq_values


def ensure_readability(ostrings, ovalues, read_mol_f):
    strings = []
    values = []

    for s, v in zip(ostrings, ovalues):
        mol = read_mol_f(s)
        if mol is not None:
            strings.append(s)
            values.append(v)
    return strings, values


# hERG public data
def process_herg(list_csvs, keep_operators=True):
    df = pd.read_csv(list_csvs[0], sep="\t")

    for idx, csv in enumerate(list_csvs):
        if idx > 0:
            df_next = pd.read_csv(csv, sep="\t")
            df = pd.concat([df, df_next])

    # filter only IC50, nM, = data.
    condition = (df.Value_type == "IC50") & (df.Unit == "nM")
    if not keep_operators:
         condition = condition & (df.Relation == "=")

    df = df.loc[
        condition,
        ["Canonical_smiles", "Value"],
    ]

    df.Value = -np.log10(df.Value * 1e-9)  # pIC50 conversion
    df.drop_duplicates(inplace=True)

    # average values with several measurements
    uq_smiles, uq_values = mean_by_key(df, "Canonical_smiles", "Value")

    # drop faulty molecules
    print("Dropping faulty molecules...")
    inchis, values = smi_to_inchi_with_val(uq_smiles, uq_values)

    with open(os.path.join(DATA_PATH, "herg", "data_herg.pt"), "wb") as handle:
        pickle.dump([inchis, values], handle)


def process_ppb():
    inchis = []
    values = []

    # first dataset
    xlsxs = glob(os.path.join(DATA_PATH, "ppb", "11095_2013_1023_MOESM[2-4]_ESM.xlsx"))
    for idx, xlsx in enumerate(xlsxs):
        ppb_col = "Experimental_%PPB" if idx < 3 else "Experimental PPB_[%]"
        df1 = pd.read_excel(xlsx)
        df1 = df1.loc[:, ["SMILES", ppb_col]]
        inchis_1, values_1 = smi_to_inchi_with_val(df1["SMILES"], df1[ppb_col])
        inchis.extend(inchis_1)
        values.extend(values_1)

    # second dataset
    df2 = pd.read_excel(os.path.join(DATA_PATH, "ppb", "ci6b00291_si_001.xlsx"))
    df2 = df2.loc[:, ["SMILES", "Fub"]]
    df2["Value"] = 100 * (1 - df2["Fub"])
    inchis_2, values_2 = smi_to_inchi_with_val(df2["SMILES"], df2["Value"])
    inchis.extend(inchis_2)
    values.extend(values_2)

    # third dataset
    df3 = pd.read_excel(
        os.path.join(DATA_PATH, "ppb", "cmdc201700582-sup-0001-misc_information.xlsx"),
        sheet_name=4,
    )
    df3 = df3.loc[:, ["SMILES", "PPB_Traditional_assay（serve as the true value）"]]
    df3["Value"] = 100 * df3["PPB_Traditional_assay（serve as the true value）"]
    inchis_3, values_3 = smi_to_inchi_with_val(df3["SMILES"], df3["Value"])
    inchis.extend(inchis_3)
    values.extend(values_3)

    # fourth dataset
    df4 = pd.read_excel(
        os.path.join(DATA_PATH, "ppb", "jm051245vsi20061025_033631.xls")
    )
    df4 = df4.loc[:, ["NAME (Drug or chemical  name)", "PBexp(%)"]]

    for mol_name, val in tqdm(
        zip(df4["NAME (Drug or chemical  name)"], df4["PBexp(%)"]), total=len(df4)
    ):
        ans = requests.get(IUPAC_REST.format(mol_name))
        if ans.status_code == 200:
            inchi = ans.content.decode("utf8")
            mol = MolFromInchi(inchi)
            if mol is not None:
                inchis.append(inchi)
                values.append(val)

    # fifth dataset
    df5 = pd.read_excel(os.path.join(DATA_PATH, "ppb", "mp8b00785_si_002.xlsx"))
    df5 = df5.loc[:, ["canonical_smiles", "fup"]]
    df5["Value"] = 100 * (1 - df5["fup"])
    inchis_5, values_5 = smi_to_inchi_with_val(df5["canonical_smiles"], df5["Value"])
    inchis.extend(inchis_5)
    values.extend(values_5)

    # sixth dataset
    df6 = pd.read_html(os.path.join(DATA_PATH, "ppb", "kratochwil2002.html"), header=0)[
        0
    ]
    df6 = df6.loc[:, ["Compound", "fb (%)b"]].dropna()

    for mol_name, val in tqdm(zip(df6["Compound"], df6["fb (%)b"]), total=len(df6)):
        ans = requests.get(IUPAC_REST.format(mol_name))
        if ans.status_code == 200:
            inchi = ans.content.decode("utf8")
            mol = MolFromInchi(inchi)
            if mol is not None:
                inchis.append(inchi)
                values.append(val)

    # join them all together
    df = pd.DataFrame({"inchi": inchis, "values": values})

    # average values w. equal inchi and check readability
    print("Averaging values and ensuring rdkit readability...")
    inchis, values = mean_by_key(df, "inchi", "values")

    inchis, values = ensure_readability(inchis, values, MolFromInchi)

    with open(os.path.join(DATA_PATH, "ppb", "data_ppb.pt"), "wb") as handle:
        pickle.dump([inchis, values], handle)


def process_caco2():
    # peerJ data
    df1 = pd.read_excel(os.path.join(DATA_PATH, "caco2", "peerj-03-1405-s001.xls"))
    df1 = df1.loc[:, ["InChI", "Caco-2 Papp * 10^6 cm/s"]]
    df1.dropna(inplace=True)
    df1["Value"] = np.log10(df1["Caco-2 Papp * 10^6 cm/s"] * 1e-6)

    # plos one data
    df2 = pd.read_csv(os.path.join(DATA_PATH, "caco2", "caco2perm_pone.csv"))
    df2["Value"] = np.log10(df2["Papp (Caco-2) [cm/s]"])
    df2 = df2.loc[:, ["name", "Value"]]
    df2.dropna(inplace=True)

    print("Querying InchI strings from IUPAC names...")
    inchis = []
    values = []

    for mol_name, val in tqdm(zip(df2["name"], df2["Value"]), total=len(df2)):
        ans = requests.get(IUPAC_REST.format(mol_name))
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


def process_cyp():
    df = pd.read_csv(os.path.join(DATA_PATH, "cyp", "CYP3A4.csv"), header=0, sep=";")
    df["Value"] = [1 if class_ == "Active" else 0 for class_ in df["Class"]]
    inchis, values = smi_to_inchi_with_val(df["SMILES"], df["Value"])
    df = pd.DataFrame({"inchi": inchis, "values": values})
    inchis, values = mean_by_key(df, "inchi", "values")

    with open(os.path.join(DATA_PATH, "cyp", "data_cyp.pt"), "wb") as handle:
        pickle.dump([inchis, values], handle)


if __name__ == "__main__":
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # hERG public data
    herg_list = glob(os.path.join(DATA_PATH, "herg", "part*.tsv"))
    process_herg(herg_list, keep_operators=True)

    # caco2 data
    process_caco2()

    # ppb data
    process_ppb()

    # cyp data
    process_cyp()

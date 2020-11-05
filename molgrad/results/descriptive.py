import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.inchi import MolFromInchi
from rdkit.Chem.Lipinski import NumHDonors
from tqdm import tqdm

from molgrad.utils import DATA_PATH, FIG_PATH

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 20,
    }
)


LABEL_GUIDE = {
    "ppb": r"$F_b(\%)$",
    "caco2": r"$P_\mathrm{app}$",
    "herg": r"$p\mathrm{IC}_{50}$",
}


DATASET_GUIDE = {"ppb": "PPB", "caco2": "Caco-2", "herg": "hERG", "cyp": "CYP3A4"}

# Boxplot with descriptive values for all molecules

if __name__ == "__main__":
    mws = []
    logps = []
    nhdonors = []
    values = []
    dataset = []

    for data in list(LABEL_GUIDE.keys()) + ["cyp"]:
        with open(os.path.join(DATA_PATH, data, f"data_{data}.pt"), "rb") as handle:
            inchis, v = pickle.load(handle)

        values.extend(v)

        for inchi in tqdm(inchis):
            mol = MolFromInchi(inchi)
            mws.append(MolWt(mol))
            logps.append(MolLogP(mol))
            nhdonors.append(NumHDonors(mol))
            dataset.append(DATASET_GUIDE[data])

    df = pd.DataFrame(
        {
            "Molecular weight (gr./mol)": mws,
            r"cLog$P$": logps,
            "No. hydrogen donors": nhdonors,
            "values": values,
            "dataset": dataset,
        }
    )

    f, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].grid(alpha=0.5)
    axs[1].grid(alpha=0.5)
    axs[2].grid(alpha=0.5)

    sns.boxplot(
        y="Molecular weight (gr./mol)",
        x="dataset",
        data=df,
        ax=axs[0],
        showfliers=False,
        palette="Set2",
    )
    sns.boxplot(
        y=r"cLog$P$", x="dataset", data=df, ax=axs[1], showfliers=False, palette="Set2"
    )
    sns.boxplot(
        y="No. hydrogen donors",
        x="dataset",
        data=df,
        ax=axs[2],
        showfliers=False,
        palette="Set2",
    )

    axs[0].set_xlabel("")
    axs[1].set_xlabel("")
    axs[2].set_xlabel("")

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_PATH, "descriptive.pdf"))
    plt.close()

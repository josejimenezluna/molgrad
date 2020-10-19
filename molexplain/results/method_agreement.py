import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import torch
from rdkit.Chem.inchi import MolFromInchi
from tqdm import tqdm

from molexplain.net import MPNNPredictor
from molexplain.train import DEVICE, TASK_GUIDE
from molexplain.utils import DATA_PATH, MODELS_PATH, BASELINE_MODELS_PATH
from molexplain.vis import molecule_importance
from molexplain.vis_baseline import molecule_importance_diff


N_VERSIONS = 3


def method_agreement(
    aimportance_i, aimportance_j, bimportance_i=None, bimportance_j=None
):
    """
    Computes agreement between two coloring methods. 
    """
    asign_i, asign_j = np.sign(aimportance_i), np.sign(aimportance_j)
    col_mean = np.mean(asign_i == asign_j)

    if bimportance_i is not None and bimportance_j is not None:
        bsign_i, bsign_j = np.sign(bimportance_i), np.sign(bimportance_j)
        bmean = np.mean(bsign_i == bsign_j)
        col_mean = np.mean([col_mean, bmean])
    return col_mean


def plot_agreement(scores):
    sns.pairplot(scores)
    plt.show()


if __name__ == "__main__":
    importances = {}

    for data in TASK_GUIDE.keys():
        print(f"Now computing importances for dataset {data}...")
        imp = [[] for _ in range(N_VERSIONS)]
        g_imp = [[] for _ in range(N_VERSIONS)]
        imp_rf = []

        model_pt = os.path.join(MODELS_PATH, f"{data}_noHs.pt")
        model_rf = load(os.path.join(BASELINE_MODELS_PATH, f"rf_{data}.pt"))

        with open(
            os.path.join(DATA_PATH, f"{data}", f"data_{data}.pt"), "rb"
        ) as handle:
            inchis, _ = pickle.load(handle)

        output_f = torch.sigmoid if data == "cyp" else None

        model = MPNNPredictor(
            node_in_feats=49,
            edge_in_feats=10,
            global_feats=4,
            n_tasks=1,
            output_f=output_f,
        ).to(DEVICE)

        model.load_state_dict(torch.load(model_pt, map_location=DEVICE))

        for inchi in tqdm(inchis):
            mol = MolFromInchi(inchi)
            for version in range(1, N_VERSIONS + 1):
                _, _, atom_importance, _, global_importance = molecule_importance(
                    mol, model, version=version
                )
                imp[version - 1].append(atom_importance)
                g_imp[version - 1].append(global_importance)

            _, _, i_rf = molecule_importance_diff(mol, model_rf, task=TASK_GUIDE[data])
            imp_rf.append(i_rf)

        importances[f"{data}_graph"] = imp
        importances[f"{data}_global"] = g_imp
        importances[f"{data}_rf"] = imp_rf

    with open(os.path.join(DATA_PATH, "importances.pt"), "wb") as handle:
        pickle.dump(importances, handle)

    # Model comparison
    agreement_d = {}

    for data in TASK_GUIDE.keys():
        agreement_m = np.zeros((N_VERSIONS + 1, N_VERSIONS + 1), dtype=np.float32)
        imp_mpnn = importances[f"{data}_graph"]
        imp_rf = importances[f"{data}_rf"]

        for idx_i in range(N_VERSIONS):
            for idx_j in range(N_VERSIONS):
                if idx_j > idx_i:
                    col_means = []
                    for aimportances_i, aimportances_j in zip(
                        imp_mpnn[idx_i], imp_mpnn[idx_j]
                    ):
                        col_means.append(
                            method_agreement(aimportances_i, aimportances_j)
                        )
                    agreement_m[idx_i, idx_j] = np.mean(col_means)
            col_means = []
            for aimportances_i, aimportances_rf in zip(imp_mpnn[idx_i], imp_rf):
                col_means.append(method_agreement(aimportances_i, aimportances_rf))

            agreement_m[idx_i, N_VERSIONS] = np.mean(col_means)
        agreement_m += agreement_m.T.copy()
        agreement_m += np.eye(N_VERSIONS + 1)
        agreement_d[f"{data}"] = agreement_m

    with open(os.path.join(DATA_PATH, "agreement.pt"), "wb") as handle:
        pickle.dump(agreement_d, handle)

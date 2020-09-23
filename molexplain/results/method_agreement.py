import os
import pickle

import numpy as np
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


if __name__ == "__main__":
    importances = {}

    for data in TASK_GUIDE.keys():
        print(f"Now computing importances for dataset {data}...")
        imp = [[] for _ in range(N_VERSIONS)]
        imp_rf = []

        model_pt = os.path.join(MODELS_PATH, f"{data}_noHs.pt")
        model_rf = load(os.path.join(BASELINE_MODELS_PATH, f"rf_{data}.pt"))

        with open(
            os.path.join(DATA_PATH, f"{data}", f"data_{data}.pt"), "rb"
        ) as handle:
            inchis, _ = pickle.load(handle)

        model = MPNNPredictor(
            node_in_feats=49, edge_in_feats=10, global_feats=4, n_tasks=1
        ).to(DEVICE)

        model.load_state_dict(torch.load(model_pt, map_location=DEVICE))

        for inchi in tqdm(inchis):
            mol = MolFromInchi(inchi)
            for version in range(N_VERSIONS):
                _, _, atom_importance, _, _ = molecule_importance(
                    mol, model, version=version
                )
                imp[version].append(atom_importance)

            _, _, i_rf = molecule_importance_diff(mol, model_rf)
            imp_rf.append(i_rf)

        # importances[f"{data}_mpnn"] = imp
        importances[f"{data}_rf"] = imp_rf

    with open(os.path.join(DATA_PATH, f"importances_{data}"), "wb") as handle:
        pickle.dump(importances, handle)

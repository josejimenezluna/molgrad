import os
import pickle

import numpy as np
from joblib import Parallel, delayed
from rdkit.Chem.AllChem import GetMorganFingerprint
from rdkit.Chem.inchi import MolFromInchi
from rdkit.DataStructs import TanimotoSimilarity

from molexplain.utils import DATA_PATH, MODELS_PATH
from molexplain.prod import predict


def tanimoto_sim(mol_i, mol_j, radius=2):
    fp_i, fp_j = (
        GetMorganFingerprint(mol_i, radius),
        GetMorganFingerprint(mol_j, radius),
    )
    return TanimotoSimilarity(fp_i, fp_j)


def parallel_wrapper(mol, rest_inchis, n_total):
    sims = np.zeros(n_total, dtype=np.float32)
    n_rest = len(rest_inchis)
    fill_idx = n_total - n_rest

    for inchi in rest_inchis:
        mol_j = MolFromInchi(inchi)
        sims[fill_idx] = tanimoto_sim(mol, mol_j)
        fill_idx += 1
    return sims


def sim_matrix(inchis):
    n_total = len(inchis)
    sims = Parallel(n_jobs=-1, verbose=100, backend="multiprocessing")(
        delayed(parallel_wrapper)(MolFromInchi(inchi), inchis[(idx + 1) :], n_total)
        for idx, inchi in enumerate(inchis)
    )
    sims = np.stack(sims)
    sims += sims.copy().T
    sims += np.eye(n_total)
    return sims


def diff_matrix(values):
    n_total = len(values)
    diff = np.zeros((n_total, n_total), dtype=np.float32)
    for idx_i, val_i in enumerate(values):
        for idx_j, val_j in enumerate(values):
            if idx_i < idx_j:
                diff[idx_i, idx_j] = val_i - val_j
    diff += diff.copy().T
    return diff


if __name__ == "__main__":
    with open(os.path.join(DATA_PATH, "ppb", "data_ppb.pt"), "rb") as handle:
        inchis, values = pickle.load(handle)

    # Chemical similarity between ligands
    sims = sim_matrix(inchis_herg)

    with open(os.path.join(DATA_PATH, "herg", "sim_herg.pt"), "wb") as handle:
        pickle.dump(sims, handle)

    # Experimental difference
    diff_exp = diff_matrix(values)

    # Predicted difference
    w_path = os.path.join(MODELS_PATH, "ppb_noHs.pt")
    preds = predict(inchis, w_path)

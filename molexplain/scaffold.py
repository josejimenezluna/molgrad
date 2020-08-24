import os
import pickle

import numpy as np
import torch
from joblib import Parallel, delayed
from rdkit.Chem.AllChem import GetMorganFingerprint
from rdkit.Chem.inchi import MolFromInchi
from rdkit.DataStructs import TanimotoSimilarity

from molexplain.utils import DATA_PATH, MODELS_PATH
from molexplain.prod import predict


def tanimoto_sim(mol_i, mol_j, radius=2):
    """Computes Tanimoto similarity between two mols

    Parameters
    ----------
    mol_i : rdkit.mol
    mol_j : rdkit.mol
    radius : int, optional
        [description], by default 2

    Returns
    -------
    float
    """
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
    """Computes pairwise similarity matrix between all compounds in the `inchis` list.

    Parameters
    ----------
    inchis : list
        A list of inchi strings
    Returns
    -------
    np.ndarray
    """
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
    """Computes pairwise difference of values in 

    Parameters
    ----------
    values : list
        list of numeric values

    Returns
    -------
    np.ndarray
        Array with pairwise differences
    """
    n_total = len(values)
    diff = np.zeros((n_total, n_total), dtype=np.float32)
    for idx_i, val_i in enumerate(values):
        for idx_j, val_j in enumerate(values):
            if idx_i < idx_j:
                diff[idx_i, idx_j] = val_i - val_j
    diff += diff.copy().T
    return diff


if __name__ == "__main__":
    with open(os.path.join(DATA_PATH, "cyp", "data_cyp.pt"), "rb") as handle:
        inchis, values = pickle.load(handle)

    # Chemical similarity between ligands
    sims = sim_matrix(inchis)

    with open(os.path.join(DATA_PATH, "cyp", "sim_cyp.pt"), "wb") as handle:
        pickle.dump(sims, handle)

    # Experimental difference
    diff_exp = diff_matrix(values)
    np.save(os.path.join(DATA_PATH, "cyp", "diff_exp.npy"), arr=diff_exp)

    # Predicted difference
    w_path = os.path.join(MODELS_PATH, "cyp_noHs.pt")
    preds = predict(inchis, w_path, output_f=torch.sigmoid).squeeze()

    print('Test R: {:.3f}'.format(np.corrcoef(values, preds)[0, 1]))

    diff_hat = diff_matrix(preds)
    np.save(os.path.join(DATA_PATH, "cyp", "diff_hat.npy"), arr=diff_hat)

    #

    w_path = os.path.join(MODELS_PATH, "cyp_noHs_notest.pt")
    preds_notest = predict(inchis, w_path, output_f=torch.sigmoid).squeeze()

    print('Training R: {:.3f}'.format(np.corrcoef(values, preds_notest)[0, 1]))

    diff_hat_notest = diff_matrix(preds_notest)
    np.save(os.path.join(DATA_PATH, "cyp", "diff_hat_notest.npy"), arr=diff_hat_notest)

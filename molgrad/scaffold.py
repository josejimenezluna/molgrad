import os
import pickle

import numpy as np
import torch
from joblib import Parallel, delayed
from rdkit.Chem.AllChem import GetMorganFingerprint
from rdkit.Chem.inchi import MolFromInchi
from rdkit.DataStructs import TanimotoSimilarity

from molgrad.train import TASK_GUIDE
from molgrad.utils import DATA_PATH, MODELS_PATH
from molgrad.prod import predict


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


if __name__ == "__main__":
    for data in TASK_GUIDE.keys():
        print(f"Now computing scaffold similarity and difference matrices for endpoint {data}")
        with open(os.path.join(DATA_PATH, f"{data}", f"data_{data}.pt"), "rb") as handle:
            inchis, values = pickle.load(handle)

        # Only CYP requires sigmoid act. function 
        if data == "cyp":
            output_f = torch.sigmoid
        else:
            output_f = None

        # Chemical similarity between ligands
        sims = sim_matrix(inchis)

        with open(os.path.join(DATA_PATH, f"{data}", f"sim_{data}.pt"), "wb") as handle:
            pickle.dump(sims, handle)

        # Experimental difference
        diff_exp = np.subtract.outer(values, values)
        np.save(os.path.join(DATA_PATH, f"{data}", "diff_exp.npy"), arr=diff_exp)

        # Predicted difference
        w_path = os.path.join(MODELS_PATH, f"{data}_noHs.pt")
        preds = predict(inchis, w_path, output_f=output_f).squeeze().cpu().numpy()
        np.save(os.path.join(DATA_PATH, f"{data}", "preds.npy"), arr=preds)

        diff_hat = np.subtract.outer(preds, preds)
        np.save(os.path.join(DATA_PATH, f"{data}", "diff_hat.npy"), arr=diff_hat)

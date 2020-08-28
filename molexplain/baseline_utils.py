import os
from copy import deepcopy

import numpy as np
from rdkit.Chem import AllChem, DataStructs


def gen_dummy_atoms(mol, dummy_atom_no=47):
    """For every atom in Molecule `mol`, generates a copy
    with a dummy_atom_no substitution.

    Parameters
    ----------
    mol : rdkit.mol
    dummy_atom_no : int, optional
        Atomic number of the mol to substitute, by default 11

    Returns
    -------
    [list]
        A list of modified mols.
    """
    mod_mols = []
    for idx_atom in range(mol.GetNumAtoms()):
        mol_cpy = deepcopy(mol)
        mol_cpy.GetAtomWithIdx(idx_atom).SetAtomicNum(dummy_atom_no)
        mod_mols.append(mol_cpy)
    return mod_mols


def featurize_ecfp4(mol, fp_size=1024, bond_radius=2):
    """Generates ECFP4 fingerprint for `mol`

    Parameters
    ----------
    mol : rdkit.mol
    fp_size : int, optional
        fingerprint length, by default 1024
    bond_radius : int, optional
        radius of the bond, by default 2

    Returns
    -------
    np.ndarray
        ECFP4 fingerprint
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=fp_size)
    arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def diff_importance(mol, model, fp_size=1024, bond_radius=2, dummy_atom_no=47):
    """Returns atom importance based on dummy substitutions

    Parameters
    ----------
    mol : rdkit.mol
    model : None
        An instance of a model that has a `predict` method
    fp_size : int, optional
    bond_radius : int, optional
    dummy_atom_no : int, optional

    Returns
    -------
    np.ndarray
        Atom importances
    """
    og_fp = featurize_ecfp4(mol, fp_size, bond_radius)
    og_pred = model.predict(og_fp[np.newaxis, :])

    mod_mols = gen_dummy_atoms(mol, dummy_atom_no)

    mod_fps = [featurize_ecfp4(mol) for mol in mod_mols]
    mod_fps = np.vstack(mod_fps)
    mod_preds = model.predict(mod_fps)
    return og_pred - mod_preds


if __name__ == "__main__":
    import pickle
    from molexplain.utils import DATA_PATH
    from rdkit.Chem.inchi import MolFromInchi
    from tqdm import tqdm

    with open(os.path.join(DATA_PATH, "ppb", "data_ppb.pt"), "rb") as handle:
        inchis, values = pickle.load(handle)

    # featurize all mols
    feats = [featurize_ecfp4(MolFromInchi(inchi)) for inchi in tqdm(inchis)]
    feats = np.vstack(feats)

    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=1000, n_jobs=-2)
    rf.fit(feats, values)

    preds = rf.predict(feats)

    # trial
    diffs = [
        diff_importance(MolFromInchi(inchi), rf, dummy_atom_no=47)
        for inchi in tqdm(inchis)
    ]
    avg_diffs = [np.mean(diff) for diff in diffs]
    sum_diffs = [np.sum(diff) for diff in diffs]
    max_diffs = [np.max(np.abs(diff)) for diff in diffs]

    print(np.corrcoef(values, avg_diffs)[0, 1])
    print(np.corrcoef(values, sum_diffs)[0, 1])
    print(np.corrcoef(values, max_diffs)[0, 1])

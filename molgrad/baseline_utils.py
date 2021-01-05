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
        Atomic number of the mol to substitute, by default 47

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


def diff_importance(
    mol, model, task="regression", fp_size=1024, bond_radius=2, dummy_atom_no=47
):
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

    if task == "regression":
        pred_fun = lambda x: model.predict(x)
    elif task == "binary":
        pred_fun = lambda x: model.predict_proba(x)[:, 1]

    og_pred = pred_fun(og_fp[np.newaxis, :])

    mod_mols = gen_dummy_atoms(mol, dummy_atom_no)

    mod_fps = [featurize_ecfp4(mol) for mol in mod_mols]
    mod_fps = np.vstack(mod_fps)
    mod_preds = pred_fun(mod_fps)
    return og_pred - mod_preds


def pred_baseline(mol, model, task="regression", fp_size=1024, bond_radius=2):
    og_fp = featurize_ecfp4(mol, fp_size, bond_radius)

    if task == "regression":
        pred_fun = lambda x: model.predict(x)
    elif task == "binary":
        pred_fun = lambda x: model.predict_proba(x)[:, 1]
    
    return pred_fun(og_fp[np.newaxis, :])

import numpy as np
from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdmolops import AddHs

from molgrad.baseline_utils import diff_importance
from molgrad.vis import determine_atom_col, determine_bond_col


def molecule_importance_diff(
    mol,
    model,
    task="regression",
    fp_size=1024,
    bond_radius=2,
    dummy_atom_no=47,
    eps=1e-3,
    vis_factor=0.1,
    normalize=False,
    img_width=400,
    img_height=200,
    addHs=False,
):
    """Colors molecule based on prediction difference when substituting
    individual atoms by dummies.

    Parameters
    ----------
    mol : rdkit mol
    model : 
        An instance of a model that implements a predict method
    fp_size : int, optional
        ECFP4 fingerprint size, by default 1024
    bond_radius : int, optional
        ECFP4 bond radius, by default 2
    dummy_atom_no : int, optional
        Atomic number of the dummy atom to use for substitution, by default 47 (Ag)
    vis_factor : float, optional
        Visualization factor, by default 1.0
    normalize : bool, optional
        Whether to normalize the computed atom importances, by default False
    img_width, img_height: int, optional
        Size of the generated SVG in px, by default 400, 200
    addHs : bool, optional
        Whether to use explicit hydrogens in the calculation, by default False

    Returns
    -------
    svg : str
        String of the generated SVG
    SVG : img
        Image of the generated SVG.
    atom_importance: np.ndarray
        Computed atomic importances
    """
    if addHs:
        mol = AddHs(mol)

    atom_importance = diff_importance(
        mol, model, task, fp_size, bond_radius, dummy_atom_no
    )

    if normalize:
        mean, std = np.mean(atom_importance), np.std(atom_importance)
        atom_importance -= mean
        atom_importance /= std

    highlightAtomColors = determine_atom_col(mol, atom_importance, eps=eps)
    highlightAtoms = list(highlightAtomColors.keys())

    highlightBondColors = determine_bond_col(highlightAtomColors, mol)
    highlightBonds = list(highlightBondColors.keys())

    highlightAtomRadii = {
        k: np.abs(v) * vis_factor for k, v in enumerate(atom_importance)
    }
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(img_width, img_height)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlightAtoms,
        highlightAtomColors=highlightAtomColors,
        highlightAtomRadii=highlightAtomRadii,
        highlightBonds=highlightBonds,
        highlightBondColors=highlightBondColors,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")
    return svg, SVG(svg), atom_importance

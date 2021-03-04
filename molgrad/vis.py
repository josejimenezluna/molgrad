import numpy as np
from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdmolops import AddHs

from molgrad.ig import integrated_gradients
from molgrad.net_utils import get_global_features, mol_to_dgl

rdDepictor.SetPreferCoordGen(True)

GREEN_COL = (0, 1, 0)
RED_COL = (1, 0, 0)


def determine_atom_col(mol, atom_importance, eps=1e-5):
    """ Colors atoms with positive and negative contributions
    as green and red respectively, using an `eps` absolute
    threshold.

    Parameters
    ----------
    mol : rdkit mol
    atom_importance : np.ndarray
        importances given to each atom
    bond_importance : np.ndarray
        importances given to each bond
    version : int, optional
        1. does not consider bond importance
        2. bond importance is taken into account, but fixed
        3. bond importance is treated the same as atom importance, by default 2
    eps : float, optional
        threshold value for visualization - absolute importances below `eps`
        will not be colored, by default 1e-5

    Returns
    -------
    dict
        atom indexes with their assigned color
    """
    atom_col = {}

    for idx, v in enumerate(atom_importance):
        if v > eps:
            atom_col[idx] = GREEN_COL
        if v < -eps:
            atom_col[idx] = RED_COL

    return atom_col


def determine_bond_col(atom_col, mol):
    """Colors bonds depending on whether the atoms involved
    share the same color.

    Parameters
    ----------
    atom_col : np.ndarray
        coloring assigned to each atom index
    mol : rdkit mol

    Returns
    -------
    dict
        bond indexes with assigned color
    """
    bond_col = {}

    for idx, bond in enumerate(mol.GetBonds()):
        atom_i_idx, atom_j_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if atom_i_idx in atom_col and atom_j_idx in atom_col:
            if atom_col[atom_i_idx] == atom_col[atom_j_idx]:
                bond_col[idx] = atom_col[atom_i_idx]
    return bond_col


def molecule_importance(
    mol,
    model,
    task=0,
    n_steps=50,
    version=2,
    eps=1e-4,
    vis_factor=1.0,
    feature_scale=True,
    img_width=400,
    img_height=200,
    addHs=False,
):
    """Colors molecule according to the integrated gradients method for
    a particular `task`, using a Monte Carlo approximation with `n_steps`.
    Uses a `vis_factor` multiplicative parameter for clearer visualization
    purposes.

    Parameters
    ----------
    mol : rdkit mol
    model : MPNNPredictor instance
        A trained instance of a message passing network model
    task : int, optional
        Task for which to compute atom importances, by default 0
    n_steps : int, optional
        Number of steps in the Monte Carlo approx, by default 50
    version : int, optional
        Version of the algorithm to use (check determine_atom_col
        function), by default 2
    eps : float, optional
        threshold value for visualization - absolute importances below `eps`
        will not be colored, by default 1e-5, by default 1e-4
    vis_factor : float, optional
        value that is multiplied to the atom importances for visualization
        purposes, by default 1.0
    feature_scale: bool, optional
        whether to scale the resulting gradients by the original features
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
    bond_importance: np.ndarray
        Computed bond importances
    global_importance: np.ndarray
        Computed global importances
    """
    if addHs:
        mol = AddHs(mol)
    graph = mol_to_dgl(mol)
    g_feat = get_global_features(mol)
    atom_importance, bond_importance, global_importance = integrated_gradients(
        graph,
        g_feat,
        model,
        task=task,
        n_steps=n_steps,
        version=version,
        feature_scale=feature_scale,
    )

    # bond importances gets distributed across atoms if version > 1
    if version > 1:
        bond_idx = []

        for bond in mol.GetBonds():
            bond_idx.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

        for (atom_i_idx, atom_j_idx), b_imp in zip(bond_idx, bond_importance):
            atom_importance[atom_i_idx] += b_imp / 2
            atom_importance[atom_j_idx] += b_imp / 2

    highlightAtomColors = determine_atom_col(mol, atom_importance, eps=eps)
    highlightAtoms = list(highlightAtomColors.keys())

    highlightBondColors = determine_bond_col(highlightAtomColors, mol)
    highlightBonds = list(highlightBondColors.keys())

    highlightAtomRadii = {
        k: np.abs(v) * vis_factor for k, v in enumerate(atom_importance)
    }

    rdDepictor.Compute2DCoords(mol, canonOrient=True)
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
    return svg, SVG(svg), atom_importance, bond_importance, global_importance

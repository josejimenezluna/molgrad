import numpy as np
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import MolFromInchi, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from molexplain.ig import integrated_gradients
from molexplain.net_utils import mol_to_dgl, get_global_features

GREEN_COL = (0, 1, 0)
RED_COL = (1, 0, 0)


def determine_atom_col(atom_importance, eps=1e-5):
    """
    Colors atoms with positive and negative contributions
    as green and red respectively, using an `eps` absolute
    threshold.
    """
    atom_col = {}

    for idx, v in enumerate(atom_importance):
        if v > eps:
            atom_col[idx] = GREEN_COL
        if v < -eps:
            atom_col[idx] = RED_COL
    return atom_col


def determine_bond_col(atom_col, mol):
    """
    Colors bonds depending on whether the atoms involved
    share the same color.
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
    eps=1e-5,
    vis_factor=10,
    img_width=400,
    img_height=200,
):
    """
    Colors molecule according to the integrated gradients method for
    a particular `task`, using a Monte Carlo approximation with `n_steps`.
    Uses a `vis_factor` multiplicative parameter for clearer visualization
    purposes.
    """
    graph = mol_to_dgl(mol)
    g_feat = get_global_features(mol)
    atom_importance, global_importance = integrated_gradients(graph, g_feat,model, task=task, n_steps=n_steps)

    highlightAtomColors = determine_atom_col(atom_importance, eps=eps)
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
    return SVG(svg), global_importance

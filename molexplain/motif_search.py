import os
import numpy as np
from rdkit.Chem import MolFromSmarts

FATTY_ACID_PATT = MolFromSmarts("C-C-C-C(-[OH])=O")

# PPB:
# fatty acids


def is_fatty_acid(mol):
    if mol.HasSubstructMatch(FATTY_ACID_PATT):
        atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
        uq, counts = np.unique(atom_types, return_counts=True)
        if ["C", "O"] == uq.tolist() and counts[1] == 2:
            return True
        return False


if __name__ == "__main__":
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.inchi import MolFromInchi

    import pickle
    from molexplain.utils import DATA_PATH

    with open(os.path.join(DATA_PATH, "ppb", "data_ppb.pt"), "rb") as handle:
        inchis, values = pickle.load(handle)

    from tqdm import tqdm

    fatty_acids = []
    idxs = []
    for idx, inchi in enumerate(tqdm(inchis)):
        if is_fatty_acid(MolFromInchi(inchi)):
            fatty_acids.append(inchi)
            idxs.append(idx)

    vals = [values[idx] for idx in idxs]

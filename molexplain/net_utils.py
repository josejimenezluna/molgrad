import numpy as np
import rdkit
import torch
from rdkit.Chem.inchi import MolFromInchi
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import GetPeriodicTable
from torch.utils.data import Dataset

import dgl

ATOM_TYPES = [
    "Ag",
    "As",
    "B",
    "Br",
    "C",
    "Ca",
    "Cl",
    "F",
    "H",
    "I",
    "K",
    "Li",
    "Mg",
    "N",
    "Na",
    "O",
    "P",
    "S",
    "Se",
    "Si",
    "Te",
    "Zn",
]


CHIRALITY = [
    rdkit.Chem.rdchem.ChiralType.CHI_OTHER,
    rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
]


HYBRIDIZATION = [
    rdkit.Chem.rdchem.HybridizationType.OTHER,
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
]


def mol_to_dgl(mol, requires_input_grad=False):
    g = dgl.DGLGraph()
    g.add_nodes(mol.GetNumAtoms())
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    features = []

    pd = GetPeriodicTable()
    ComputeGasteigerCharges(mol)

    for atom in mol.GetAtoms():
        atom_feat = []
        atom_type = [0] * len(ATOM_TYPES)
        atom_type[ATOM_TYPES.index(atom.GetSymbol())] = 1

        chiral = [0] * len(CHIRALITY)
        chiral[CHIRALITY.index(atom.GetChiralTag())] = 1

        ex_valence = atom.GetExplicitValence()
        charge = atom.GetFormalCharge()

        hybrid = [0] * len(HYBRIDIZATION)
        hybrid[HYBRIDIZATION.index(atom.GetHybridization())] = 1

        valence = atom.GetImplicitValence()
        aromatic = int(atom.GetIsAromatic())
        ex_hs = atom.GetNumExplicitHs()
        im_hs = atom.GetNumImplicitHs()
        rad = atom.GetNumRadicalElectrons()
        ring = int(atom.IsInRing())

        mass = pd.GetAtomicWeight(atom.GetSymbol())
        vdw = pd.GetRvdw(atom.GetSymbol())
        pcharge = float(atom.GetProp('_GasteigerCharge'))

        atom_feat.extend(atom_type)
        atom_feat.extend(chiral)
        atom_feat.append(ex_valence)
        atom_feat.append(charge)
        atom_feat.extend(hybrid)
        atom_feat.append(valence)
        atom_feat.append(aromatic)
        atom_feat.append(ex_hs)
        atom_feat.append(im_hs)
        atom_feat.append(rad)
        atom_feat.append(ring)
        atom_feat.append(mass)
        atom_feat.append(vdw)
        atom_feat.append(pcharge)
        features.append(atom_feat)

    for bond in mol.GetBonds():
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    g.ndata["feat"] = torch.FloatTensor(features)
    g.ndata["feat"].requires_grad = requires_input_grad
    return g


class GraphData(Dataset):
    def __init__(self, inchi, labels, requires_input_grad=False):
        self.inchi = inchi
        self.labels = np.array(labels, dtype=np.float32)
        self.requires_input_grad = requires_input_grad

        assert len(self.inchi) == len(self.labels)

    def __getitem__(self, idx):
        return (
            mol_to_dgl(
                MolFromInchi(self.inchi[idx]),
                requires_input_grad=self.requires_input_grad,
            ),
            self.labels[idx]
        )

    def __len__(self):
        return len(self.inchi)


def collate_pair(samples):
    graphs_i, labels = map(list, zip(*samples))
    batched_graph_i = dgl.batch(graphs_i)
    return batched_graph_i, torch.Tensor(labels)


if __name__ == "__main__":
    inchis = ["InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"] * 128
    labels = [0.0] * 128

    gd = GraphData(inchis, labels)
    g_i, label = gd[1]

    from torch.utils.data import DataLoader

    data_loader = DataLoader(gd, batch_size=32, shuffle=True, collate_fn=collate_pair)

    b_i, labels = next(iter(data_loader))

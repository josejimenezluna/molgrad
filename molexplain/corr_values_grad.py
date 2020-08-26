import os
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit.Chem.inchi import MolFromInchi
from tqdm import tqdm

from molexplain.net import MPNNPredictor
from molexplain.train import DEVICE
from molexplain.utils import DATA_PATH, MODELS_PATH
from molexplain.vis import molecule_importance


if __name__ == "__main__":
    with open(os.path.join(DATA_PATH, "ppb", "data_ppb.pt"), "rb") as handle:
        inchis, values = pickle.load(handle)

    model_pt = os.path.join(MODELS_PATH, "ppb_noHs.pt")

    model = MPNNPredictor(node_in_feats=49,
                        edge_in_feats=10,
                        global_feats=4,
                        n_tasks=1).to(DEVICE)
 
    model.load_state_dict(torch.load(model_pt,
                                    map_location=DEVICE))

    atom_importances = []

    for inchi in tqdm(inchis):
        mol = MolFromInchi(inchi)
        _, _, ai, _, _ = molecule_importance(MolFromInchi(inchi),
                                            model,
                                            task=0,
                                            version=2,
                                            addHs=False)
        atom_importances.append(ai)

    with open(os.path.join(DATA_PATH, 'ppb', 'atom_importances.pt'), 'wb') as handle:
        pickle.dump(atom_importances, handle)

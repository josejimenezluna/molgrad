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


DATASETS = ["caco2", "herg", "cyp", "ppb"]
VERSION = 1
OUTPUT_F = None

if __name__ == "__main__":
    for data in DATASETS:
        print("Computing atom importances for dataset {}...".format(data))

        with open(
            os.path.join(DATA_PATH, f"{data}", f"data_{data}.pt"), "rb"
        ) as handle:
            inchis, values = pickle.load(handle)

        model_pt = os.path.join(MODELS_PATH, f"{data}_noHs_notest.pt")

        if data == "cyp":
            OUTPUT_F = torch.sigmoid

        model = MPNNPredictor(
            node_in_feats=49,
            edge_in_feats=10,
            global_feats=4,
            output_f=OUTPUT_F,
            n_tasks=1,
        ).to(DEVICE)

        model.load_state_dict(torch.load(model_pt, map_location=DEVICE))

        atom_importances = []

        for inchi in tqdm(inchis):
            mol = MolFromInchi(inchi)
            _, _, ai, _, _ = molecule_importance(
                MolFromInchi(inchi), model, task=0, version=VERSION, addHs=False
            )
            atom_importances.append(ai)

        with open(
            os.path.join(DATA_PATH, f"{data}", f"atom_importances_v{VERSION}.pt"), "wb"
        ) as handle:
            pickle.dump(atom_importances, handle)

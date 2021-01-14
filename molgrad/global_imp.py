import os
import pickle

import torch
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from rdkit.Chem import MolFromInchi
from molgrad.net import MPNNPredictor
from molgrad.train import TASK_GUIDE, SEED, N_FOLDS, DEVICE
from molgrad.utils import DATA_PATH, MODELS_PATH
from molgrad.vis import molecule_importance


if __name__ == "__main__":
    for data in TASK_GUIDE.keys():
        print(f"Now computing oof global importances for dataset {data}...")

        with open(
            os.path.join(DATA_PATH, f"{data}", f"data_{data}.pt"), "rb"
        ) as handle:
            inchis, values = pickle.load(handle)

        inchis = np.array(inchis)
        values = np.array(values)[:, np.newaxis]

        # Only CYP requires sigmoid act. function
        if data == "cyp":
            output_f = torch.sigmoid
        else:
            output_f = None

        # Using production model
        print("Production model running...")
        w_path = os.path.join(MODELS_PATH, f"{data}_noHs.pt")

        model = MPNNPredictor(
            node_in_feats=49,
            edge_in_feats=10,
            global_feats=4,
            n_tasks=1,
            output_f=output_f,
        ).to(DEVICE)

        model.load_state_dict(torch.load(w_path, map_location=DEVICE))

        gis = [
            molecule_importance(MolFromInchi(inchi), model)[4] for inchi in tqdm(inchis)
        ]
        global_importances = np.vstack(gis)
        np.save(
            os.path.join(DATA_PATH, f"importances{data}.npy"), arr=global_importances
        )

        # Using oof models
        global_importances_oof = []

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        for idx_split, (_, idx_test) in enumerate(kf.split(inchis)):
            print("Split {}/{} running...".format(idx_split + 1, N_FOLDS))
            inchis_test, values_test = (
                inchis[idx_test].tolist(),
                values[idx_test, :].squeeze().tolist(),
            )

            w_path = os.path.join(MODELS_PATH, f"{data}_noHs_fold{idx_split}.pt")

            model = MPNNPredictor(
                node_in_feats=49,
                edge_in_feats=10,
                global_feats=4,
                n_tasks=1,
                output_f=output_f,
            ).to(DEVICE)

            model.load_state_dict(torch.load(w_path, map_location=DEVICE))

            gis = [
                molecule_importance(MolFromInchi(inchi), model)[4]
                for inchi in tqdm(inchis_test)
            ]
            global_importances_oof.extend(gis)

        global_importances_oof = np.vstack(global_importances_oof)

        np.save(
            os.path.join(DATA_PATH, f"importances_oof_{data}.npy"),
            arr=global_importances_oof,
        )

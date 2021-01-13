import os
import pickle

import numpy as np
import torch
from sklearn.model_selection import KFold

from molgrad.prod import predict
from molgrad.scaffold import sim_matrix
from molgrad.train import N_FOLDS, SEED, TASK_GUIDE
from molgrad.utils import DATA_PATH, MODELS_PATH

if __name__ == "__main__":
    for data in TASK_GUIDE.keys():
        print(
            "Now computing scaffold out-of-fold similarity and difference matrices for endpoint {}".format(
                data
            )
        )
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

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        for idx_split, (_, idx_test) in enumerate(kf.split(inchis)):
            print("Split {}/{} running...".format(idx_split + 1, N_FOLDS))
            inchis_test, values_test = inchis[idx_test].tolist(), values[idx_test, :].squeeze().tolist()

            # Chemical similarity oof
            sims = sim_matrix(inchis_test)

            np.save(
                os.path.join(
                    DATA_PATH, f"{data}", f"sim_{data}_fold{idx_split}.npy"
                ),
                arr=sims,
            )

            # Experimental difference
            diff_exp = np.subtract.outer(values_test, values_test)
            np.save(
                os.path.join(DATA_PATH, f"{data}", f"diff_exp_fold{idx_split}.npy"),
                arr=diff_exp,
            )

            # Predicted difference
            w_path = os.path.join(MODELS_PATH, f"{data}_noHs_fold{idx_split}.pt")
            preds = predict(inchis_test, w_path, output_f=output_f).squeeze().cpu().numpy()
            np.save(
                os.path.join(DATA_PATH, f"{data}", f"preds_fold{idx_split}.npy"),
                arr=preds,
            )

            diff_hat = np.subtract.outer(preds, preds)
            np.save(
                os.path.join(DATA_PATH, f"{data}", f"diff_hat_fold{idx_split}.npy"),
                arr=diff_hat,
            )

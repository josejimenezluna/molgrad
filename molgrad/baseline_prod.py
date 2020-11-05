import os
import pickle

import numpy as np
from joblib import dump
from rdkit.Chem.inchi import MolFromInchi
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold

from molgrad.baseline_utils import featurize_ecfp4
from molgrad.utils import BASELINE_MODELS_PATH, DATA_PATH

TASK_GUIDE = {
    "ppb": "regression",
    "caco2": "regression",
    "herg": "regression",
    "cyp": "binary",
}
N_FOLDS = 10
N_ESTIMATORS = 1000


if __name__ == "__main__":
    os.makedirs(BASELINE_MODELS_PATH, exist_ok=True)

    for data in TASK_GUIDE.keys():
        print(f"Now processing {data} dataset...")
        task = TASK_GUIDE[data]

        if task == "regression":
            base_model = RandomForestRegressor

        elif task == "binary":
            base_model = RandomForestClassifier

        else:
            raise ValueError("Task not supported")

        with open(
            os.path.join(DATA_PATH, f"{data}", f"data_{data}.pt"), "rb"
        ) as handle:
            inchis, values = pickle.load(handle)

        inchis = np.array(inchis)
        values = np.array(values)[:, np.newaxis]
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1337)

        fps = np.vstack([featurize_ecfp4(MolFromInchi(inchi)) for inchi in inchis])
        rf = base_model(n_estimators=N_ESTIMATORS, n_jobs=-1)
        rf.fit(fps, values.squeeze())

        if task == "regression":
            yhat = rf.predict(fps)
        elif task == "binary":
            yhat = rf.predict_proba(fps)

        np.save(
            os.path.join(
                DATA_PATH, f"{data}", f"{data}_yhat_rf.npy"
            ),
            arr=yhat,
        )
        dump(
            rf, os.path.join(BASELINE_MODELS_PATH, f"rf_{data}.pt")
        )

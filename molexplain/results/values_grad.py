import os
import pickle

import numpy as np
from molexplain.results.method_agreement import N_VERSIONS
from molexplain.train import TASK_GUIDE
from molexplain.utils import DATA_PATH

if __name__ == "__main__":
    with open(os.path.join(DATA_PATH, "importances.pt"), "rb") as handle:
        importances = pickle.load(handle)

    corr_data = {}

    for data in TASK_GUIDE.keys():
        with open(os.path.join(DATA_PATH, f"data_{data}.pt"), "rb") as handle:
            _, values = pickle.load(handle)

        global_imp = np.array(importances[f"{data}_global"][1], dtype=np.float32)
        corrs = [np.corrcoef(values, global_imp[:, idx])[0, 1] for idx in global_imp.shape[1]]
        corr_data[f'{data}'] = corrs

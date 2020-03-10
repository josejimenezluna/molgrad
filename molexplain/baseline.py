import os
import multiprocessing

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.inchi import MolFromInchi
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from molexplain.utils import PROCESSED_DATA_PATH
from molexplain.train import rmse

FP_SIZE = 1024
NUM_WORKERS = multiprocessing.cpu_count()


def featurize(inchis):
    feats = []

    for inchi in inchis:
        mol = MolFromInchi(inchi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_SIZE)
        arr = np.zeros((1,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        feats.append(arr)
    return np.vstack(feats)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "CHEMBL3301372.csv"), header=0)
    # df['st_value'] = -np.log10(1e-9 *  df['st_value'])
    df_train, df_test = train_test_split(df, test_size=.2, random_state=1337)

    features_train = featurize(df_train['inchi'].to_list())
    features_test = featurize(df_test['inchi'].to_list())

    rf = RandomForestRegressor(n_estimators=1000, n_jobs=1)
    rf.fit(features_train, df_train['st_value'].values)

    yhat = rf.predict(features_test)

    print(np.corrcoef((df_test['st_value'].values, yhat)))
    print(rmse(df_test['st_value'].values, yhat))

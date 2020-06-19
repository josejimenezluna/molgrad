import os
import numpy as np
import pandas as pd
import pickle

from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.inchi import MolFromInchi, MolToInchi

from molexplain.utils import PROCESSED_DATA_PATH, RAW_DATA_PATH


RDLogger.DisableLog("rdApp.*")

def clean_data(smiles):
    """
    Performs SMILES to INCHI conversion.
    """
    inchis = []
    invalid_idx = []

    for idx, sm in enumerate(smiles):
        try:
            mol = MolFromSmiles(sm)
            inchi = MolToInchi(mol)
            mol_back = MolFromInchi(inchi)
            if mol_back is not None:
                inchis.append(inchi)
            else:
                invalid_idx.append(idx)
        except:
            invalid_idx.append(idx)
            continue
    return inchis, invalid_idx

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'CYP3A4.csv'), header=0, sep=';')
    smiles = df['SMILES'].to_numpy()

    inchis, invalid_idx = clean_data(smiles)

    inchis = np.array(inchis)
    values = np.array([1.0 if l == 'Active' else 0.0 for l in df['Class']])[:, np.newaxis]
    value_idx = np.setdiff1d(np.arange(len(values)), np.array(invalid_idx))
    values = values[value_idx, :]
    mask = np.array([True for l in range(values.shape[0])])[:, np.newaxis]

    with open(os.path.join(PROCESSED_DATA_PATH, 'cyp_data.pt'), 'wb') as handle:
        pickle.dump((inchis, values, mask), handle)

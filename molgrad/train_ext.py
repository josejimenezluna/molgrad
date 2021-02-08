import argparse
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from rdkit.Chem import MolFromSmiles

from molgrad.net import MPNNPredictor
from molgrad.net_utils import GraphData, collate_pair
from molgrad.train import (
    BATCH_SIZE,
    DEVICE,
    INITIAL_LR,
    N_EPOCHS,
    N_MESSPASS,
    NUM_WORKERS,
    train_loop,
)
from molgrad.utils import DATA_PATH, LOG_PATH, MODELS_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s || %(name)s | %(levelname)s: %(message)s",
    datefmt="%Y/%m/%d %I:%M:%S %p",
)
LOGGER = logging.getLogger("molgrad_train")



def smiles_check(smiles):
    failed_idx = []
    for idx, smi in tqdm(enumerate(smiles), total=len(smiles)):
        mol = MolFromSmiles(smi)
        if mol is None:
            failed_idx.append(idx)
    return failed_idx



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train your own message passing model")

    parser.add_argument(
        "-data",
        dest="data",
        type=str,
        required=True,
        help="Path to a comma-separeted (.csv) file with SMILES and a target variable",
    )

    parser.add_argument(
        "-smiles_col",
        dest="smiles_col",
        type=str,
        required=True,
        help="Name of the column with target smiles",
    )

    parser.add_argument(
        "-target_col",
        dest="target_col",
        type=str,
        required=True,
        help="Name of the column with the target values",
    )

    parser.add_argument(
        "-output",
        dest="output",
        type=str,
        required=True,
        help="Output path for storing model weights.",
    )

    parser.add_argument(
        "-task",
        dest="task",
        type=str,
        required=False,
        default="regression",
        help="Type of training tasks. Options: regression, binary",
    )

    parser.add_argument(
        "-epochs",
        dest="epochs",
        type=int,
        required=False,
        default=N_EPOCHS,
        help="Number of training epochs. Default 250",
    )

    parser.add_argument(
        "-lr",
        dest="lr",
        type=float,
        required=False,
        default=INITIAL_LR,
        help="Learning rate for SGD. Default 1e-4",
    )

    parser.add_argument(
        "-workers",
        dest="workers",
        type=int,
        required=False,
        default=NUM_WORKERS,
        help="Number of CPU threads to use. Default all available threads.",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        LOGGER.info(
            f"Using device {torch.cuda.get_device_name()} for training."
        )
    else:
        LOGGER.warning(
            f"A CUDA-capable device was not found. Using CPU for training, which can take considerably longer."
        )

    if args.task == "regression":
        loss_fn = F.mse_loss

    elif args.task == "binary":
        loss_fn = F.binary_cross_entropy_with_logits
    else:
        raise ValueError("Task type not recognized. Use either ´regression´ or ´binary´.")

    df = pd.read_csv(args.data)
    smiles, values = df[args.smiles_col].values, df[args.target_col].values
    
    LOGGER.info("Verifying SMILES integrity...")
    failed_idx = smiles_check(smiles)
    if len(failed_idx) > 0:
        LOGGER.warning(f"{len(failed_idx)} SMILES strings could not be parsed by RDkit. Removing them from training.")
        smiles = smiles[np.setdiff1d(np.arange(len(smiles)), failed_idx)]
        values = values[np.setdiff1d(np.arange(len(values)), failed_idx)]
    

    values = values[:, np.newaxis]
    mask = np.array([True for l in range(values.shape[0])])[:, np.newaxis]

    data_train = GraphData(smiles, values, mask, add_hs=False, inchi=False)

    sample_item = data_train[0]
    a_dim = sample_item[0].ndata["feat"].shape[1]
    e_dim = sample_item[0].edata["feat"].shape[1]
    g_dim = len(sample_item[1])


    loader_train = DataLoader(
        data_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_pair,
        num_workers=NUM_WORKERS,
    )

    model = MPNNPredictor(
        node_in_feats=a_dim,
        edge_in_feats=e_dim,
        global_feats=g_dim,
        n_tasks=values.shape[1],
        num_step_message_passing=N_MESSPASS,
        output_f=None,
    ).to(DEVICE)

    opt = Adam(model.parameters(), lr=INITIAL_LR)

    LOGGER.info("Now training...")

    train_losses = []

    for epoch_no in range(N_EPOCHS):
        print("Train epoch {}/{}...".format(epoch_no + 1, N_EPOCHS))
        t_l = train_loop(loader_train, model, loss_fn, opt)
        train_losses.extend(t_l)

    torch.save(model.state_dict(), args.output)

import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from molexplain.net import MPNNPredictor
from molexplain.net_utils import GraphData, collate_pair
from molexplain.utils import DATA_PATH, MODELS_PATH, LOG_PATH
from molexplain.train import (
    N_MESSPASS,
    BATCH_SIZE,
    INITIAL_LR,
    N_EPOCHS,
    DEVICE,
    NUM_WORKERS,
    rmse,
    train_loop,
)

TASK = "regression"

if __name__ == "__main__":
    if TASK == "regression":
        loss_fn = F.mse_loss

    elif TASK == "binary":
        loss_fn = F.binary_cross_entropy_with_logits

    else:
        raise ValueError("Task not supported")

    # public training
    with open(os.path.join(DATA_PATH, "caco2", "data_caco2.pt"), "rb") as handle:
        inchis, values = pickle.load(handle)

    inchis = np.array(inchis)
    values = np.array(values)[:, np.newaxis]
    mask = np.array([True for l in range(values.shape[0])])[:, np.newaxis]

    data = GraphData(inchis, values, mask, add_hs=False)

    sample_item = data[0]
    a_dim = sample_item[0].ndata["feat"].shape[1]
    e_dim = sample_item[0].edata["feat"].shape[1]
    g_dim = len(sample_item[1])

    loader_train = DataLoader(
        data,
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

    train_losses = []

    for epoch_no in range(N_EPOCHS):
        print("Train epoch {}/{}...".format(epoch_no + 1, N_EPOCHS))
        t_l = train_loop(loader_train, model, loss_fn, opt)
        train_losses.extend(t_l)

    os.makedirs(MODELS_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_PATH, "caco2_noHs_prod.pt"))

    os.makedirs(LOG_PATH, exist_ok=True)
    np.save(os.path.join(LOG_PATH, 'caco2_noHs_prod.pt'), arr=train_losses)

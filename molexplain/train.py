import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from molexplain.net import MPNNPredictor
from molexplain.net_utils import GraphData, collate_pair
from molexplain.utils import DATA_PATH, MODELS_PATH

N_MESSPASS = 12

BATCH_SIZE = 32
INITIAL_LR = 1e-4
N_EPOCHS = 250

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = multiprocessing.cpu_count()

rmse = lambda x, y: np.sqrt(np.mean((x - y) ** 2))


def train_loop(loader, model, loss_fn, opt):
    """
    Runs an entire training epoch for `model` using the data stored in `loader`,
    the loss function `loss_fn` and optimizer `opt`.
    """
    model = model.train()
    progress = tqdm(loader)

    losses = []

    for g, g_feat, label, mask in progress:
        g = g.to(DEVICE)
        g_feat = g_feat.to(DEVICE)
        label = label.to(DEVICE)
        mask = mask.to(DEVICE)
        label = label[mask]

        opt.zero_grad()
        out = model(g, g_feat)

        out = out[mask]
        loss = loss_fn(out, label)
        loss.backward()
        opt.step()

        progress.set_postfix({"loss": loss.item()})
        losses.append(loss.item())
    return losses


def eval_loop(loader, model, progress=True):
    """
    Computes prediction for all the data stored in `loader` for `model`. 
    """
    model = model.eval()
    if progress:
        loader = tqdm(loader)

    ys = []
    yhats = []
    masks = []

    for g, g_feat, label, mask in loader:
        with torch.no_grad():
            g = g.to(DEVICE)
            g_feat = g_feat.to(DEVICE)
            out = model(g, g_feat)
            ys.append(label.cpu())
            yhats.append(out.cpu())
            masks.append(mask)
    return torch.cat(ys), torch.cat(yhats), torch.cat(masks)


def metrics(ys, yhats, masks):
    """
    Computes correlation coefficient and RMSE between target `ys` and
    predicted `yhats` values, taking into account missing values specified by `masks`.
    """
    n_tasks = ys.shape[1]
    rmses = []
    corrs = []

    for task_no in range(n_tasks):
        y, yhat = (
            ys[masks[:, task_no], task_no].numpy(),
            yhats[masks[:, task_no], task_no].numpy(),
        )
        rmses.append(rmse(y, yhat))
        corrs.append(np.corrcoef(y, yhat)[0, 1])
    return rmses, corrs


if __name__ == "__main__":
    # caco2 public training
    with open(os.path.join(DATA_PATH, "caco2", "data_caco2.pt"), "rb") as handle:
        inchis, values = pickle.load(handle)

    inchis = np.array(inchis)
    values = np.array(values)[:, np.newaxis]
    mask = np.array([True for l in range(values.shape[0])])[:, np.newaxis]

    idx_train, idx_test = train_test_split(
        np.arange(len(inchis)), test_size=0.2, random_state=1337
    )

    inchis_train, inchis_test = inchis[idx_train], inchis[idx_test]
    values_train, values_test = values[idx_train, :], values[idx_test, :]
    mask_train, mask_test = mask[idx_train, :], mask[idx_test, :]

    data_train = GraphData(inchis_train, values_train, mask, add_hs=False)
    data_test = GraphData(inchis_test, values_test, mask_test, add_hs=False)

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

    loader_test = DataLoader(
        data_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
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
        t_l = train_loop(loader_train, model, F.mse_loss, opt)
        train_losses.extend(t_l)

        y_test, yhat_test, mask_test = eval_loop(loader_test, model, progress=False)
        rmse_test, corr_test = metrics(y_test, yhat_test, mask_test)
        print(
            "Test RMSE:[{}], R: [{}]".format(
                "\t".join("{:.3f}".format(x) for x in rmse_test),
                "\t".join("{:.3f}".format(x) for x in corr_test),
            )
        )

    os.makedirs(os.path.join(MODELS_PATH), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_PATH, "caco2_noHs.pt"))

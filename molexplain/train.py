import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from molexplain.net import GAT
from molexplain.net_utils import GraphData, collate_pair
from molexplain.utils import PROCESSED_DATA_PATH, MODELS_PATH

NUM_LAYERS = 6
NUM_HEADS = 12
NUM_HIDDEN = 128
NUM_GLOBAL_HIDDEN = 32
NUM_OUTHEADS = 32

BATCH_SIZE = 32
INITIAL_LR = 1e-4
N_EPOCHS = 20

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
        loss = loss_fn(label, out)
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
    rs = []
    rmses = []

    for task_no in range(n_tasks):
        y, yhat = (
            ys[masks[:, task_no], task_no].numpy(),
            yhats[masks[:, task_no], task_no].numpy(),
        )
        rs.append(np.corrcoef((y, yhat))[0, 1])
        rmses.append(rmse(y, yhat))
    return rs, rmses


if __name__ == "__main__":
    inchis = np.load(os.path.join(PROCESSED_DATA_PATH, "inchis.npy"))
    values = np.load(os.path.join(PROCESSED_DATA_PATH, "values.npy"))
    mask = np.load(os.path.join(PROCESSED_DATA_PATH, "mask.npy"))

    idx_train, idx_test = train_test_split(
        np.arange(len(inchis)), test_size=0.2, random_state=1337
    )

    inchis_train, inchis_test = inchis[idx_train], inchis[idx_test]
    values_train, values_test = values[idx_train, :], values[idx_test, :]
    mask_train, mask_test = mask[idx_train, :], mask[idx_test, :]

    data_train = GraphData(inchis_train, values_train, mask_train)
    data_test = GraphData(inchis_test, values_test, mask_test)

    sample_item = data_train[0]
    in_dim = sample_item[0].ndata["feat"].shape[1]
    n_global = len(sample_item[1])

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

    model = GAT(
        num_layers=NUM_LAYERS,
        in_dim=in_dim,
        n_global=n_global,
        num_hidden=NUM_HIDDEN,
        global_hidden=NUM_GLOBAL_HIDDEN,
        num_classes=values.shape[1],
        heads=([NUM_HEADS] * NUM_LAYERS) + [NUM_OUTHEADS],
        activation=F.relu,
        residual=True,
    ).to(DEVICE)

    opt = Adam(model.parameters(), lr=INITIAL_LR)

    train_losses = []

    for epoch_no in range(N_EPOCHS):
        print("Train epoch {}/{}...".format(epoch_no + 1, N_EPOCHS))
        t_l = train_loop(loader_train, model, F.mse_loss, opt)
        train_losses.extend(t_l)

        y_test, yhat_test, mask_test = eval_loop(loader_test, model, progress=False)
        r, rmse_ = metrics(y_test, yhat_test, mask_test)
        print(
            "Test R:[{}], RMSE: [{}]".format(
                "\t".join("{:.3f}".format(x) for x in r),
                "\t".join("{:.3f}".format(x) for x in rmse_),
            )
        )

    os.makedirs(os.path.join(MODELS_PATH), exist_ok=True)
    torch.save(model, os.path.join(MODELS_PATH, "AZ_ChEMBL_global.pt"))

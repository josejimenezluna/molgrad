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

from molexplain.net_gat import GAT
from molexplain.net_utils import GraphData, collate_pair
from molexplain.utils import PROCESSED_DATA_PATH

BATCH_SIZE = 32
N_EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = multiprocessing.cpu_count()

rmse = lambda x, y: np.sqrt(np.mean((x - y) ** 2))


def train_loop(loader, model, loss_fn, opt):
    model = model.train()
    progress = tqdm(loader)

    losses = []

    for g, label in progress:
        g = g.to(DEVICE)
        label = label.unsqueeze(1).to(DEVICE)

        opt.zero_grad()
        out = model(g).squeeze(0)
        loss = loss_fn(label, out)
        loss.backward()
        opt.step()

        progress.set_postfix({"loss": loss.item()})
        losses.append(loss.item())
    return losses


def eval_loop(loader, model, progress=True):
    model = model.eval()
    if progress:
        loader = tqdm(loader)

    ys = []
    yhats = []

    for g, label in loader:
        with torch.no_grad():
            g = g.to(DEVICE)
            out = model(g)
            ys.append(label.unsqueeze(1).cpu())
            yhats.append(out.squeeze(0).cpu())
    return torch.cat(ys), torch.cat(yhats)


def metrics(ys, yhats):
    r = np.corrcoef((ys.squeeze().numpy(), yhats.squeeze().numpy()))[0, 1]
    rmse_ = rmse(ys.squeeze().numpy(), yhats.squeeze().numpy())
    return r, rmse_


if __name__ == "__main__":
    inchis = np.load(os.path.join(PROCESSED_DATA_PATH, 'inchis.npy'))
    values = np.load(os.path.join(PROCESSED_DATA_PATH, 'values.npy'))
    mask = np.load(os.path.join(PROCESSED_DATA_PATH, 'mask.npy'))

    data_train = GraphData(inchis, values, mask)
    # data_test = GraphData(inchis, values, mask)

    loader_train = DataLoader(
        data_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_pair,
        num_workers=NUM_WORKERS,
    )

    # loader_test = DataLoader(
    #     data_test,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     collate_fn=collate_pair,
    #     num_workers=NUM_WORKERS,
    # )

    model = GAT(
        num_layers=6,
        in_dim=45,
        num_hidden=128,
        num_classes=1,
        heads=([12] * 6) + [32],
        activation=F.relu,
        residual=True,
    ).to(DEVICE)

    opt = Adam(model.parameters(), lr=1e-5)

    train_losses = []

    for epoch_no in range(N_EPOCHS):
        print("Train epoch {}/{}...".format(epoch_no + 1, N_EPOCHS))
        t_l = train_loop(loader_train, model, F.mse_loss, opt)
        train_losses.extend(t_l)

        # y_test, yhat_test = eval_loop(loader_test, model, progress=False)
        # r, rmse_ = metrics(y_test, yhat_test)
        # print("Test R: {:.2f}, RMSE: {:.2f}".format(r, rmse_))

import os
import multiprocessing

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from molexplain.net import Regressor
from molexplain.net_utils import GraphData, collate_pair
from molexplain.utils import PROCESSED_DATA_PATH

BATCH_SIZE = 32
N_EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = multiprocessing.cpu_count()


def train_loop(loader, model, loss_fn, opt):
    model = model.train()
    progress = tqdm(loader)

    losses = []

    for g, label in progress:
        g = g.to(DEVICE)
        label = label.unsqueeze(1).to(DEVICE)

        opt.zero_grad()
        out = model(g)
        loss = loss_fn(label, out)
        loss.backward()
        opt.step()

        progress.set_postfix({"loss": loss.item()})
        losses.append(loss.item())
    return losses


def eval_loop(loader, model):
    model = model.eval()
    progress = tqdm(loader)

    ys = []
    yhats = []

    for g, label in progress:
        with torch.no_grad():
            g = g.to(DEVICE)
            out = model(g)
            ys.append(label.unsqueeze(1).cpu())
            yhats.append(out.cpu())
    return torch.cat(ys), torch.cat(yhats)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "CHEMBL3301365.csv"), header=0)
    data = GraphData(df.inchi.to_list(), df.st_value.to_list())

    loader = DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_pair,
        num_workers=NUM_WORKERS,
    )

    model = Regressor(in_dim=42).to(DEVICE)
    opt = Adam(model.parameters())

    train_losses = []

    for epoch_no in range(N_EPOCHS):
        print("Epoch {}/{}...".format(epoch_no + 1, N_EPOCHS))
        t_l = train_loop(loader, model, F.mse_loss, opt)
        train_losses.extend(t_l)

    loader = DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pair, num_workers=NUM_WORKERS
    )
    ys, yhats = eval_loop(loader, model)
    print(np.corrcoef((ys.squeeze().numpy(), yhats.squeeze().numpy())))

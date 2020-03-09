import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from molexplain.net import Regressor
from molexplain.net_utils import GraphData
from molexplain.utils import PROCESSED_DATA_PATH

BATCH_SIZE = 32
N_EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(loader, model, loss_fn, opt):
    model = model.train()
    progress = tqdm(loader)

    losses = []

    for g, label in progress:
        g = g.to(DEVICE)
        label = label.to(DEVICE)

        opt.zero_grad()
        out = model(g)
        loss = loss_fn(label, out)
        opt.step()

        progress.set_postfix({'loss': loss.item()})
        losses.append(loss.item())
    return losses


def eval_loop():
    # TODO
    pass




if __name__ == "__main__":
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'CHEMBL3301365.csv'), header=0)
    data = GraphData(df.inchi.to_list(), df.st_value.to_list())

    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    model = Regressor(in_dim=42).to(DEVICE)
    opt = Adam(model.parameters())

    for epoch_no in range(N_EPOCHS):
        print('Epoch {}/{}...'.format(epoch_no + 1, N_EPOCHS))
        train_loop(loader, model, F.mse_loss, opt)


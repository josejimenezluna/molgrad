import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from molexplain.net import MPNNPredictor
from molexplain.net_utils import GraphData, collate_pair
from molexplain.utils import MODELS_PATH, PROCESSED_DATA_PATH

N_MESSPASS = 12

BATCH_SIZE = 32
INITIAL_LR = 1e-4
N_EPOCHS = 150

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
    accs = []
    aucs = []

    for task_no in range(n_tasks):
        y, yhat = (
            ys[masks[:, task_no], task_no].numpy(),
            yhats[masks[:, task_no], task_no].numpy(),
        )
        accs.append(accuracy_score(y, yhat > 0.5))
        aucs.append(roc_auc_score(y, yhat))
    return accs, aucs


if __name__ == "__main__":
    df = pd.read_csv('../cyp/CYP3A4.csv', header=0, sep=';')
    smiles = df['SMILES'].to_numpy()

    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.inchi import MolFromInchi, MolToInchi

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

    inchis = np.array(inchis)
    values = np.array([1.0 if l == 'Active' else 0.0 for l in df['Class']])[:, np.newaxis]
    value_idx = np.setdiff1d(np.arange(len(values)), np.array(invalid_idx))
    values = values[value_idx, :]
    mask = np.array([True for l in range(values.shape[0])])[:, np.newaxis]

    # idx_train, idx_test = train_test_split(
    #     np.arange(len(inchis)), test_size=0.2, random_state=1337
    # )

    # inchis_train, inchis_test = inchis[idx_train], inchis[idx_test]
    # values_train, values_test = values[idx_train, :], values[idx_test, :]
    # mask_train, mask_test = mask[idx_train, :], mask[idx_test, :]

    data_train = GraphData(inchis, values, mask, add_hs=True)
    # data_test = GraphData(inchis_test, values_test, mask_test, add_hs=False)

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

    # loader_test = DataLoader(
    #     data_test,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     collate_fn=collate_pair,
    #     num_workers=NUM_WORKERS,
    # )

    model = MPNNPredictor(
        node_in_feats=a_dim,
        edge_in_feats=e_dim,
        global_feats=g_dim,
        n_tasks=values.shape[1],
        num_step_message_passing=N_MESSPASS,
        output_f=torch.sigmoid
    ).to(DEVICE)

    opt = Adam(model.parameters(), lr=INITIAL_LR)

    train_losses = []

    for epoch_no in range(N_EPOCHS):
        print("Train epoch {}/{}...".format(epoch_no + 1, N_EPOCHS))
        t_l = train_loop(loader_train, model, F.binary_cross_entropy, opt)
        train_losses.extend(t_l)

        # y_test, yhat_test, mask_test = eval_loop(loader_test, model, progress=False)
        # acc, auc = metrics(y_test, yhat_test, mask_test)
        # print(
        #     "Test acc:[{}], AUC: [{}]".format(
        #         "\t".join("{:.3f}".format(x) for x in acc),
        #         "\t".join("{:.3f}".format(x) for x in auc),
        #     )
        # )

    os.makedirs(os.path.join(MODELS_PATH), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_PATH, "CYP3A4_Hs.pt"))

import multiprocessing
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from molgrad.net import MPNNPredictor
from molgrad.net_utils import GraphData, collate_pair
from molgrad.utils import DATA_PATH, MODELS_PATH, LOG_PATH

N_FOLDS = 10
N_MESSPASS = 12
SEED = 1337

BATCH_SIZE = 32
INITIAL_LR = 1e-4
N_EPOCHS = 250

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = multiprocessing.cpu_count()

TASK_GUIDE = {
    "ppb": "regression",
    "caco2": "regression",
    "herg": "regression",
    "cyp": "binary",
}
rmse = lambda x, y: np.sqrt(np.mean((x - y) ** 2))


def train_loop(loader, model, loss_fn, opt):
    """ Runs an entire training epoch for `model` using the data stored in `loader`,
    the loss function `loss_fn` and optimizer `opt`.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
    model : torch net
    loss_fn : torch.nn.functional function
    opt : torch.optim optimizer

    Returns
    -------
    list
        Training losses
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
    """Computes prediction for all the data stored in `loader` for `model`. 

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
    model : torch net
    progress : bool, optional
        whether to show progress bar, by default True

    Returns
    -------
    ys :
        original values
    yhats :
        predicted values
    masks :
        masks in case of multitask learning
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


def metrics(ys, yhats, masks, task='regression', b_threshold=0.5):
    """ Computes correlation coefficient and RMSE between target `ys` and
    predicted `yhats` values, taking into account missing values specified by `masks`.

    Parameters
    ----------
    ys : torch.Tensor
        original values
    yhats : torch.Tensor
        predicted values
    masks : torch.Tensor
        masks in the case of multitask learning
    b_threshold : float, optional
        threshold to filter positives for binary clas. tasks, by default 0.5

    Returns
    -------
    metrics
    """
    n_tasks = ys.shape[1]
    metric_1 = []
    metric_2 = []

    for task_no in range(n_tasks):
        y, yhat = (
            ys[masks[:, task_no], task_no].numpy(),
            yhats[masks[:, task_no], task_no].numpy(),
        )
        if task == "regression":
            metric_1.append(rmse(y, yhat))
            metric_2.append(np.corrcoef(y, yhat)[0, 1])
            metric_name_1 = "RMSE"
            metric_name_2 = "R"
        elif task == "binary":
            metric_1.append(accuracy_score(y, yhat > b_threshold))
            metric_2.append(roc_auc_score(y, yhat))
            metric_name_1 = "Acc."
            metric_name_2 = "AUC"
        else:
            raise ValueError("Task not supported.")
    return metric_1, metric_2, metric_name_1, metric_name_2


if __name__ == "__main__":
    for data in TASK_GUIDE.keys():
        print(f'Now validating k-fold predictions for dataset {data}...')
        if TASK_GUIDE[data] == "regression":
            loss_fn = F.mse_loss

        elif TASK_GUIDE[data] == "binary":
            loss_fn = F.binary_cross_entropy_with_logits

        else:
            raise ValueError("Task not supported")

        with open(os.path.join(DATA_PATH, f"{data}", f"data_{data}.pt"), "rb") as handle:
            inchis, values = pickle.load(handle)

        inchis = np.array(inchis)
        values = np.array(values)[:, np.newaxis]
        mask = np.array([True for l in range(values.shape[0])])[:, np.newaxis]
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        for idx_split, (idx_train, idx_test) in enumerate(kf.split(inchis)):
            print(f"Fold {idx_split}/{N_FOLDS}...")
            inchis_train, inchis_test = inchis[idx_train], inchis[idx_test]
            values_train, values_test = values[idx_train, :], values[idx_test, :]
            mask_train, mask_test = mask[idx_train, :], mask[idx_test, :]

            data_train = GraphData(inchis_train, values_train, mask_train, add_hs=False)
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
                t_l = train_loop(loader_train, model, loss_fn, opt)
                train_losses.extend(t_l)

                y_test, yhat_test, mask_test = eval_loop(loader_test, model, progress=False)
                metric_1, metric_2, mn1, mn2 = metrics(y_test, yhat_test, mask_test, task=TASK_GUIDE[data])
                print(
                    "Test {}:[{}], {}: [{}]".format(
                        mn1,
                        "\t".join("{:.3f}".format(x) for x in metric_1),
                        mn2,
                        "\t".join("{:.3f}".format(x) for x in metric_2),
                    )
                )

            # Save model, predictions and training losses
            os.makedirs(MODELS_PATH, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(MODELS_PATH, f"{data}_noHs_fold{idx_split}.pt"),
            )

            np.save(
                os.path.join(
                    DATA_PATH, f"{data}", f"{data}_noHs_yhat_fold{idx_split}.npy"
                ),
                arr=yhat_test.numpy(),
            )
            np.save(
                os.path.join(DATA_PATH, f"{data}", f"{data}_noHs_y_fold{idx_split}.npy"),
                arr=y_test.numpy(),
            )

            os.makedirs(LOG_PATH, exist_ok=True)
            np.save(
                os.path.join(LOG_PATH, f"{data}_noHs_fold{idx_split}.pt"),
                arr=train_losses,
            )


import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from molgrad.net import MPNNPredictor
from molgrad.net_utils import (
    GraphData,
    collate_pair_prod,
    mol_to_dgl,
    get_global_features,
)
from molgrad.train import DEVICE, NUM_WORKERS, N_MESSPASS


def predict(
    inchis, w_path, n_tasks=1, batch_size=32, output_f=None, add_hs=False, progress=True
):
    """Predicts values for a list of `inchis` given model weights `w_path`.

    Parameters
    ----------
    inchis : list
        A list of inchis that we wish to predict values for
    w_path : pickle file path
        A path to model weights, pickled.
    n_tasks : int, optional
        number of tasks, by default 1
    batch_size : int, optional
    output_f : [type], optional
        Activation function to apply on the output layer if necessary, by default None
    progress : bool, optional
        Show progress bar, by default True

    Returns
    -------
    np.ndarray
        Predictions.
    """
    data = GraphData(inchis, train=False, add_hs=add_hs)
    sample_item = data[0]
    a_dim = sample_item[0].ndata["feat"].shape[1]
    e_dim = sample_item[0].edata["feat"].shape[1]
    g_dim = len(sample_item[1])

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pair_prod,
        num_workers=NUM_WORKERS,
    )

    if progress:
        loader = tqdm(loader)

    model = MPNNPredictor(
        node_in_feats=a_dim,
        edge_in_feats=e_dim,
        global_feats=g_dim,
        n_tasks=n_tasks,
        num_step_message_passing=N_MESSPASS,
        output_f=output_f,
    ).to(DEVICE)

    model.load_state_dict(torch.load(w_path, map_location=DEVICE))

    yhats = []

    for g, g_feat in loader:
        with torch.no_grad():
            g = g.to(DEVICE)
            g_feat = g_feat.to(DEVICE)
            out = model(g, g_feat)
            yhats.append(out.cpu())
    return torch.cat(yhats)


def predict_mol(mol, model):
    pred = model(
        mol_to_dgl(mol).to(DEVICE),
        torch.Tensor(get_global_features(mol)[np.newaxis, :]).to(DEVICE),
    )
    return pred[0]

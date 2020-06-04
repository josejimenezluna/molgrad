import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from molexplain.net_utils import GraphData, collate_pair
from molexplain.train import DEVICE
from molexplain.utils import MODELS_PATH, PROCESSED_DATA_PATH


def gen_steps(graph, g_feat, n_steps):
    """
    Generates straight path between the node features of `graph`
    using a Monte Carlo approx. of `n_steps`.
    """
    graphs = []
    g_feats = []

    feat = graph.ndata["feat"]
    g_feat = torch.as_tensor(g_feat)

    for step in range(n_steps + 1):
        factor = step / n_steps
        g = deepcopy(graph)
        g.ndata["feat"] = factor * feat
        g_feat_step = (factor * g_feat).unsqueeze(0)
        graphs.append(g)
        g_feats.append(g_feat_step)
    return graphs, g_feats


def integrated_gradients(graph, g_feat, model, task, n_steps=50):
    """
    Computes path integral of the node features of `graph` for a
    specific `task` number, using a Monte Carlo approx. of `n_steps`. 
    """
    graphs, g_feats = gen_steps(graph, g_feat, n_steps=n_steps)
    values_atom = []
    values_edge = []
    values_global = []

    for g, gf in zip(graphs, g_feats):
        g = g.to(DEVICE)
        gf = gf.to(DEVICE)
        g.ndata["feat"].requires_grad_()
        g.edata["feat"].requires_grad_()
        gf.requires_grad_()

        preds = model(g, gf)

        preds[0][task].backward()
        atom_grads = g.ndata["feat"].grad.unsqueeze(2)
        edge_grads = g.edata['feat'].grad.unsqueeze(2)
        values_atom.append(atom_grads)
        values_edge.append(edge_grads)
        values_global.append(gf.grad)
    return (
        torch.cat(values_atom, dim=2).mean(dim=(1, 2)).cpu().numpy(),
        torch.cat(values_edge, dim=2).mean(dim=(1, 2)).cpu().numpy()
        torch.cat(values_global).mean(axis=0).cpu().numpy(),
    )


if __name__ == "__main__":
    from molexplain.net import MPNNPredictor


    inchis = np.load(os.path.join(PROCESSED_DATA_PATH, "inchis.npy"))
    values = np.load(os.path.join(PROCESSED_DATA_PATH, "values.npy"))
    mask = np.load(os.path.join(PROCESSED_DATA_PATH, "mask.npy"))

    model = MPNNPredictor(
        node_in_feats=46, edge_in_feats=4, n_tasks=values.shape[1]
    ).to(DEVICE)

    model.load_state_dict(torch.load(os.path.join(MODELS_PATH, 'AZ_ChEMBL_MPNN.pt')))

    data = GraphData(inchis, values, mask)
    graph, g_feat, _, _ = data[20]

    atom_importance, edge_importance, global_importance = integrated_gradients(
        graph, g_feat, model, task=3, n_steps=50
    )

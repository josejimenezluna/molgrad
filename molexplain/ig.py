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
    values_bond = []
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
        bond_grads = g.edata['feat'].grad.unsqueeze(2)
        values_atom.append(atom_grads)
        values_bond.append(bond_grads)
        values_global.append(gf.grad)
    return (
        torch.cat(values_atom, dim=2).mean(dim=(1, 2)).cpu().numpy(),
        torch.cat(values_bond, dim=2).mean(dim=(1, 2)).cpu().numpy(),
        torch.cat(values_global).mean(axis=0).cpu().numpy()
    )

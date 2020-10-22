import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from molexplain.net_utils import GraphData, collate_pair
from molexplain.train import DEVICE
from molexplain.utils import MODELS_PATH, PROCESSED_DATA_PATH


def gen_steps(graph, g_feat, n_steps, version=2):
    """
    Generates straight path between the node features of `graph`
    using a Monte Carlo approx. of `n_steps`.
    """
    graphs = []
    g_feats = []

    feat = graph.ndata["feat"]
    if version == 3:
        e_feat = graph.edata["feat"]
    g_feat = torch.as_tensor(g_feat)

    for step in range(1, n_steps + 1):
        factor = step / n_steps
        g = deepcopy(graph)
        g.ndata["feat"] = factor * feat
        if version == 3:
            g.edata["feat"] = factor * e_feat
        g_feat_step = (factor * g_feat).unsqueeze(0)
        graphs.append(g)
        g_feats.append(g_feat_step)
    return graphs, g_feats


# def noise_tunnel(graph, g_feat, n_samples=10, version=2, mean=0.0, std=1.0):
#     """Noise tunnel prototype

#     Parameters
#     ----------
#     graph : DGL graph
#     g_feat : torch.Tensor
#     n_samples : int, optional
#         number of samples to add noise to, by default 10
#     version : int, optional
#     mean : float, optional
#     std : float, optional

#     Returns
#     -------
#     list
#         A list of graphs with their respective node and edge
#         features with noise added to them.
#     """
#     graphs = []

#     feat = graph.ndata['feat']
#     e_feat = graph.edata['feat']

#     for _ in range(n_samples):
#         g = deepcopy(graph)
#         g.ndata['feat'] = feat + (mean + std * torch.randn(feat.shape))
#         if version == 3:
#             g.edata['feat'] = e_feat + (mean + std * torch.randn(e_feat.shape))
#         graphs.append(g)
#     return graphs


def integrated_gradients(graph, g_feat, model, task=0, n_steps=50, version=2):
    """Computes path integral of the node features of `graph` for a
    specific `task` number, using a Monte Carlo approx. of `n_steps`. 

    Parameters
    ----------
    graph : DGL graph
    g_feat : torch.Tensor
    model : MPNN 
    task : int, optional
    n_steps : int, optional
    version : int, optional

    Returns
    -------
    atom_importances : torch.Tensor
    bond_importances : torch.Tensor
    values_global : torch.Tensor
    """
    graphs, g_feats = gen_steps(graph, g_feat, n_steps=n_steps, version=version)
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
        bond_grads = g.edata["feat"].grad.unsqueeze(2)
        values_atom.append(atom_grads)
        values_bond.append(bond_grads)
        values_global.append(gf.grad)

    cat_atom = torch.cat(values_atom, dim=2).mean(dim=2).cpu() * graph.ndata["feat"]
    cat_bond = torch.cat(values_bond, dim=2).mean(dim=2).cpu() * graph.edata["feat"]
    cat_global = torch.cat(values_global).mean(dim=0).cpu() * g_feat
    return (
        cat_atom.mean(dim=1).numpy(),
        cat_bond.mean(dim=1).numpy(),
        cat_global.numpy(),
    )



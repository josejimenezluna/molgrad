from copy import deepcopy

import torch

from molgrad.train import DEVICE


def gen_steps(graph, n_steps, version=2):
    """
    Generates straight path between the node features of `graph`
    using a Monte Carlo approx. of `n_steps`.
    """
    graphs = []

    feat = graph.ndata["feat"]
    if version == 3:
        e_feat = graph.edata["feat"]

    for step in range(1, n_steps + 1):
        factor = step / n_steps
        g = deepcopy(graph)
        g.ndata["feat"] = factor * feat
        if version == 3:
            g.edata["feat"] = factor * e_feat
        graphs.append(g)
    return graphs



def integrated_gradients(
    graph, model, task=0, n_steps=50, version=2, feature_scale=True
):
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
    """
    graphs = gen_steps(graph, n_steps=n_steps, version=version)
    values_atom = []
    values_bond = []

    for g in graphs:
        g = g.to(DEVICE)
        g.ndata["feat"].requires_grad_()
        g.edata["feat"].requires_grad_()

        preds = model(g)

        preds[0][task].backward()
        atom_grads = g.ndata["feat"].grad.unsqueeze(2)
        bond_grads = g.edata["feat"].grad.unsqueeze(2)
        values_atom.append(atom_grads)
        values_bond.append(bond_grads)

    cat_atom = torch.cat(values_atom, dim=2).mean(dim=2).cpu()
    cat_bond = torch.cat(values_bond, dim=2).mean(dim=2).cpu()

    if feature_scale:
        cat_atom *= graph.ndata["feat"]
        cat_bond *= graph.edata["feat"]

    return (
        cat_atom.mean(dim=1).numpy(),
        cat_bond.mean(dim=1).numpy(),
    )


import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from molexplain.net_utils import GraphData, collate_pair
from molexplain.train import DEVICE
from molexplain.utils import MODELS_PATH, PROCESSED_DATA_PATH


def gen_steps(graph, n_steps):
    graphs = []
    feat = graph.ndata["feat"].detach()

    for step in range(n_steps + 1):
        g = deepcopy(graph)
        g.ndata["feat"] = step / n_steps * feat
        g.ndata["feat"].requires_grad = True
        graphs.append(g)
    return graphs


def integrated_gradients(graph, model, task, n_steps=50):
    graphs = gen_steps(graph, n_steps=n_steps)
    values_steps = []

    for g in graphs:
        # g = g.to(DEVICE)
        preds = model(g)
        preds[0][task].backward(retain_graph=True)
        atom_grads = g.ndata["feat"].grad
        values_steps.append(atom_grads)
    return torch.cat(values_steps)


if __name__ == "__main__":
    model = torch.load(os.path.join(MODELS_PATH, "AZ_ChEMBL.pt")).cpu()

    inchis = np.load(os.path.join(PROCESSED_DATA_PATH, "inchis.npy"))
    values = np.load(os.path.join(PROCESSED_DATA_PATH, "values.npy"))
    mask = np.load(os.path.join(PROCESSED_DATA_PATH, "mask.npy"))

    data = GraphData(inchis, values, mask, requires_input_grad=True)
    graph, _, _ = data[0]

    preds = model(graph)
    preds[0][0].backward(retain_graph=True)
    print(graph.ndata["feat"].grad)

    graphs = gen_steps(graph, 50)
    g = graphs[-1]
    preds = model(g)
    preds[0][0].backward(retain_graph=True)
    print(g.ndata["feat"].grad)

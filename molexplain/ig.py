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

    for step in range(n_steps + 1):
        new = graph
        new.ndata["feat"] = step / n_steps * graph.ndata["feat"]
        graphs.append(new)
    return graphs


def integrated_gradients(graph, model, task, n_steps=50):
    graphs = gen_steps(graph, n_steps=n_steps)

    values_steps = []

    for g in graphs:
        g = g.to(DEVICE)
        preds = model(g)
        preds[task].backward()
        atom_grads = g.ndata["feat"].grad
        values_steps.append(atom_grads)
    return torch.cat(values_steps)


# if __name__ == "__main__":
#     model = torch.load(os.path.join(MODELS_PATH, "AZ_ChEMBL.pt")).to(DEVICE)

#     inchis = np.load(os.path.join(PROCESSED_DATA_PATH, "inchis.npy"))
#     values = np.load(os.path.join(PROCESSED_DATA_PATH, "values.npy"))
#     mask = np.load(os.path.join(PROCESSED_DATA_PATH, "mask.npy"))

#     data = GraphData(inchis, values, mask, requires_input_grad=True)
#     data[0]

#     loader = DataLoader(
#         data,
#         batch_size=1,
#         shuffle=True,
#         collate_fn=collate_pair,
#     )

#     graph, _, _ = next(iter(loader))

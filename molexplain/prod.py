import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from molexplain.net import MPNNPredictor
from molexplain.net_utils import GraphData, collate_pair_prod
from molexplain.train import DEVICE, NUM_WORKERS, N_MESSPASS


def predict(inchis, w_path, n_tasks=1, batch_size=32, progress=True):
    data = GraphData(inchis, train=False, add_hs=False)

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
        output_f=None,
    ).to(DEVICE)

    model.load_state_dict(torch.load(w_path, map_location=DEVICE))

    yhats = []

    for g, g_feat in loader:
        with torch.no_grad():
            g = g.to(DEVICE)
            g_feat = g_feat.to(DEVICE)
            out = model(g, g_feat)
            yhats.append(out.cpu())
    return torch.cat(yhats).numpy()


if __name__ == "__main__":
    import pickle
    from molexplain.utils import MODELS_PATH, DATA_PATH

    with open(os.path.join(DATA_PATH, "cyp", "data_cyp.pt"), "rb") as handle:
        inchis, _ = pickle.load(handle)

    inchis = inchis[:50]

    w_path = os.path.join(MODELS_PATH, "cyp_noHs_notest.pt")
    yhats = predict(inchis, w_path)
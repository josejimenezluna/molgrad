import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv, edge_softmax


class GAT(nn.Module):
    """
    Graph attention neural network architecture. Adapted from
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/gat.py 
    """

    def __init__(
        self, num_layers, in_dim, num_hidden, num_classes, heads, activation, residual
    ):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.linear = nn.Linear(heads[-1], num_classes)
        # input projection (no residual)
        self.gat_layers.append(
            GATConv(
                in_feats=in_dim,
                out_feats=num_hidden,
                num_heads=heads[0],
                residual=False,
                activation=self.activation,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(
                    in_feats=num_hidden * heads[l - 1],
                    out_feats=num_hidden,
                    num_heads=heads[l],
                    residual=residual,
                    activation=self.activation,
                )
            )
        # output projection
        self.gat_layers.append(
            GATConv(
                in_feats=num_hidden * heads[-2],
                out_feats=num_classes,
                num_heads=heads[-1],
                residual=True,
                activation=None,
            )
        )

    def forward(self, g):
        h = g.ndata["feat"]
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        g.ndata["h"] = self.gat_layers[-1](g, h)
        latent = dgl.sum_nodes(g, "h").mean(axis=-1)
        if len(latent.shape) == 1:  ## TODO: Need a better solution
            latent = latent.unsqueeze(0)
        return self.linear(latent)

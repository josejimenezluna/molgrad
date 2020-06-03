import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv, edge_softmax, Set2Set

from dgllife.model.gnn.mpnn import MPNNGNN

class GAT(nn.Module):
    """
    Graph attention neural network architecture. Adapted from
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/gat.py 
    """

    def __init__(
        self,
        num_layers,
        in_dim,
        n_global,
        num_hidden,
        global_hidden,
        num_classes,
        heads,
        activation,
        residual,
    ):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.global_subnet = nn.Sequential(
            nn.Linear(n_global, global_hidden),
            nn.ReLU(),
            nn.Linear(global_hidden, global_hidden),
            nn.ReLU(),
        )

        self.linear = nn.Linear(heads[-1] + global_hidden, num_classes)
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

    def forward(self, g, g_feat):
        h = g.ndata["feat"]
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        g.ndata["h"] = self.gat_layers[-1](g, h)
        latent = dgl.sum_nodes(g, "h").mean(axis=-1)
        if len(latent.shape) == 1:  ## TODO: Need a better solution
            latent = latent.unsqueeze(0)

        latent_global = self.global_subnet(g_feat)
        cat = torch.cat([latent, latent_global], dim=1)
        return self.linear(cat)




class MPNNPredictor(nn.Module):
    """MPNN for regression and classification on graphs.

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_hidden_feats : int
        Size for the hidden edge representations. Default to 128.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    num_step_set2set : int
        Number of set2set steps. Default to 6.
    num_layer_set2set : int
        Number of set2set layers. Default to 3.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(MPNNPredictor, self).__init__()

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.predict = nn.Sequential(
            nn.Linear(2 * node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )

    def forward(self, g):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, g.ndata['feat'], g.edata['feat'])
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)
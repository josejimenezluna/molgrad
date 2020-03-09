import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')


def reduce_f(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce_f)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Regressor(nn.Module):
    def __init__(self, in_dim=42, hidden_dim=512, n_tasks=1):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)
        ])
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_tasks)

    def forward(self, g):
        h = g.ndata['feat']
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        latent = dgl.sum_nodes(g, 'h')
        latent = torch.relu(self.linear(latent))
        return self.output(latent)


# if __name__ == "__main__":
#     from molexplain.net_utils import GraphData, collate_pair
#     inchis = ["InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"] * 128
#     labels = [1] * 128

#     gd = GraphData(inchis, labels, requires_input_grad=True)
#     g, label = gd[1]

#     from torch.utils.data import DataLoader
#     n_feat = g.ndata['feat'].shape[1]

#     net = Regressor(n_feat, 128)
#     out = net(g)
#     out.backward()
    
#     print(g.ndata['feat'].grad)

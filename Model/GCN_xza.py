import torch.nn as nn
import torch
import dgl
import dgl.function as fn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# msg = fn.copy_src(src='h', out='m')

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        for data in node.data:
            node.data[data] = self.linear(node.data[data])
            node.data[data] = self.activation(node.data[data])
        return node.data

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g):
        # Initialize the node features with h.
        # g.update_all(fn.copy_src(src='in_degrees', out='m'), fn.mean('m', 'in_degrees'))
        g.update_all(fn.copy_src(src='charIDx', out='n'), fn.mean('n', 'charIDx'))
        g.apply_nodes(func=self.apply_mod)
        return g
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, embedding_matrix, device):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.embedding = nn.Embedding(embedding_matrix.size()[0], embedding_matrix.size()[1], _weight = embedding_matrix)
        self.device = device

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        # g.ndata['in_degrees'] = g.in_degrees().view(-1, 1).float()
        for data in g.ndata:
            g.ndata[data] = g.ndata[data].to(self.device)        
        g.ndata['charIDx'] = self.embedding(g.ndata['charIDx'])
        for conv in self.layers:
            g = conv(g)
        # g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'charIDx')
        return self.classify(hg)
import torch.nn as nn
import torch
import dgl
import dgl.function as fn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
msg = fn.copy_src(src='h', out='m')
def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}
class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        node.data['h'] = node.data['h'].to(device)
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')
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
        for data in g.ndata:
            g.ndata[data] = g.ndata[data].to(self.device)  
        h = self.embedding(g.ndata['charIDx'])
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
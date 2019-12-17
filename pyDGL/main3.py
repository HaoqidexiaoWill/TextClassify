import dgl
import dgl.function as fn
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from LoadData import ZhihuDataset
import networkx as nx
import sqlite3
import pandas as pd

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')
class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h' : h}
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, None)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))
class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h
net = Net()

def load_data():
    data = ZhihuDataset()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.ByteTensor(data.train_mask)
    test_mask = th.ByteTensor(data.test_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask
def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
import time
import numpy as np
g, features, labels, train_mask, test_mask = load_data()
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
dur = []
def predict(logits):
    return (np.argmax(logits, axis=-1) // 3)[:375]

def train():
    for epoch in range(100):
        if epoch >=3:
            t0 = time.time()

        net.train()
        logits = net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >=3:
            dur.append(time.time() - t0)

        acc = evaluate(net, g, features, labels, test_mask)
        model_to_save = net.module if hasattr(net, 'module') else net
        output_model_file = "./pytorch_model.bin"
        torch.save(model_to_save.state_dict(), output_model_file)

        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))
def test():
    g, features, labels, train_mask, test_mask = load_data()
    net = Net()
    net.load_state_dict(torch.load('pytorch_model.bin'))
    net.eval()
    acc = evaluate(net, g, features, labels, test_mask)
    print("Test Acc {:.4f}".format(acc))


def predicts():
    conn = sqlite3.connect("./data/zhihu.db")
    following_data = pd.read_sql('select user_url, followee_url from Following where followee_url in (select user_url from User where agree_num > 50000) and user_url in (select user_url from User where agree_num > 50000)', conn)
    user_data = pd.read_sql('select * from User where agree_num > 50000',conn)
    conn.close()
    # print(user_data.head(30))
    G = nx.DiGraph()
    cnt = 0
    for d in following_data.iterrows():
        G.add_edge(d[1][0], d[1][1])
        cnt += 1
    g, features, labels, train_mask, test_mask = load_data()
    net = Net()
    net.load_state_dict(torch.load('pytorch_model.bin'))
    net.eval()
    with th.no_grad():
        logits = net(g, features)
        logits = logits[test_mask]
    logits = logits.cpu().numpy()
    predicts = predict(logits)
    user_data['predict'] = predicts
    print(user_data.head(30))
    user_data.to_csv('result.csv')
    following_data.to_csv('graph.csv')
    test()



if __name__ == "__main__":
    # train()

    predicts()

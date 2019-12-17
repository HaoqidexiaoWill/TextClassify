import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from load_data import CoraDataset,ZhihuDataset,JingBoDataset
import time
import numpy as np
import glob
import random
def gcn_message(edges):
    # The argument is a batch of edges.
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    return {'msg' : edges.src['h']}

def gcn_reduce(nodes):
    # The argument is a batch of nodes.
    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')
    
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        # torch.nn.init.xavier_uniform(self.linear.weight)
        torch.nn.init.kaiming_normal_(self.linear.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, node):        
        h = self.linear(node.data['h'])        
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
    def __init__(self,input_num = 1433,output_num = 7):
        super(Net, self).__init__()
        # self.gcn1 = GCN(input_num, 16, F.relu)
        self.gcn1 = GCN(input_num, 32, F.relu)
        self.gcn2 = GCN(32, output_num, F.relu)

    
    def forward(self, g, features):
        # print(features.size())
        # exit()
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x

# net = Net(input_num = 542,output_num = 2 )
# net = Net(input_num = 5,output_num=2)
net = Net(input_num = 1045,output_num=2)
# net = Net()
# print(net)


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)     # indices 预测值  == label便签。correct个数
    return correct.item() * 1.0 / len(labels)  #预测正确的个数/总标签个数
def load_cora_data():
    # data = CoraDataset()
    # data = ZhihuDataset()
    data = JingBoDataset()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)

    train = data.train.cuda()
    valid = data.valid.cuda()
    test = data.test.cuda()

    g = DGLGraph(data.graph)
    return g, features, labels, train,valid,test
g, features, labels, train,valid,test = load_cora_data()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

import os
def train_model(epoch):
    t = time.time()
    net.train()
    optimizer.zero_grad()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)

    loss_train = F.nll_loss(logp[train], labels[train])
    acc_train = accuracy(logp[train], labels[train])
    loss_train.backward()
    optimizer.step()
    loss_valid = F.nll_loss(logp[valid], labels[valid])
    acc_valid = accuracy(logp[valid], labels[valid])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_valid.item()),
          'acc_val: {:.4f}'.format(acc_valid),
          'time: {:.4f}s'.format(time.time() - t))

    # return loss_train.item()
    return loss_valid.item()

def main():

    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = 10000
    best_epoch = 0
    patience = 100

    for epoch in range(10000):
        loss_values.append(train_model(epoch))
        torch.save(net.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
            if bad_counter == patience:
                break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    return  best_epoch


def compute_test(best_epoch):
    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    # net.load_state_dict(torch.load('./model_save/{}.pkl'.format(best_epoch)))
    net.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
    net.eval()
    # output = net(features, adj)
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)

    loss_test = F.nll_loss(logp[test], labels[test])
    acc_test = accuracy(logp[test], labels[test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test)
    )

if  __name__ == '__main__':
    best_epoch = main()
    # Testing
    # compute_test(114)



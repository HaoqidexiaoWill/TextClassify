import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GraphAttention import GAT
from dgl import DGLGraph
from load_data import CoraDataset,ZhihuDataset
import requests
import time
import numpy as np

def load_cora_data():
    data = CoraDataset()

    # data = ZhihuDataset()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)

    train = data.train.cuda()
    valid = data.valid.cuda()
    test = data.test.cuda()

    g = DGLGraph(data.graph)
    return g, features, labels, train,valid,test


def main():
    g, features, labels, train,valid,test = load_cora_data()
    features = features.cuda()
    labels = labels.cuda()
    train= train.cuda()
    valid = valid.cuda()
    test = test.cuda()
    # 创建模型
    net = GAT(g, in_dim=features.size()[1], hidden_dim=8, out_dim=7, num_heads=8).cuda()
    # print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # 主流程

    for epoch in range(30):
        logits = net(features)
        logp = F.log_softmax(logits, 1)
        loss_train = F.nll_loss(logp[train], labels[train])
        acc_train = evaluate(net, features, labels, train)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        acc_valid = evaluate(net, features, labels, valid)
        loss_valid = F.nll_loss(logp[valid], labels[valid])

        print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_valid.item()),
          'acc_val: {:.4f}'.format(acc_valid)
        )
    return 0
def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)
if  __name__ == '__main__':
    best_epoch = main()
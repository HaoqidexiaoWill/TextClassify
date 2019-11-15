from DATAProcess.LoadDataDoubanDGL import DataDOUBAN
from itertools import cycle
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import operator
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).to(device)
os.environ["CUDA_VISIBLE_DEVICES"]='1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# Sends a message of node feature h.
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
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = g.in_degrees().view(-1, 1).float()
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


# Create training and test sets.
trainset = DataDOUBAN(mode='train')
testset = DataDOUBAN(mode='test')
# Use PyTorch's DataLoader and the collate function
# defined before.
data_loader = DataLoader(
    trainset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate)

# Create model
model = Classifier(1, 256, trainset.num_classes).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
iter = 0
epoch_loss = 0
best_MRR = 0
data_loader = cycle(data_loader)
for epoch in range(20):
    epoch_loss = 0
    bg,label = next(data_loader)
    prediction = model(bg)
    loss = loss_func(prediction, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)


model.eval()
# Convert a list of tuples to two lists
# test_X, test_Y = map(list, zip(*testset))
# test_bg = dgl.batch(test_X)
# test_Y = torch.tensor(test_Y).float().view(-1, 1)
# probs_Y = torch.softmax(model(test_bg), 1).to('cpu').detach().numpy()
# predict_label = np.argmax(probs_Y, axis=1)
# print(predict_label.shape)
# test_Y = test_Y.to('cpu').detach().numpy()
# print(test_Y.shape)
scores = []
labels = []
ids = testset.ids
test_loader = DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate)
for iter, (bg, label) in enumerate(test_loader):
    logits = model(bg).detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    scores.append(logits)
    labels.append(label)
scores = np.concatenate(scores, 0)
labels = np.concatenate(labels, 0)



def is_valid_query(each_answer):
    # 计算指标的时候对答案标签的合法性进行判断避免除0
    num_pos = 0
    num_neg = 0
    for label, score in each_answer:
        if label > 0:
            num_pos += 1
        else:
            num_neg += 1
    if num_pos > 0 and num_neg > 0:
        return True
    else:
        return False

def compute_DOUBAN(ID,scores,labels):
    MRR,num_query = 0,0
    results = defaultdict(list)
    predict = pd.DataFrame({'scores': scores[:, 1],'labels': labels,'ID':ID})
    for index, row in predict.iterrows():
        results[row[2]].append((row[1],row[0]))

    for key,value in results.items():
        if not is_valid_query(value) : continue
        num_query +=1
        sorted_result = sorted(value, key=operator.itemgetter(1), reverse=True)
        for index_, final_result in enumerate(sorted_result):
            label,scores = final_result
            if label>0:
                MRR += 1.0/(index_+1)
                break

    predict['rank']= predict['scores'].groupby(predict['ID']).rank(ascending = False)
    predict['rec_rank'] = predict['rank'].rdiv(1)
    mrr = predict[predict['labels'] == 1]['rec_rank'].sum()/(predict[predict['labels'] == 1].shape[0])

    MAP = 0
    for key ,value in results.items():
        if not is_valid_query(value): continue
        sorted_result = sorted(value, key=operator.itemgetter(1), reverse=True)
        num_relevant_resp = 0
        AVP = 0 # 每个文档的平均准确率
        for index_,final_result in enumerate(sorted_result):
            each_label,each_score = final_result
            if each_label > 0:
                num_relevant_resp += 1
                precision  = num_relevant_resp/(index_+1)
                AVP += precision
        AVP = AVP/num_relevant_resp
        MAP += AVP

    Precision_1 = 0
    for key, value in results.items():
        if not is_valid_query(value): continue
        sorted_result = sorted(value, key=operator.itemgetter(1), reverse=True)
        # 预测的label取最后概率向量里面最大的那一个作为预测结果
        label, score = sorted_result[0]
        if label > 0:
            Precision_1 += 1

    return MRR/num_query,mrr,MAP/num_query,Precision_1/num_query


assert len(ids) == len(scores) == len(labels)
eval_DOUBAN_MRR,eval_DOUBAN_mrr,eval_DOUBAN_MAP,eval_Precision1 = compute_DOUBAN(ids,scores,labels)
print(
    'eval_MRR',eval_DOUBAN_MRR,eval_DOUBAN_mrr,
    'eval_MAP',eval_DOUBAN_MAP,
    'eval_Precision1',eval_Precision1)




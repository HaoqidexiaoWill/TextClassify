import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from load_data import CoraDataset,ZhihuDataset
import time
import numpy as np
import glob
import random
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')
def gcn_message(edges):
    # The argument is a batch of edges.
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    return {'msg' : edges.src['h']}

def gcn_reduce(nodes):
    # The argument is a batch of nodes.
    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}
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
class GCN_Net(nn.Module):
    def __init__(self,input_num = 1433,output_num = 7):
        super(GCN_Net, self).__init__()
        # self.gcn1 = GCN(input_num, 16, F.relu)
        self.gcn1 = GCN(input_num, 32, F.relu)
        self.gcn2 = GCN(32, output_num, F.relu)

    
    def forward(self, g, features):

        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x
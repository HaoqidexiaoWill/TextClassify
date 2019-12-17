from __future__ import absolute_import

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os, sys
import dgl
import torch
def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def _encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask
class ZhihuDataset(object):
    def __init__(self):
        self.name = 'cora'
        self._load()

    def _load(self):
        idx_features_labels = np.genfromtxt('./data/zhihu.content', dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = _encode_onehot(idx_features_labels[:, -1])
        self.num_labels = labels.shape[1]

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("./data/zhihu.cites", dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]),
                             (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        self.graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())

        features = _normalize(features)
        self.features = np.array(features.todense())
        self.labels = np.where(labels)[1]

        self.train_mask = _sample_mask(range(400), labels.shape[0])
        self.val_mask = _sample_mask(range(400, 500), labels.shape[0])
        self.test_mask = _sample_mask(range(500, 1500), labels.shape[0])

        self.train = torch.LongTensor(range(400))
        self.valid = torch.LongTensor(range(400, 500))
        self.test = torch.LongTensor(range(500, 1500))

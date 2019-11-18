import math
import pandas as pd
import os
import networkx as nx
import numpy as np
import dgl
import torch
from dgl.graph import DGLGraph
from collections import defaultdict
from gensim.models import word2vec 


class DataDOUBAN(object):
    def __init__(self, mode='train'):
        super(DataDOUBAN, self).__init__()
        self.mode = mode
        self.data_dir = '/home/lsy2018/TextClassification/DATA/DATA_DOUBAN/data_1102/'
        self.pretrained_emb_path = os.path.join(os.getcwd(),'Embedding', 'sgns.weibo.bigram-char')
        self.cache_dir = '/home/lsy2018/TextClassification/DATA/DATA_DOUBAN/cache_data/'
        self.vocab2IDX = {}
        self.ids = []
        self.graphs = []
        self.labels = []
        self.edge_list = []
        self.edge_weight = []
        self.build_graph()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def generate_graph(self, text, label):
        # 建立一个空的无向图G
        g = dgl.DGLGraph()
        # 分字
        text_list = [char for char in text]
        # print(label)
        # 依次遍历每个字
        for i, each_char in enumerate(text_list):
            # 依次遍历词i 之后窗口范围内的词
            if each_char not in self.vocab2IDX:
                self.vocab2IDX[each_char] = len(self.vocab2IDX)
            if not g.has_node(i):
                g.add_nodes(1)
                g.nodes[[i]].data['charIDx'] = torch.tensor([self.vocab2IDX[each_char]])
            for j in range(i + 1, i + 5):
                # 词j不能超出整个句子
                if j >= len(text_list):
                    break
                if text_list[j] not in self.vocab2IDX:
                    self.vocab2IDX[text_list[j]] = len(self.vocab2IDX)
                if not g.has_node(j):
                    g.add_nodes(1)
                    g.nodes[[j]].data['charIDx'] = torch.LongTensor([self.vocab2IDX[text_list[j]]]) 

                if not g.has_edge_between(i, j):
                    g.add_edge(i, j)
                    edgeIDxij = g.edge_id(i, j)
                    g.edges[edgeIDxij].data['edgeWeight'] = torch.zeros(1)
                    g.add_edge(j, i)
                    edgeIDxji = g.edge_id(i, j)
                    g.edges[edgeIDxji].data['edgeWeight'] = torch.zeros(1)
                else:
                    edgeIDxij = g.edge_id(i, j)
                    g.edges[edgeIDxij].data['edgeWeight'] = torch.add(g.edges[edgeIDxij].data['edgeWeight'], 1)
                    edgeIDxji = g.edge_id(j, i)
                    g.edges[edgeIDxji].data['edgeWeight'] = torch.add(g.edges[edgeIDxji].data['edgeWeight'], 1)

        # print('We have %d nodes.' % g.number_of_nodes())
        # print('We have %d edges.' % g.number_of_edges())
        self.graphs.append(g)
        self.labels.append(label)

    def build_graph(self):
        if self.mode == 'train':
            df = pd.read_csv(os.path.join(self.data_dir, 'dev.csv'), sep=',',
                             names=['dialogid', 'history', 'utterance', 'response', 'label'],nrows =5000)
        else:
            df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'), sep=',',
                             names=['dialogid', 'history', 'utterance', 'response', 'label'],nrows = 1000)
        print('行数', df.shape[0])
        examples = []
        for index, row in df.iterrows():
            self.ids.append(row[0])
            text = row[1] + row[2] + row[3]
            self.generate_graph(text, row[4])
        num_graphs = len(self.graphs)

        for i in range(num_graphs):
            # self.graphs[i] = DGLGraph(self.graphs[i])
            # add self edges
            nodes = self.graphs[i].nodes()
            self.graphs[i].add_edges(nodes, nodes)

    def build_embedding_matrix(self):
        embedding_matrix = []
        w2v = word2vec.Word2VecKeyedVectors.load_word2vec_format(self.pretrained_emb_path, binary=False)
        for word in self.vocab2IDX.keys():
            try:
                embedding_matrix.append(w2v[word])
            except:
                embedding_matrix.append(np.random.rand(w2v['a'].shape[0]))
        return torch.Tensor(embedding_matrix)

    # def write_cache(self):
    #     for graph in self.graphs:
            # print(graph.ndata['charIDx'])


        # with open(cache_path, 'w') as cache:


    @property
    def num_classes(self):
        return 2


if __name__ == '__main__':
    data = DataDOUBAN()
    # data.build_graph()
    # data.write_cache()


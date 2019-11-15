import math
import pandas as pd
import os
import networkx as nx
import numpy as np
import dgl
from dgl.graph import DGLGraph
from collections import defaultdict



class DataDOUBAN(object):
    def __init__(self, mode='train'):
        super(DataDOUBAN, self).__init__()
        self.mode = mode
        self.data_dir = '/home/lsy2018/TextClassification/DATA/DATA_DOUBAN/data_1102/'
        self.ids = []
        self.graphs = []
        self.labels = []
        self.build_graph()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def generate_graph(self, text, label):
        # 建立一个空的无向图G
        g = dgl.DGLGraph()
        # 定义共现词典
        common_mention = defaultdict(int)
        # 分字
        text_list = [char for char in text]
        # print(label)
        # 依次遍历每个词
        for i, each_char in enumerate(text_list):
            # 依次遍历词i 之后窗口范围内的词
            for j in range(i + 1, i + 5):
                # 词j 不能超出整个句子
                if j >= len(text_list):
                    break
                # 将词i和词j作为key，出现的次数作为value，添加到共现词典中
                common_mention[(each_char, text_list[j])] += 1
        vocab2idx = {}
        edge_list = []
        for each_common_mention in common_mention:
            source = each_common_mention[0]
            target = each_common_mention[1]

            if source not in vocab2idx:
                vocab2idx[source] = len(vocab2idx)
                source_id = vocab2idx[source]
            else:
                source_id = vocab2idx[source]

            if target not in vocab2idx:
                vocab2idx[target] = len(vocab2idx)
                target_id = vocab2idx[target]
            else:
                target_id = vocab2idx[target]
            edge_list.append((source_id, target_id))
        # print(len(vocab2idx))
        g.add_nodes(len(vocab2idx))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        g.add_edges(dst, src)
        self.graphs.append(g)
        self.labels.append(label)

    def build_graph(self):
        if self.mode == 'train':
            df = pd.read_csv(os.path.join(self.data_dir, 'dev.csv'), sep=',',
                             names=['dialogid', 'history', 'utterance', 'response', 'label'])
        else:
            df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'), sep=',',
                             names=['dialogid', 'history', 'utterance', 'response', 'label'])
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

    @property
    def num_classes(self):
        return 2


if __name__ == '__main__':
    data = DataDOUBAN()
    data.build_graph()


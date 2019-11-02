import os
import math
import random
import torch
import torch.nn.functional as f

import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]
class TextCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pretrained_embeddings):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)

        self.convs = Conv1d(embedding_dim, n_filters, filter_sizes)

        self.fc = Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, utt,resp):
        text, _ = utt
        # text: [sent len, batch size]
        text = text.permute(1, 0)  # 维度换位,
        # text: [batch size, sent len]

        embedded = self.embedding(text)
        # embedded: [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)

        # embedded = [batch size, emb dim, sent len]

        conved = self.convs(embedded)

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n] - 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)
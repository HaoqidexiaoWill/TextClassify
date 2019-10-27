from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers.modeling_bertLSTM import BertPreTrainedModel,BertModel

from .modeling_utils import (WEIGHTS_NAME, CONFIG_NAME, PretrainedConfig, PreTrainedModel,
                             prune_linear_layer, add_start_docstrings)

class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.batch_size = 32

        rnn_hidden_size = 300
        num_layers = 2
        dropout = 0.2

        self.rnn = nn.LSTM(config.hidden_size, rnn_hidden_size, num_layers, bidirectional=True, batch_first=True,
                           dropout=dropout)

        self.classifier = nn.Linear(rnn_hidden_size * 2, config.num_labels)

        # region embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(config.hidden_size, rnn_hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=rnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # conv 模块
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=rnn_hidden_size),
            nn.ReLU(),
            nn.Conv1d(rnn_hidden_size, rnn_hidden_size,kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=rnn_hidden_size),
            nn.ReLU(),
            nn.Conv1d(rnn_hidden_size,rnn_hidden_size,kernel_size=3, padding=1),
        )
        # repeat 模块
        self.max_len = 128
        resnet_block_list = []
        while(self.max_len >2):
            resnet_block_list.append(ResnetBlock(rnn_hidden_size))
            self.max_len = self.max_len//2
        self.resnet_layer = nn.Sequential(*resnet_block_list)

        # 最后的全连接层
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size*self.max_len,self.num_labels),
            nn.BatchNorm1d(self.num_labels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.num_labels,self.num_labels)
        )

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        encoded_layers, _ = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids,
                                      token_type_ids=flat_token_type_ids,
                                      attention_mask=flat_attention_mask, head_mask=head_mask)
        encoded_layers = self.dropout(encoded_layers)

        textA_mask = token_type_ids.transpose(1,2).expand_as(encoded_layers)
        utternace = encoded_layers.mul(textA_mask.float())
        textB_mask = torch.sub(torch.ones_like(encoded_layers).float(), textA_mask.float())
        response = encoded_layers.mul(textB_mask.float())


        utternace_permute = utternace.permute(0,2,1)
        utternace_region_embedding = self.region_embedding(utternace_permute)
        response_permute = response.permute(0, 2, 1)
        response_permute_region_embedding = self.region_embedding(response_permute)


        conv_block = self.conv_block(self.region_embedding)
        resnet_layer = self.resnet_layer(conv_block)
        out = resnet_layer.permute(0,2,1)
        out = out.contiguous().view(-1,300*self.num_labels)
        logits  = self.fc(out)


        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return nn.functional.softmax(logits, -1)



    def GateAttention(self,utt,resp):
        pass





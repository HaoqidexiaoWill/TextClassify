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

# class Conv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, filter_sizes):
#         super(Conv1d, self).__init__()
#         self.convs = nn.ModuleList([
#             nn.Conv1d(in_channels=in_channels,
#                       out_channels=out_channels,
#                       kernel_size=fs)
#             for fs in filter_sizes
#         ])
#
#         self.init_params()
#
#     def init_params(self):
#         for m in self.convs:
#             nn.init.xavier_uniform_(m.weight.data)
#             nn.init.constant_(m.bias.data, 0.1)
#
#     def forward(self, x):
#         return [nn.functional.relu(conv(x)) for conv in self.convs]
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

        n_filters = 200
        filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # self.convs = Conv1d(config.hidden_size, n_filters, filter_sizes)
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
            # nn.BatchNorm1d(self.num_labels),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(self.num_labels,self.num_labels)
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
        '''
        print(encoded_layers.size())
        print(token_type_ids)
        print(token_type_ids.size())
        textA_mask = token_type_ids.transpose(1,2).expand_as(encoded_layers)
        print(textA_mask.size())
        print(textA_mask)
        # print(encoded_layers)
        utternace = encoded_layers.mul(textA_mask.float())
        print(utternace)

        textB_mask = torch.sub(torch.ones_like(encoded_layers).float(), textA_mask.float())  # 减
        print(textA_mask)
        print(textB_mask)
        print(textB_mask.size())
        response = encoded_layers.mul(textB_mask.float())
        '''
        encoded_layers = encoded_layers.permute(0,2,1)
        region_embedding = self.region_embedding(encoded_layers)
        conv_block = self.conv_block(region_embedding)
        resnet_layer = self.resnet_layer(conv_block)
        out = resnet_layer.permute(0,2,1)
        # print(encoded_layers.size())
        out = out.contiguous().view(-1,300*self.num_labels)
        # print(encoded_layers.size(),region_embedding.size(),conv_block.size(),resnet_layer.size(),out.size())
        # print(self.fc)
        # exit()
        logits  = self.fc(out)


        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return nn.functional.softmax(logits, -1)


        '''
        utternace_transpose = utternace.permute(0, 2, 1)
        response_transpose = response.permute(0, 2, 1)
        utternace_conv = self.convs(utternace_transpose)
        response_conv = self.convs(response_transpose)
        '''
        '''
        print(utternace.size(),response.size())
        # torch.Size([4, 128, 768]) torch.Size([4, 128, 768])
        utternace_conv_region = self.conv_region(utternace.unsqueeze(1) )
        response_conv_region = self.conv_region(response.unsqueeze(1) )
        print(utternace_conv_region.size(),response_conv_region.size())
        # [batch_size, filter_num, seq_len-3+1, bert_dim-bert_dim+1=1]
        # torch.Size([4, 200, 126, 1]) torch.Size([4, 200, 126, 1])

        # padding
        utterance_padding_conv = self.padding_conv(utternace_conv_region)
        response_padding_conv = self.padding_conv(response_conv_region)
        print(utternace_conv_region.size(),response_conv_region.size())
        # [batch_size, filter_num, seq_len, 1]
        # torch.Size([4, 200, 126, 1]) torch.Size([4, 200, 126, 1])
        '''
    def GateAttention(self,utt,resp):
        pass




    '''
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
        # encoded_layers: [batch_size, seq_len, bert_dim]

        _, (hidden, cell) = self.rnn(encoded_layers)
        # outputs: [batch_size, seq_len, rnn_hidden_size * 2]
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 连接最后一层的双向输出

        logits = self.classifier(hidden)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return nn.functional.softmax(logits, -1)
    '''
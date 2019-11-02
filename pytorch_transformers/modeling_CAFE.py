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
from torch.nn import functional as F
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

class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        scale = q.size(-1) ** -0.5
        attention = torch.bmm(q, k.transpose(1, 2))* scale
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context
class CAFE(nn.Module):
    def __init__(self, rnn_hidden_size):
        super(CAFE, self).__init__()
        self.channel_size = rnn_hidden_size
        self.hidden_size = 200
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
        self.FeedForward = nn.Sequential(

            # nn.BatchNorm1d(num_features=self.channel_size),
            nn.Linear(self.channel_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.FeedForward2 = nn.Sequential(

            # nn.BatchNorm1d(num_features=self.channel_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.FeedForward3 = nn.Sequential(
            # nn.BatchNorm1d(num_features=self.channel_size),
            nn.Linear(self.channel_size*3, self.channel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.rnn = nn.LSTM(self.hidden_size*9, rnn_hidden_size, num_layers = 2, bidirectional=True, batch_first=True, dropout=0.1)
        self.dot_product_attention = ScaledDotProductAttention()
        self.layer_norm = nn.LayerNorm(self.hidden_size)



    def forward(self,utt,resp):
        # print(utt.size())
        # print(self.FeedForward)
        utt_feedforward = self.FeedForward(utt)
        resp_feedforward = self.FeedForward(resp)

        # resp 对 utt 的attention加权表示
        context_utt_ = self.dot_product_attention(utt_feedforward, resp_feedforward, utt_feedforward)
        context_utt_ = self.FeedForward2(context_utt_)
        context_utt = self.layer_norm(utt_feedforward+context_utt_)

        # utt 对 resp 的attention 加权表示
        context_resp_ = self.dot_product_attention(resp_feedforward, utt_feedforward, resp_feedforward)
        context_resp_ = self.FeedForward2(context_resp_)
        context_resp = self.layer_norm(resp_feedforward+context_resp_)

        # utt 的selfattention
        selfatt_utt_ = self.dot_product_attention(utt_feedforward, utt_feedforward, utt_feedforward)
        selfatt_utt_ = self.FeedForward2(selfatt_utt_)
        selfatt_utt = self.layer_norm(utt_feedforward+selfatt_utt_)

        # resp 的selfattention
        selfatt_resp_ = self.dot_product_attention(resp_feedforward, resp_feedforward, resp_feedforward)
        selfatt_resp_ = self.FeedForward2(selfatt_resp_)
        selfatt_resp = self.layer_norm(resp_feedforward+selfatt_resp_)

        # print(context_resp.size(),
        #       context_resp.size(),
        #       selfatt_resp.size(),
        #       selfatt_utt.size())

        # concat
        utt_concat_context = torch.cat([context_utt,utt_feedforward],dim = -1)
        resp_concat_context = torch.cat([context_resp,resp_feedforward],dim = -1)
        utt_concat_selfatt= torch.cat([selfatt_utt,utt_feedforward],dim = -1)
        resp_concat_selfatt = torch.cat([selfatt_resp,resp_feedforward],dim = -1)

        # diff
        utt_diff_context = context_utt - utt_feedforward
        resp_diff_context = context_resp - resp_feedforward
        utt_diff_selfatt = selfatt_utt-utt_feedforward
        resp_diff_selfatt = selfatt_resp - resp_feedforward

        # dot
        utt_dot_context = context_utt * utt_feedforward
        resp_dot_context = context_resp * resp_feedforward
        utt_dot_selfatt = selfatt_utt * utt_feedforward
        resp_dot_selfatt = selfatt_resp * resp_feedforward

        utt_enhance = torch.cat([
            utt_feedforward,
            utt_concat_context,
            utt_concat_selfatt,
            utt_diff_context,
            utt_diff_selfatt,
            utt_dot_context,
            utt_dot_selfatt
        ],dim = -1)

        resp_enhance = torch.cat([
            resp_feedforward,
            resp_concat_context,
            resp_concat_selfatt,
            resp_diff_context,
            resp_diff_selfatt,
            resp_dot_context,
            resp_dot_selfatt
        ],dim = -1)

        # print(utt_enhance.size(),resp_enhance.size())
        encoder_outputs_utt, _= self.rnn(utt_enhance)
        encoder_outputs_resp, _ = self.rnn(resp_enhance)
        # outputs: [batch_size, seq_len, rnn_hidden_size * 2]
        # print(encoder_outputs_utt.size(),encoder_outputs_resp.size())

        utt = torch.cat((encoder_outputs_utt, utt), dim = -1)
        # x: [batch_size, seq_len, rnn_hidden_size * 2 + bert_dim]
        utt = torch.tanh(self.FeedForward3(utt)).permute(0, 2, 1)

        # y2: [batch_size, rnn_hidden_size * 2, seq_len]
        utt_maxpooling= nn.functional.max_pool1d(utt, utt.size()[2]).squeeze(2)
        resp = torch.cat((encoder_outputs_resp, resp), 2)
        # x: [batch_size, seq_len, rnn_hidden_size * 2 + bert_dim]
        resp = torch.tanh(self.FeedForward3(resp)).permute(0, 2, 1)
        # y2: [batch_size, rnn_hidden_size * 2, seq_len]
        resp_maxpooling= nn.functional.max_pool1d(resp, resp.size()[2]).squeeze(2)

        return utt_maxpooling,resp_maxpooling


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.batch_size = 32

        rnn_hidden_size = 768
        num_layers = 2
        dropout = 0.2

        self.rnn = nn.LSTM(config.hidden_size, rnn_hidden_size, num_layers, bidirectional=True, batch_first=True,
                           dropout=dropout)

        self.cafe = CAFE(rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size * 2, config.num_labels)

        # 最后的全连接层
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size*2,self.num_labels),
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

        # print(token_type_ids.size())
        # exit()
        textA_mask = token_type_ids.transpose(1,2).expand_as(encoded_layers)
        utternace = encoded_layers.mul(textA_mask.float())
        textB_mask = torch.sub(torch.ones_like(encoded_layers).float(), textA_mask.float())
        response = encoded_layers.mul(textB_mask.float())
        utt_maxpooling,resp_maxpool = self.cafe(utternace,response)

        logits  = self.fc(torch.cat([utt_maxpooling,resp_maxpool],dim = -1))


        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return nn.functional.softmax(logits, -1)









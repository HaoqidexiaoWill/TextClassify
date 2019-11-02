from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from pytorch_transformers.modeling_bertLSTM import BertPreTrainedModel,BertModel


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

    def forward(self,
                input_ids_utt, token_type_ids_utt, attention_mask_utt,
                input_ids_resp,token_type_ids_resp ,attention_mask_resp,
                labels=None,position_ids=None, head_mask=None):

        flat_input_ids_utt = input_ids_utt.view(-1, input_ids_utt.size(-1))
        flat_token_type_ids_utt = token_type_ids_utt.view(-1, token_type_ids_utt.size(-1)) if token_type_ids_utt is not None else None
        flat_attention_mask_utt = attention_mask_utt.view(-1, attention_mask_utt.size(-1)) if attention_mask_utt is not None else None

        flat_input_ids_resp = input_ids_resp.view(-1, input_ids_resp.size(-1))
        flat_token_type_ids_resp = token_type_ids_resp.view(-1, token_type_ids_resp.size(-1)) if token_type_ids_resp is not None else None
        flat_attention_mask_resp = attention_mask_resp.view(-1, attention_mask_resp.size(-1)) if attention_mask_resp is not None else None

        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None


        _, pooling_utt = self.bert(
            input_ids=flat_input_ids_utt,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids_utt,
            attention_mask=flat_attention_mask_utt, head_mask=head_mask)

        _, pooling_resp = self.bert(
            input_ids=flat_input_ids_resp,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids_resp,
            attention_mask=flat_attention_mask_resp, head_mask=head_mask)

        # print(pooling_utt.size(),pooling_resp.size())
        # exit()

        logits  = self.fc(torch.cat([pooling_utt,pooling_resp],dim = -1))


        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return nn.functional.softmax(logits, -1)









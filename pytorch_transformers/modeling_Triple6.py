from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from Model.BasicModel import KMaxPooling1D,TripleAttention,ScaledDotProductAttention,TextCNN1D
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

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.batch_size = 32
        self.hidden_size = config.hidden_size

        rnn_hidden_size = config.hidden_size
        num_layers = 2
        dropout = 0.2

        self.rnn = nn.LSTM(config.hidden_size, rnn_hidden_size, num_layers, bidirectional=True, batch_first=True,
                           dropout=dropout)

        self.W2 = nn.Linear(config.hidden_size*2, config.hidden_size)

        # 最后的全连接层
        self.classfier = nn.Sequential(
            nn.Linear(config.hidden_size*6,config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_size,self.num_labels)
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.FeedForward = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.FeedForward2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.triple_attention = TripleAttention(hidden_size=self.hidden_size)

        self.apply(self.init_weights)


    def forward(self,
                input_ids, token_type_ids, attention_mask,
                utterance_mask,response_mask,history_mask,
                labels=None,position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None


        sequence_output, pooling = self.bert(
            input_ids=flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask, head_mask=head_mask)

        history_mask = history_mask.view(-1) == 1
        utterance_mask = utterance_mask.view(-1) == 1
        response_mask = response_mask.view(-1) == 1

        history_mask_output = sequence_output.view(-1,sequence_output.size(2))[history_mask].view(sequence_output.size(0),-1,sequence_output.size(2))
        utterance_mask_output = sequence_output.view(-1, sequence_output.size(2))[utterance_mask].view(sequence_output.size(0),-1,sequence_output.size(2))
        response_mask_output = sequence_output.view(-1, sequence_output.size(2))[response_mask].view(sequence_output.size(0),-1,sequence_output.size(2))

        context_hist,context_resp,context_utt = self.triple_attention(
            history = history_mask_output,
            utterance = utterance_mask_output,
            response = response_mask_output
        )


        for i in range(2):

            hist_rnn , _= self.rnn(context_hist)
            utt_rnn , _= self.rnn(context_utt)
            resp_rnn , _= self.rnn(context_resp)

            hist_rnn = self.FeedForward(hist_rnn)+context_hist
            utt_rnn = self.FeedForward(utt_rnn)+context_utt
            resp_rnn = self.FeedForward(resp_rnn)+context_resp
            context_hist,context_resp,context_utt = self.triple_attention(
                history = hist_rnn,
                utterance = utt_rnn,
                response = resp_rnn
        )


        hist_rnn_cat = torch.cat((history_mask_output, context_hist), 2)
        utt_rnn_cat = torch.cat((utterance_mask_output, context_resp), 2)
        resp_rnn_cat = torch.cat((response_mask_output, context_utt), 2)

        # x: [batch_size, seq_len, rnn_hidden_size * 2 + bert_dim]

        hist_rnn_cat = torch.tanh(self.W2(hist_rnn_cat)).permute(0, 2, 1)
        utt_rnn_cat = torch.tanh(self.W2(utt_rnn_cat)).permute(0, 2, 1)
        resp_rnn_cat = torch.tanh(self.W2(resp_rnn_cat)).permute(0, 2, 1)


        hist_rnn_maxpooling = nn.functional.avg_pool1d(hist_rnn_cat, hist_rnn_cat.size()[2]).squeeze(2)
        utt_rnn_maxpooling = nn.functional.max_pool1d(utt_rnn_cat, utt_rnn_cat.size()[2]).squeeze(2)
        resp_rnn_maxpooling = nn.functional.max_pool1d(resp_rnn_cat, resp_rnn_cat.size()[2]).squeeze(2)

        hist_rnn_avgpooling = nn.functional.avg_pool1d(hist_rnn_cat, hist_rnn_cat.size()[2]).squeeze(2)
        utt_rnn_avgpooling = nn.functional.avg_pool1d(utt_rnn_cat, utt_rnn_cat.size()[2]).squeeze(2)
        resp_rnn_avgpooling = nn.functional.avg_pool1d(resp_rnn_cat, resp_rnn_cat.size()[2]).squeeze(2)

        history_cat = self.dropout(torch.cat((hist_rnn_maxpooling,hist_rnn_avgpooling), dim=1))
        utterance_cat = self.dropout(torch.cat((utt_rnn_maxpooling,utt_rnn_avgpooling), dim=1))
        response_cat = self.dropout(torch.cat((resp_rnn_maxpooling,resp_rnn_avgpooling), dim=1))
        # print(history_cat.size())


        logits  = self.classfier(torch.cat([history_cat,utterance_cat,response_cat],dim = -1))


        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return nn.functional.softmax(logits, -1)









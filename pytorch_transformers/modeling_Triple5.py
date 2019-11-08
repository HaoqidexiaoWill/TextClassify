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

        rnn_hidden_size = 768
        num_layers = 2
        dropout = 0.2

        self.rnn = nn.LSTM(config.hidden_size*3, rnn_hidden_size, num_layers, bidirectional=True, batch_first=True,
                           dropout=dropout)

        self.W2 = nn.Linear(config.hidden_size + 2 * rnn_hidden_size, config.hidden_size)
        # self.classifier = nn.Linear(rnn_hidden_size * 2, config.num_labels)

        n_filters = 200
        filter_sizes = [1,2,3,4,5,6,7,8,9,10]
        self.classifier = nn.Linear(len(filter_sizes) * n_filters*3+config.hidden_size*3+config.hidden_size*3, config.num_labels)
        self.convs = TextCNN1D(config.hidden_size*3, n_filters,filter_sizes)

        # 最后的全连接层
        self.fc = nn.Sequential(
            nn.Linear(len(filter_sizes) * n_filters*3+config.hidden_size*3+config.hidden_size*3,config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_size,self.num_labels)
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.FeedForward2 = nn.Sequential(

            # nn.BatchNorm1d(num_features=self.channel_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.triple_attention = TripleAttention(hidden_size=self.hidden_size)

        self.kmax_pooling = KMaxPooling1D(k = 5)

        self.self_attention = ScaledDotProductAttention()
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
        hist_selfatt = self.self_attention(history_mask_output,history_mask_output,history_mask_output)
        utt_selfatt = self.self_attention(utterance_mask_output,utterance_mask_output,utterance_mask_output)
        resp_selfatt = self.self_attention(response_mask_output,response_mask_output,response_mask_output)

        hist_att_cat = torch.cat((
            history_mask_output,
            context_hist,
            hist_selfatt), 2)

        utt_att_cat = torch.cat((
            utterance_mask_output,
            context_utt,
            utt_selfatt),2)
        resp_att_cat = torch.cat((
            response_mask_output,
            context_resp,
            resp_selfatt),2)


        history_conved =  self.convs(hist_att_cat.permute(0, 2, 1))
        utterance_conved = self.convs(utt_att_cat.permute(0, 2, 1))
        response_conved = self.convs(resp_att_cat.permute(0, 2, 1))

        history_pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2)for conv in history_conved]
        utterance_pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2)for conv in utterance_conved]
        response_pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2)for conv in response_conved]



        hist_rnn , _= self.rnn(hist_att_cat)
        utt_rnn , _= self.rnn(utt_att_cat)
        resp_rnn , _= self.rnn(resp_att_cat)

        # outputs: [batch_size, seq_len, rnn_hidden_size * 2]

        hist_rnn_cat = torch.cat((history_mask_output, hist_rnn), 2)
        utt_rnn_cat = torch.cat((utterance_mask_output, utt_rnn), 2)
        resp_rnn_cat = torch.cat((response_mask_output, resp_rnn), 2)

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


        history_pooled.append(hist_rnn_maxpooling)
        utterance_pooled.append(utt_rnn_maxpooling)
        response_pooled.append(resp_rnn_maxpooling)

        history_pooled.append(hist_rnn_avgpooling)
        utterance_pooled.append(utt_rnn_avgpooling)
        response_pooled.append(resp_rnn_avgpooling)

        history_cat = self.dropout(torch.cat(history_pooled, dim=1))
        utterance_cat = self.dropout(torch.cat(utterance_pooled, dim=1))
        response_cat = self.dropout(torch.cat(response_pooled, dim=1))


        logits  = self.fc(torch.cat([history_cat,utterance_cat,response_cat],dim = -1))


        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return nn.functional.softmax(logits, -1)









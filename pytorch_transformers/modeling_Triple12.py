from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
# from torch.nn import TransformerEncoder

from Model.BasicModel import TripleAttentionHighWay,ScaledDotProductAttention,TextCNN1D,ResnetCNN1D
from pytorch_transformers.modeling_bertLSTM import BertPreTrainedModel,BertModel


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.batch_size = 32
        self.hidden_size = config.hidden_size

        self.num_layers = 2

        n_filters = 200
        filter_sizes = [1,2,3,4,5,6,7,8,9,10]
        self.classifier = nn.Linear(len(filter_sizes) * n_filters*3+config.hidden_size*3+config.hidden_size*3, config.num_labels)
        self.convs = TextCNN1D(config.hidden_size, n_filters,filter_sizes)
        self.resnet = ResnetCNN1D(config.hidden_size)
        # 最后的全连接层
        self.fc = nn.Sequential(
            nn.Linear(len(filter_sizes) * n_filters*3*2,config.hidden_size),
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
        self.triple_attention = TripleAttentionHighWay(hidden_size=self.hidden_size)
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

        resnet_hist = self.resnet(context_hist.permute(0, 2, 1))
        resnet_utt = self.resnet(context_utt.permute(0,2,1))
        resnet_resp = self.resnet(context_resp.permute(0,2,1))

        context_hist,context_resp,context_utt = self.triple_attention(
            history = resnet_hist.permute(0,2,1),
            utterance = resnet_utt.permute(0,2,1),
            response = resnet_resp.permute(0,2,1)
        )

        # resnet_hist = self.resnet(context_hist.permute(0, 2, 1))
        # resnet_utt = self.resnet(context_utt.permute(0,2,1))
        # resnet_resp = self.resnet(context_resp.permute(0,2,1))
        #
        # context_hist,context_resp,context_utt = self.triple_attention(
        #     history = resnet_hist.permute(0,2,1),
        #     utterance = resnet_utt.permute(0,2,1),
        #     response = resnet_resp.permute(0,2,1)
        # )


        history_conved =  self.convs(context_hist.permute(0, 2, 1))
        utterance_conved = self.convs(context_resp.permute(0, 2, 1))
        response_conved = self.convs(context_utt.permute(0, 2, 1))



        # context_hist,context_resp,context_utt = self.triple_attention(
        #     history = history_conved,
        #     utterance = utterance_conved,
        #     response = response_conved
        # )




        history_max_pooling = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2)for conv in history_conved]
        utterance_max_pooling = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2)for conv in utterance_conved]
        response_max_pooling = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2)for conv in response_conved]
        history_mean_pooling = [nn.functional.avg_pool1d(conv, conv.shape[2]).squeeze(2)for conv in history_conved]
        utterance_mean_pooling = [nn.functional.avg_pool1d(conv, conv.shape[2]).squeeze(2)for conv in utterance_conved]
        response_mean_pooling = [nn.functional.avg_pool1d(conv, conv.shape[2]).squeeze(2)for conv in response_conved]


        history_cat_maxpooling = self.dropout(torch.cat(history_max_pooling, dim=1))
        utterance_cat_maxpooling = self.dropout(torch.cat(utterance_max_pooling, dim=1))
        response_cat_maxpooling = self.dropout(torch.cat(response_max_pooling, dim=1))
        history_cat_meanpooling = self.dropout(torch.cat(history_mean_pooling, dim=1))
        utterance_cat_meanpooling = self.dropout(torch.cat(utterance_mean_pooling, dim=1))
        response_cat_meanpooling = self.dropout(torch.cat(response_mean_pooling, dim=1))



        logits  = self.fc(torch.cat([
            history_cat_maxpooling,
            utterance_cat_maxpooling,
            response_cat_maxpooling,
            history_cat_meanpooling,
            utterance_cat_meanpooling,
            response_cat_meanpooling],dim = -1))


        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return nn.functional.softmax(logits, -1)









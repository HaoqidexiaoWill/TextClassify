from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch import nn
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from pytorch_transformers.modeling_bertLSTM import BertPreTrainedModel,BertModel


class BertForTokenClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.domain_num_labels = 32
        self.dependcy_num_labels = 4
        self.domainslot_labels = 26

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier_domain = nn.Linear(config.hidden_size, self.domain_num_labels)
        # self.classifier_dependcy = nn.Linear(config.hidden_size, self.dependcy_num_labels)
        self.valuestart_numlabels = 2
        self.valueend_numlabels = 2
        self.domainslot_numlabels = 3
        self.classifier_tokenstart = nn.Sequential(
            nn.Linear(config.hidden_size,config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_size,self.valuestart_numlabels)
        )
        self.classifier_tokensend = nn.Sequential(
            nn.Linear(config.hidden_size,config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_size,self.valueend_numlabels)
        )
        self.classifier_sentence_domainslot = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_size, self.domainslot_numlabels)
        )
        # self.classifier_tokens_domainslot = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(config.hidden_size, self.classifier_tokens_domainslot)
        # )
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=2)



        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                utterance_mask = None,domain_mask = None,
                slot_mask = None,hist_mask = None,
                label_value_start=None,label_value_end=None,
                label_domainslot = None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        sequence_output, pooling = self.bert(
            input_ids=flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask, head_mask=head_mask)


        sequence_output = self.dropout(sequence_output)

        # print(utterance_mask.size())
        # print(domain_mask.size())
        # print(slot_mask.size())
        # print(hist_mask.size())
        # print(label_value_start.size())
        # print(label_value_end.size())
        # print(label_domainslot.size())
        # exit()

        segment_mask = token_type_ids.view(-1) == 1
        domain_mask = domain_mask.view(-1) == 1
        slot_mask =slot_mask.view(-1)==1
        hist_mask = hist_mask.view(-1) ==1

        utterance_mask_output = sequence_output.view(-1, sequence_output.size(2))[segment_mask].view(
            sequence_output.size(0), -1, sequence_output.size(2))
        domain_mask_output = sequence_output.view(-1, sequence_output.size(2))[domain_mask].view(
            sequence_output.size(0), -1, sequence_output.size(2))
        slot_mask_output = sequence_output.view(-1, sequence_output.size(2))[slot_mask].view(
            sequence_output.size(0), -1, sequence_output.size(2))
        hist_mask_output = sequence_output.view(-1, sequence_output.size(2))[hist_mask].view(
            sequence_output.size(0), -1, sequence_output.size(2))

        # print(utterance_mask_output.size())
        # print(domain_mask_output.size())
        # print(slot_mask_output.size())
        # print(hist_mask_output.size())
        # exit()


        logits_value_start = self.classifier_tokenstart(sequence_output)
        logits_value_end = self.classifier_tokensend(sequence_output)
        logits_domainslot = self.classifier_sentence_domainslot(pooling)


        if label_value_start is not None:
            loss_fct = CrossEntropyLoss()
            loss_domainslot = loss_fct(
                logits_domainslot.view(-1,self.domainslot_numlabels),
                label_domainslot.view(-1))

            active_loss_valuestart = utterance_mask.view(-1) == 1
            # print(logits_value_start.size())
            # exit()
            active_logits_valuestart = logits_value_start.view(-1, self.valuestart_numlabels)[active_loss_valuestart]
            active_labels_tokenstart = label_value_start.view(-1)[active_loss_valuestart]
            # print(active_logits_valuestart.size())
            # print(active_labels_tokenstart.size())
            # exit()
            loss_value_start = loss_fct(active_logits_valuestart, active_labels_tokenstart)

            active_loss_valueend = utterance_mask.view(-1) == 1
            active_logits_valueend = logits_value_end.view(-1, self.valueend_numlabels)[active_loss_valueend]
            active_labels_valueend = label_value_end.view(-1)[active_loss_valueend]
            loss_value_end = loss_fct(active_logits_valueend, active_labels_valueend)
            return loss_value_start,loss_value_end,loss_domainslot
        else:
            return logits_value_start,logits_value_end,logits_domainslot
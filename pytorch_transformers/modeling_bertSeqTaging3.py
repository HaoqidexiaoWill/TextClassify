from __future__ import absolute_import, division, print_function, unicode_literals

from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_transformers.modeling_bertLSTM import BertPreTrainedModel,BertModel


class BertForTokenClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.domain_num_labels = 32
        self.dependcy_num_labels = 4

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier_domain = nn.Linear(config.hidden_size, self.domain_num_labels)
        # self.classifier_dependcy = nn.Linear(config.hidden_size, self.dependcy_num_labels)
        self.classifier_domain = nn.Sequential(
            nn.Linear(config.hidden_size,config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_size,self.domain_num_labels)
        )
        self.classifier_dependcy = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_size, self.dependcy_num_labels)
        )



        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                label_domain=None,label_dependcy = None,
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
        logits_domian = self.classifier_domain(sequence_output)
        logits_dependcy = self.classifier_dependcy(sequence_output)


        if label_domain is not None and label_dependcy is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss_domain = attention_mask.view(-1) == 1
                active_logits_domain = logits_domian.view(-1, self.domain_num_labels)[active_loss_domain]
                active_labels_domain = label_domain.view(-1)[active_loss_domain]
                loss_domain = loss_fct(active_logits_domain, active_labels_domain)

                active_loss_dependcy = attention_mask.view(-1) == 1
                active_logits_dependcy = logits_dependcy.view(-1, self.dependcy_num_labels)[active_loss_dependcy]
                active_labels_dependcy = label_dependcy.view(-1)[active_loss_dependcy]
                loss_dependcy = loss_fct(active_logits_dependcy, active_labels_dependcy)
            else:
                loss_domain = loss_fct(logits_domian.view(-1, self.domain_num_labels), label_domain.view(-1))
                loss_dependcy = loss_fct(logits_dependcy.view(-1, self.dependcy_num_labels), label_dependcy.view(-1))
            return loss_domain,loss_dependcy
        else:
            logits_domain = nn.functional.softmax(logits_domian, -1)
            logits_dependcy = nn.functional.softmax(logits_dependcy, -1)
            return logits_domain,logits_dependcy
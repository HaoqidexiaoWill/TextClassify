"""PyTorch BERT model. """

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

from .modeling_utils import (WEIGHTS_NAME, CONFIG_NAME, PretrainedConfig, PreTrainedModel,
                             prune_linear_layer, add_start_docstrings)
from torch.autograd import Variable
from modeling_bertOrigin import BertModel, BertPreTrainedModel
from STM_model import Conv3DNet, MaskLSTM
logger = logging.getLogger(__name__)


BERT_START_DOCSTRING = r"""    The BERT model was proposed in
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~pytorch_transformers.BertConfig`): Model configuration class with all the parameters of the model. 
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~pytorch_transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``
                
                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``
                
                ``token_type_ids:   0   0   0   0  0     0   0``

            Bert is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`pytorch_transformers.BertTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""   

class STM(nn.Module):
    def __init__(self, args, pretrain_embedding):
        super(STM, self).__init__()

        self.emb_dim = args.emb_dim
        self.mem_dim = args.mem_dim
        self.max_turns_num = args.max_turns_num
        self.max_options_num = args.max_options_num
        self.max_seq_len = args.max_seq_len
        self.rnn_layers = args.rnn_layers

        self.embedding = nn.Embedding(args.vocab_size, args.emb_dim)
        if pretrain_embedding is not None:
            self.embedding.weight.data.copy_(pretrain_embedding)
            self.embedding.weight.requires_grad = True

        self.context_encoder = MaskLSTM(in_dim=args.emb_dim, out_dim=args.mem_dim,batch_first=True,
                                        dropoutP=args.dropoutP)
        self.candidate_encoder = MaskLSTM(in_dim=args.emb_dim, out_dim=args.mem_dim, batch_first=True,
                                        dropoutP=args.dropoutP)

        self.extractor = Conv3DNet(args.rnn_layers+1, 36)
        self.extractor.apply(self.weights_init)


        self.dropout_module = nn.Dropout(args.dropoutP)
        self.criterion = nn.CrossEntropyLoss()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname != 'Conv3DNet':
            m.weight.data.normal_(0.0, 0.02)

    def forward(self, contexts_ids, candidates_ids, labels_ids):
        """

        :param contexts_ids: (batch_size, turns_length, seq_length)
        :param candidates_ids: (batch_size, candidates_set_size, seq_length)
        :param labels_ids:  (batch_size, )
        :return:
        """

        context_seq_len = (contexts_ids != 0).sum(dim=-1).long()
        context_turn_len = (context_seq_len != 0).sum(dim=-1).long()
        candidate_seq_len = (candidates_ids != 0).sum(dim=-1).long()
        candidate_turn_len = (candidate_seq_len != 0).sum(dim=-1).long()

        contexts_emb = self.dropout_module(self.embedding(contexts_ids))
        candidates_emb = self.dropout_module(self.embedding(candidates_ids))

        ###
        context_seq_len_inputs = context_seq_len.view(-1)
        candidate_seq_len_inputs = candidate_seq_len.view(-1)

        all_context_hidden = [contexts_emb]
        all_candidate_hidden = [candidates_emb]
        for layer_id in range(self.rnn_layers):
            contexts_inputs = all_context_hidden[-1].view(-1, self.max_seq_len, self.emb_dim)
            candidates_inputs = all_candidate_hidden[-1].view(-1, self.max_seq_len, self.emb_dim)

            contexts_hidden = self.context_encoder[layer_id](contexts_inputs, context_seq_len_inputs)
            candidates_hidden = self.candidate_encoder[layer_id](candidates_inputs, candidate_seq_len_inputs)

            all_context_hidden.append(contexts_hidden.view(-1, self.max_turns_num, self.max_seq_len, 2*self.mem_dim))
            all_candidate_hidden.append(candidates_hidden.view(-1, self.max_options_num, self.max_seq_len, 2 * self.mem_dim))

        all_context_hidden = torch.stack(all_context_hidden, dim=1)
        all_candidate_hidden = torch.stack(all_candidate_hidden, dim=2)

        spatio_temproal_features = torch.einsum('bltik, boljk->boltij', (all_context_hidden, all_candidate_hidden)) / math.sqrt(300)

        spatio_temproal_features = spatio_temproal_features.contiguous().view(-1, self.rnn_layers+1, self.max_turns_num, self.max_seq_len, self.max_seq_len)

        logits = self.extractor(spatio_temproal_features)

        logits = logits.view(-1, self.max_options_num)

        loss = self.criterion(logits, labels_ids)

        return loss, logits



@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        stm_args = {}
        pretrain_embedding = []
        self.STM = STM(stm_args, pretrain_embedding)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        
        
        _, pooled_output = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        # logits: [batch_size, output_dim=2]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return nn.functional.softmax(logits,-1)


stm_args = {}
pretrain_embedding = []
class BertSTM(BertPreTrainedModel):
    def __init__(self, config, stm_args):
        super(BertSTM, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.STM = STM(stm_args, pretrain_embedding)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) 
        self.apply(self.init_weights)


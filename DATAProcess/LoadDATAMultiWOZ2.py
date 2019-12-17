import json
import re
import pandas as pd

import numpy as np
from Utils.Logger import logger
# from collections import defaultdict
class InputExample(object):

    def __init__(self,guid,text_eachturn=None,text_history=None,turn_belief=None,label_domain=None,label_dependcy=None):
        self.guid = guid
        self.text_eachturn = text_eachturn
        self.text_history = text_history
        self.turn_belief = turn_belief
        self.label_domain = label_domain
        self.label_dependcy = label_dependcy

class InputFeatures(object):
    def __init__(self,example_id,choices_features,labels_domain,labels_dependcy):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids_utt,
                'input_mask': input_mask_utt,
                'segment_ids': segment_ids_utt
            }
            for _,
                input_ids_utt,
                input_mask_utt,
                segment_ids_utt in choices_features
        ]
        self.labels_domain = labels_domain
        self.labels_dependcy = labels_dependcy
class DATAMultiWOZ:
    def __init__(self, debug, data_dir):
        self.debug = debug
        self.data_dir = data_dir
        self.Domain = {}
        self.Denpendcy = {'domain':3,'slot':2,'value':1,'other':0}

    def read_examples(self,input_file):



        examples = []
        with open(input_file, encoding='utf-8') as inf:
            for idx, line_ in enumerate(inf):
                row = json.loads(line_.strip())
                # if idx >= 1000:
                #     break
                examples.append(InputExample(
                    guid=row['DialogTurn_id'],
                    text_eachturn=row['eachturn'],
                    text_history = row['history'],
                    turn_belief = row['turn_belief'],
                    label_domain = row['label_domain'],
                    label_dependcy = row['label_dependcy']
                ))
            print(row['label_dependcy'])
            return examples

    def convert_examples_to_features(self,examples, tokenizer, max_seq_length):
        features = []
        '''
        对每一个例子进行处理
        '''
        for example_index, example in enumerate(examples):

            # eachturn_tokens = tokenizer.tokenize(example.text_eachturn)
            eachturn_tokens = example.text_eachturn.split(' ')
            eachturn_labels_domain = example.label_domain
            eachturn_labels_dependcy = example.label_dependcy

            choices_features = []


            self._truncate_seq_pair(
                eachturn_tokens,
                eachturn_labels_domain,
                eachturn_labels_dependcy,
                max_seq_length//2-2)
            tokens = ["[CLS]"] + eachturn_tokens + ["[SEP]"]
            # print(type(eachturn_labels_domain))
            # print(eachturn_labels_domain)
            labels_domain =   [0] + eachturn_labels_domain + [0]
            labels_dependcy = [0] + eachturn_labels_dependcy + [0]


            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] *len(input_ids)
            input_mask = [1] * len(input_ids)


            padding_length = max_seq_length - len(input_ids)
            input_ids += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_ids += ([0] * padding_length)
            labels_domain += ([0] * padding_length)
            labels_dependcy += ([0] * padding_length)


            assert len(input_ids) == len(input_mask) == len(segment_ids)
            assert len(input_ids) == len(labels_domain) == len(labels_dependcy)
            choices_features.append((tokens, input_ids, input_mask, segment_ids))
            if example_index < 3:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("guid: {}".format(example.guid))
                logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
                logger.info(('turn_belief:{}'.format(example.turn_belief)))
                logger.info("labels_domain: {}".format(labels_domain))
                logger.info("labels_dependcy: {}".format(labels_dependcy))
            features.append(
                InputFeatures(
                    example_id=example.guid,
                    choices_features=choices_features,
                    labels_dependcy=labels_dependcy,
                    labels_domain=labels_domain
                )
            )
        return features

    def _truncate_seq_pair(self,utterance_tokens,eachturn_labels_domain,eachturn_labels_dependcy, max_length):

        while True:
            total_length = len(utterance_tokens)
            if total_length <= max_length:
                break
            if len(utterance_tokens) > len(utterance_tokens):
                eachturn_labels_dependcy.pop()
                eachturn_labels_domain.pop()
                utterance_tokens.pop()

    def select_field(self,features, field):
        return [
            [
                choice[field]
                for choice in feature.choices_features
            ]
            for feature in features
        ]


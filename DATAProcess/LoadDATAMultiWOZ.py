import json
import re
import pandas as pd

import numpy as np
from Utils.Logger import logger
class InputExample(object):

    def __init__(self, guid, text_utt=None,text_resp  = None,label=None):
        self.guid = guid
        self.text_utt = text_utt
        self.text_resp = text_resp
        self.label = label

class InputFeatures(object):
    def __init__(self,example_id,choices_features,label):
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
        self.label = label
class DATAMultiWOZ:
    def __init__(self, debug, data_dir):
        self.debug = debug
        self.data_dir = data_dir
        self.label2id = {}

    def read_examples(self,input_file):
        df = pd.read_csv(input_file, sep='\t',names = ['dialog_turnid','belief','utterance', 'response', 'label'],nrows = 1000)
        print('行数',df.shape[0])
        examples = []
        for index, row in df.iterrows():
            examples.append(InputExample(
                guid=row[0],
                text_utt=row[2],
                text_resp = row[3],
                label=row[4]
            ))
        return examples

    def convert_examples_to_features(self,examples, tokenizer, max_seq_length):
        features = []
        '''
        对每一个例子进行处理
        '''
        for example_index, example in enumerate(examples):

            utterance_tokens = tokenizer.tokenize(example.text_utt)
            response_tokens = tokenizer.tokenize(example.text_resp)

            choices_features = []
            self._truncate_seq_pair(utterance_tokens, response_tokens, max_seq_length-3)
            tokens = ["[CLS]"] + utterance_tokens + ["[SEP]"] + response_tokens + ["[SEP]"]
            segment_ids = [0] * (len(utterance_tokens) + 2) + [1] * (len(response_tokens) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_ids += ([0] * padding_length)

            assert len(input_ids) == len(input_mask) == len(segment_ids) == max_seq_length
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

            # label = example.label
            # if label not in self.label2id:
            #     self.label2id[label] = len(self.label2id)
            #     label = self.label2id[label]
            # else:
            #     label = self.label2id[label]
            label = 0 if example.label[0] == 0 else 1
            if example_index < 3:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("guid: {}".format(example.guid))
                logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
                logger.info("label: {}".format(label))
            features.append(
                InputFeatures(
                    example_id=example.guid,
                    choices_features=choices_features,
                    label=label
                )
            )
        return features

    def _truncate_seq_pair(self,utterance_tokens, response_tokens, max_length):

        while True:
            total_length = len(utterance_tokens) + len(response_tokens)
            if total_length <= max_length:
                break
            if len(utterance_tokens) > len(response_tokens):
                utterance_tokens.pop()
            else:
                response_tokens.pop()

    def select_field(self,features, field):
        return [
            [
                choice[field]
                for choice in feature.choices_features
            ]
            for feature in features
        ]


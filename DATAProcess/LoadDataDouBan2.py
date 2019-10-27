import os
import sys
import json
import re
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from Utils.Logger import logger
class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label
class DATADOUBAN:
    def __init__(self, debug, data_dir):
        self.debug = debug
        self.data_dir = data_dir

    def read_examples(self,input_file):
        df = pd.read_csv(input_file, sep=',',names = ['dialogid', 'utterance', 'response', 'label'])
        print('行数',df.shape[0])
        examples = []
        for index, row in df.iterrows():
            examples.append(InputExample(
                guid=row[0],
                text_a=row[1],
                text_b=row[2],
                label=row[3]
            ))
        return examples
    def read_examples_test(self,input_file):
        df = pd.read_csv(input_file, sep=',',names = ['dialogid', 'utterance', 'response', 'label'])
        print('行数',df.shape[0])
        examples = []
        for index, row in df.iterrows():
            examples.append(InputExample(
                guid=row[0],
                text_a=row[1],
                text_b=row[2],
                label=row[3]
            ))
        return examples
    def convert_examples_to_features(self,examples, tokenizer, max_seq_length):
        features = []
        '''
        对每一个例子进行处理
        '''
        for example_index, example in enumerate(examples):

            context_tokens = tokenizer.tokenize(example.text_a)
            ending_tokens = tokenizer.tokenize(example.text_b)

            choices_features = []
            self._truncate_seq_pair(context_tokens, ending_tokens, max_seq_length - 3)
            # self.truncature(context_tokens, ending_tokens, max_seq_length)
            tokens = ["[CLS]"] + ending_tokens + ["[SEP]"] + context_tokens + ["[SEP]"]
            segment_ids = [0] * (len(ending_tokens) + 2) + [1] * (len(context_tokens) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_ids += ([0] * padding_length)
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

            label = example.label
            if example_index <3:
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

    def truncature(self,tokens_a,tokens_b,max_length):


        half_max_length = 40
        # print(half_max_length)
        # exit()
        print(len(tokens_a))
        print(len(tokens_b))
        # if len(tokens_a) >= half_max_length-2:
        #     tokens_a = tokens_a[:half_max_length-2]
        # # elif len(tokens_a) < half_max_length
        while True:
            if len(tokens_a) <= half_max_length-2:
                break
            else:
                tokens_a.pop()
        padding_length = half_max_length -2
        while True:
            if len(tokens_b) <= half_max_length-2:
                break
            else:
                tokens_b.pop()
        print(len(tokens_a))
        print(len(tokens_b))


        exit()


    def _truncate_seq_pair(self,tokens_a, tokens_b, max_length):

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


    def select_field(self,features, field):
        return [
            [
                choice[field]
                for choice in feature.choices_features
            ]
            for feature in features
        ]
if __name__ == "__main__":
    a = DATADOUBAN(
        debug=False,
        data_dir='/home/lsy2018/TextClassification/DATA/DATA_DOUBAN/data_1024/')

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
    def __init__(self,example_id,choices_features,label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids_utt': input_ids_utt,
                'input_mask_utt': input_mask_utt,
                'segment_ids_utt': segment_ids_utt,
                'input_ids_resp': input_ids_resp,
                'input_mask_resp': input_mask_resp,
                'segment_ids_resp': segment_ids_resp
            }
            for _,
                input_ids_utt,
                input_mask_utt,
                segment_ids_utt,
                _,
                input_ids_resp,
                input_mask_resp,
                segment_ids_resp in choices_features
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
    def convert_examples_to_features(self,examples, tokenizer, max_seq_length):
        features = []
        '''
        对每一个例子进行处理
        '''
        for example_index, example in enumerate(examples):

            context_tokens = tokenizer.tokenize(example.text_a)
            ending_tokens = tokenizer.tokenize(example.text_b)

            choices_features = []
            self.truncature(context_tokens,ending_tokens,max_seq_length)
            tokens_utt = ["[CLS]"] + context_tokens + ["[SEP]"]
            tokens_resp = ["[CLS]"] + ending_tokens + ["[SEP]"]
            segment_id_utt = [1]*(len(tokens_utt))
            input_id_utt = tokenizer.convert_tokens_to_ids(tokens_utt)
            input_mask_utt = [1]*(len(tokens_utt))
            segment_id_resp = [1]*(len(tokens_resp))
            input_id_resp = tokenizer.convert_tokens_to_ids(tokens_resp)
            input_mask_resp = [1]*(len(tokens_resp))

            padding_length_utt = max_seq_length-len(input_id_utt)
            padding_length_resp = max_seq_length-len(input_id_resp)

            input_id_utt += ([0] * padding_length_utt)
            input_mask_utt += ([0] * padding_length_utt)
            segment_id_utt += ([0] * padding_length_utt)
            input_id_resp += ([0] * padding_length_resp)
            input_mask_resp += ([0] * padding_length_resp)
            segment_id_resp += ([0] * padding_length_resp)
            assert len(input_id_utt) == len(input_mask_utt) == len(segment_id_utt) == max_seq_length
            assert len(input_id_resp) == len(input_mask_resp) == len(segment_id_resp) == max_seq_length
            choices_features.append((
                tokens_utt, input_id_utt, input_mask_utt, segment_id_utt,
                tokens_resp, input_id_resp, input_mask_resp, segment_id_resp
            ))
            label  = example.label
            if example_index < 3:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("guid: {}".format(example.guid))
                logger.info("tokensUtt: {}".format(' '.join(tokens_utt).replace('\u2581', '_')))
                logger.info("tokensRESP: {}".format(' '.join(tokens_resp).replace('\u2581', '_')))
                logger.info("label: {}".format(label))
            features.append(
                InputFeatures(
                    example_id=example.guid,
                    choices_features=choices_features,
                    label=label
                )
            )
        return features



    def truncature(self, tokens_a, tokens_b, max_length = 64):
        while True:
            if(len(tokens_a)) <= max_length-2:
                break
            else:
                tokens_a.pop()
        while True:
            if len(tokens_b) <= max_length - 2:
                break
            else:
                tokens_b.pop()
        # print('len(tokens_a)',len(tokens_a))
        # print('len(tokens_b)',len(tokens_b))


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
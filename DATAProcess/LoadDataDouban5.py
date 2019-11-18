import json
import re
import pandas as pd

import numpy as np
from Utils.Logger import logger
class InputExample(object):

    def __init__(self, guid, text_his, text_utt=None,text_resp  = None, label=None):
        self.guid = guid
        self.text_his = text_his
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
                'segment_ids': segment_ids_utt,
                'utterance_mask': utterance_mask,
                'response_mask': response_mask,
                'history_mask': history_mask
            }
            for _,
                input_ids_utt,
                input_mask_utt,
                segment_ids_utt,
                utterance_mask,
                response_mask,
                history_mask in choices_features
        ]
        self.label = label
class DATADOUBAN:
    def __init__(self, debug, data_dir):
        self.debug = debug
        self.data_dir = data_dir

    def read_examples(self,input_file):
        df = pd.read_csv(input_file, sep=',',names = ['dialogid','history', 'utterance', 'response', 'label'],nrows = 1000)
        print('行数',df.shape[0])
        examples = []
        for index, row in df.iterrows():
            examples.append(InputExample(
                guid=row[0],
                text_his=row[1],
                text_utt=row[2],
                text_resp = row[3],
                label=row[4]
            ))
        return examples

    def convert_examples_to_features(self,examples, tokenizer, max_seq_length,max_history_length = 192):
        features = []
        '''
        对每一个例子进行处理
        '''
        for example_index, example in enumerate(examples):

            history_tokens = tokenizer.tokenize(example.text_his)
            utterance_tokens = tokenizer.tokenize(example.text_utt)
            response_tokens = tokenizer.tokenize(example.text_resp)
            choices_features = []
            segment_maxlen = (max_seq_length-4)//2
            self._truncate_seq_pair(
                utterance_tokens,
                response_tokens,
                history_tokens,
                segment_maxlen,
                max_history_length)

            history_inputids = tokenizer.convert_tokens_to_ids(["[CLS]"] + history_tokens + ["[SEP]"])
            utterance_inputids = tokenizer.convert_tokens_to_ids(["[CLS]"] + utterance_tokens + ["[SEP]"])
            response_inputids = tokenizer.convert_tokens_to_ids(["[CLS]"] + response_tokens + ["[SEP]"])

            history_padding = max_history_length-2 - len(history_tokens)
            utterance_padding = segment_maxlen - len(utterance_tokens)
            response_padding = segment_maxlen - len(response_tokens)

            input_ids = utterance_inputids+[0]*utterance_padding+response_inputids+[0]*response_padding+history_inputids+[0]*history_padding
            # print(
            #     len(input_ids),
            #     len(utterance_inputids),
            #     utterance_padding,
            #     len(response_inputids),
            #     response_padding,
            #     len(history_inputids),
            #     history_padding)
            segment_ids = [1]*(segment_maxlen+2)+[0]*(segment_maxlen+2)+[1]*(max_history_length)
            input_mask = [1]*(len(utterance_inputids))+[0]*utterance_padding+[1]*len(response_inputids)+[0]*response_padding+[1]*len(history_inputids)+[0]*history_padding

            history_mask = [0]*(segment_maxlen+2+segment_maxlen+2)+[1]*max_history_length
            utterance_mask = [1]*(segment_maxlen+2)+[0]*(segment_maxlen+2+max_history_length)
            response_mask = [0]*(segment_maxlen+2)+[1]*(segment_maxlen+2)+[0]*max_history_length

            # padding_length = max_seq_length - len(input_ids)
            # input_ids += ([0] * padding_length)
            # input_mask += ([0] * padding_length)
            # segment_ids += ([0] * padding_length)
            # utterance_mask += ([0] * padding_length)
            # response_mask += ([0] * padding_length)
            # history_mask += ([0]*padding_length)

            # print(len(input_ids),len(segment_ids),len(input_mask),len(utterance_mask),len(response_mask),len(history_mask))
            assert len(input_ids) ==len(segment_ids) ==len(input_mask) ==len(utterance_mask) == len(response_mask) == len(history_mask)

            choices_features.append((utterance_tokens+response_tokens, input_ids, input_mask, segment_ids,utterance_mask,response_mask,history_mask))

            label = example.label

            if example_index < 3:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("guid: {}".format(example.guid))
                logger.info("tokens: {}".format(' '.join(utterance_tokens+response_tokens).replace('\u2581', '_')))
                logger.info("label: {}".format(label))
            features.append(
                InputFeatures(
                    example_id=example.guid,
                    choices_features=choices_features,
                    label=label
                )
            )
        return features

    def _truncate_seq_pair(
            self,utterance_tokens,
                response_tokens,
                history_tokens,
                segment_maxlen,
                max_history_length):
        while True:
            if len(utterance_tokens) <= segment_maxlen:
                break
            else:
                utterance_tokens.pop()
        while True:
            if len(response_tokens) <= segment_maxlen:
                break
            else:
                response_tokens.pop()
        while True:
            if len(history_tokens) <= max_history_length-2:
                break
            else:
                history_tokens.pop()
    def select_field(self,features, field):
        return [
            [
                choice[field]
                for choice in feature.choices_features
            ]
            for feature in features
        ]


import json
import re
import pandas as pd

import numpy as np
from Utils.Logger import logger
# from collections import defaultdict
class InputExample(object):

    def __init__(self,guid,text_eachturn=None,text_history=None,
                 turn_belief=None,label_tokens_start=None,label_tokens_end=None,
                 label_sentence_domainslot=None
                 ):
        self.guid = guid
        self.text_eachturn = text_eachturn
        self.text_history = text_history
        self.turn_belief = turn_belief
        self.label_tokens_start = label_tokens_start
        self.label_tokens_end = label_tokens_end
        self.label_sentence_domainslot = label_sentence_domainslot
        # self.label_tokens_domainslot = label_tokens_domainslot

class InputFeatures(object):
    def __init__(self,example_id,hist_token,choices_features,
                 label_tokens_start,label_tokens_end,
                 label_sentence_domainslot
                 ):
        self.example_id = example_id
        self.hist_token = hist_token
        self.choices_features = [
            {
                'input_ids': input_ids_utt,
                'input_mask': input_mask_utt,
                'segment_ids': segment_ids_utt,
                'utterance_mask':utterance_mask,
                'domainslot_mask':domainslot_mask

            }
            for _,
                input_ids_utt,
                input_mask_utt,
                segment_ids_utt,
                utterance_mask,
                domainslot_mask in choices_features
        ]
        self.label_tokens_start = label_tokens_start
        self.label_tokens_end = label_tokens_end
        self.label_sentence_domainslot = label_sentence_domainslot
        # self.label_tokens_domainslot = label_tokens_domainslot
class DATAMultiWOZ:
    def __init__(self, debug, data_dir):
        self.debug = debug
        self.data_dir = data_dir
        self.Domain = {}
        self.Denpendcy = {'domain':3,'slot':2,'value':1,'other':0}
        self.domain_slot_token = [
            'None',
            'restaurant',
            'attraction',
            'hotel',
            'train',
            'taxi',
            'hospital',
            'bus',
            'leaveat',
            'food',
            'area',
            'type',
            'arriveby',
            'internet',
            'destination',
            'name',
            'book time',
            'book people',
            'day',
            'book day',
            'book stay',
            'pricerange',
            'departure',
            'parking',
            'stars',
            'department']

    def read_examples(self,input_file):



        examples = []
        with open(input_file, encoding='utf-8') as inf:
            for idx, line_ in enumerate(inf):
                row = json.loads(line_.strip())
                if idx >= 1000:
                    break
                examples.append(InputExample(
                    guid=row['DialogTurn_id'],
                    text_eachturn=row['eachturn'],
                    text_history = row['history'],
                    turn_belief = row['turn_belief'],
                    label_tokens_start = row['label_tokens_start'],
                    label_tokens_end = row['label_tokens_end'],
                    label_sentence_domainslot = row['label_sentence_domainslot'],
                ))
            return examples

    def convert_examples_to_features(self,examples, tokenizer, max_seq_length):
        features = []
        '''
        对每一个例子进行处理
        '''
        for example_index, example in enumerate(examples):

            # eachturn_tokens = tokenizer.tokenize(example.text_eachturn)
            eachturn_tokens = example.text_history.split(' ')
            eachturn_domainslot_token = self.domain_slot_token
            eachturn_label_tokens_start= example.label_tokens_start
            eachturn_label_tokens_end = example.label_tokens_end
            eachturn_label_sentence_domainslot = example.label_sentence_domainslot
            # eachturn_label_tokens_domainslot = example.label_tokens_domainslot
            # for each in eachturn_label_tokens_domainslot:
            #     print(len(each),each)
            # exit()
            choices_features = []
            total_length = len(eachturn_tokens)
            max_seq_length_ = max_seq_length-3-len(self.domain_slot_token)
            if total_length > max_seq_length_:
                eachturn_tokens = eachturn_tokens[-max_seq_length_:]
                eachturn_label_tokens_start = eachturn_label_tokens_start[-max_seq_length_:]
                eachturn_label_tokens_end = eachturn_label_tokens_end[-max_seq_length_:]
                # eachturn_label_tokens_domainslot = eachturn_label_tokens_domainslot[-max_seq_length_:]
            tokens = ["[CLS]"] + eachturn_tokens + ["[SEP]"]
            eachturn_label_tokens_start = [0] + eachturn_label_tokens_start + [0]
            eachturn_label_tokens_end = [0] + eachturn_label_tokens_end + [0]
            # eachturn_label_tokens_domainslot.insert(0, [0]*len(eachturn_label_sentence_domainslot))
            # eachturn_label_tokens_domainslot.append([0]*len(eachturn_label_sentence_domainslot))
            # print(len(tokens))
            # exit()
            padding_length = max_seq_length-1 - len(self.domain_slot_token) - len(tokens)

            eachturn_label_tokens_start += [0]*padding_length
            eachturn_label_tokens_end += [0]*padding_length
            # for i in range(padding_length):
            #     eachturn_label_tokens_domainslot.append([0] * len(eachturn_label_sentence_domainslot))

            tokens_input_ids = tokenizer.convert_tokens_to_ids(tokens)
            domainslot_input_ids = tokenizer.convert_tokens_to_ids(eachturn_domainslot_token + ["[SEP]"])




            assert  len(eachturn_label_tokens_start) == len(eachturn_label_tokens_end)
            assert  len(self.domain_slot_token) == len(eachturn_label_sentence_domainslot)


            # input_ids = tokenizer.convert_tokens_to_ids(tokens+ eachturn_domainslot_token + ["[SEP]"])
            input_ids = tokens_input_ids + [0] * padding_length + domainslot_input_ids
            segment_ids = [0] *len(tokens_input_ids) + [0] * padding_length + [1]*(len(eachturn_domainslot_token)+1)
            input_mask = [1] * len(tokens_input_ids) + [0] * padding_length + [0]*(len(eachturn_domainslot_token)+1)
            utterance_mask = [1]*(len(tokens_input_ids)+padding_length)+[0]*(len(eachturn_domainslot_token)+1)
            domainslot_mask = [0]*(len(tokens_input_ids)+padding_length) +[1]*(len(eachturn_domainslot_token))+[0]

            assert len(input_ids) == len(input_mask) == len(segment_ids) == max_seq_length
            assert len(input_ids) == len(utterance_mask) == len(domainslot_mask)


            # print(len(input_ids))

            choices_features.append((tokens, input_ids, input_mask, segment_ids,utterance_mask,domainslot_mask))
            if example_index < 3:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("guid: {}".format(example.guid))
                logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
                logger.info("utterance_mask: {}".format(utterance_mask))
                logger.info("domainslot_mask: {}".format(domainslot_mask))
                logger.info(('turn_belief:{}'.format(example.turn_belief)))
                logger.info("eachturn_label_tokens_start: {}".format(eachturn_label_tokens_start))
                logger.info("eachturn_label_tokens_end: {}".format(eachturn_label_tokens_end))
                # logger.info("eachturn_label_tokens_domainslot: {}".format(eachturn_label_tokens_domainslot))
                logger.info("eachturn_label_sentence_domainslot:{}".format(eachturn_label_sentence_domainslot))
            features.append(
                InputFeatures(
                    example_id=example.guid,
                    hist_token = tokens + eachturn_domainslot_token + ["[SEP]"],
                    choices_features=choices_features,
                    label_tokens_start = eachturn_label_tokens_start,
                    label_tokens_end=eachturn_label_tokens_end,
                    # label_tokens_domainslot=eachturn_label_tokens_domainslot,
                    label_sentence_domainslot = eachturn_label_sentence_domainslot,


                )
            )
        return features

    def select_field(self,features, field):
        return [
            [
                choice[field]
                for choice in feature.choices_features
            ]
            for feature in features
        ]


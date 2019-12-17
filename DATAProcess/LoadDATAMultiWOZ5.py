import json
import re
import pandas as pd

import numpy as np
from Utils.Logger import logger
# from collections import defaultdict
class InputExample(object):

    def __init__(self,guid,text_eachturn=None,text_history=None,
                 label_domain=None,label_slot = None,
                 label_value_start=None,label_value_end=None,
                 label_domainslot = None
                 ):
        self.guid = guid
        self.text_eachturn = text_eachturn
        self.text_history = text_history
        self.domain = label_domain
        self.slot =label_slot
        self.label_value_start = label_value_start
        self.label_value_end = label_value_end
        self.label_domainslot = label_domainslot

class InputFeatures(object):
    def __init__(self,example_id,choices_features,
                 label_value_start,label_value_end,
                 label_domainslot
                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids_utt,
                'input_mask': input_mask_utt,
                'segment_ids': segment_ids,
                'utt_mask':utt_mask,
                'domain_mask':domain_mask,
                'slot_mask':slot_mask,
                'hist_mask':hist_mask

            }
            for _,
                input_ids_utt,
                input_mask_utt,
                segment_ids,
                utt_mask,
                domain_mask,
                slot_mask,
                hist_mask in choices_features
        ]
        self.label_value_start = label_value_start
        self.label_value_end = label_value_end
        self.label_domainslot = label_domainslot
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
                if idx >= 1000:
                    break
                examples.append(InputExample(
                    guid=row['DialogTurn_id']+'#'+row['domain']+'#'+row['slot'],
                    text_eachturn=row['eachturn'],
                    text_history = row['history'],
                    label_domain = row['domain'],
                    label_slot = row['slot'],
                    label_value_start = row['label_value_start'],
                    label_value_end = row['label_value_end'],
                    label_domainslot = row['label_domainslot']
                ))
            return examples

    def convert_examples_to_features(self,examples, tokenizer, max_seq_length):
        features = []
        '''
        对每一个例子进行处理
        '''
        for example_index, example in enumerate(examples):

            # eachturn_tokens = tokenizer.tokenize(example.text_eachturn)
            eachturn_histokens = example.text_history.split(' ')
            eachturn_utttokens = example.text_eachturn.split(' ')
            eachturn_domain = example.domain
            eachturn_slot = example.slot
            eachturn_value_start= example.label_value_start
            eachturn_value_end = example.label_value_end
            eachturn_domainslot = example.label_domainslot

            choices_features = []
            # self._truncate_seq_pair(
            #     eachturn_utttokens,
            #     eachturn_histokens,
            #     max_seq_length - 6)
            utt_length = (max_seq_length - 6)//2
            hist_length = (max_seq_length - 6) // 2
            if len(eachturn_utttokens) > utt_length:
                eachturn_utttokens = eachturn_utttokens[-utt_length:]
            utt_tokens = ["[CLS]"] + eachturn_utttokens + ["[SEP]"]
            utt_inputids_ = tokenizer.convert_tokens_to_ids(utt_tokens)
            utt_padding_length = utt_length+2-len(utt_inputids_)
            utt_inputids = utt_inputids_ + [0]*utt_padding_length
            utt_segmentid = [1]*len(utt_inputids)

            domain_slot_tokens = [eachturn_domain] +[eachturn_slot]+ ["[SEP]"]
            domain_slot_inputids = tokenizer.convert_tokens_to_ids(domain_slot_tokens)

            if len(eachturn_histokens) > hist_length:
                eachturn_histokens = eachturn_histokens[-utt_length:]
            hist_tokens = eachturn_histokens + ["[SEP]"]
            hist_inputids = tokenizer.convert_tokens_to_ids(hist_tokens)
            hist_padding_length = hist_length+1-len(hist_inputids)
            hist_inputids += [0]*hist_padding_length
            hist_segmentid = [0]*len(hist_inputids)

            tokens = utt_tokens + domain_slot_tokens + hist_tokens
            input_ids = utt_inputids + domain_slot_inputids + hist_inputids
            segment_ids = utt_segmentid + [0]+[0]+[0]+hist_segmentid
            input_mask = [1] * len(input_ids)

            utt_mask = [1]*len(utt_inputids_)+[0]*(utt_padding_length + len(domain_slot_inputids)+len(hist_inputids))
            domains_mask = [0]*len(utt_inputids)+[1]+[0]+[0]+[0]*len(hist_inputids)
            slot_mask = [0]*len(utt_inputids)+[0]+[1]+[0]+[0]*len(hist_inputids)
            hist_mask = [0]*len(utt_inputids)+[0]+[0]+[0]+[1]*len(hist_inputids)
            label_value_start = [0]+eachturn_value_start+[0]
            label_value_end = [0]+eachturn_value_end+[0]

            padding_length_domainslot = max_seq_length - len(label_value_start)
            label_value_start += ([0] * padding_length_domainslot)
            label_value_end += ([0] * padding_length_domainslot)
            assert len(input_ids)==len(utt_mask)==len(domains_mask)==len(slot_mask)==len(hist_mask)
            assert len(label_value_start) == len(label_value_end) == len(input_ids)




            choices_features.append((
                tokens, input_ids, input_mask, segment_ids,
                utt_mask,domains_mask,slot_mask,hist_mask
            ))


            if example_index < 3:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("guid: {}".format(example.guid))
                logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
                logger.info("label_value_start: {}".format(label_value_start))
                logger.info("label_value_end: {}".format(label_value_end))
                logger.info("eachturn_domainslot: {}".format(eachturn_domainslot))

            features.append(
                InputFeatures(
                    example_id=example.guid,
                    choices_features=choices_features,
                    label_value_start=label_value_start,
                    label_value_end = label_value_end,
                    label_domainslot = eachturn_domainslot
                )
            )
        return features

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            else:
                tokens_b.pop()

    def select_field(self, features, field):
        return [
            [
                choice[field]
                for choice in feature.choices_features
            ]
            for feature in features
        ]

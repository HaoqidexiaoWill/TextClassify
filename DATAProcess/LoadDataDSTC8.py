import json
import os
import csv
import json
import pandas as pd
import numpy as np
class DSTC8Features(object):
    def __init__(self, filename, turn, input_ids, input_mask, segment_ids, label_id):
        self.filename = filename
        self.turn = turn
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DSTC8Data:
    def __init__(self, num_graphs=2):
        super(DSTC8Data, self).__init__()
        self.debug_mode = False
        self.labels = []
        self.data_dir = './datapro/ubuntu/'
        self.data_process_output = './datapro/data_ubuntu/'
        self.process_data()
        # self.load_data()
        # self.conver_lines_to_features()

    def process_data(self):
        for each in ['train', 'dev']:
            if not os.path.exists(self.data_process_output): os.makedirs(self.data_process_output)
            process_jsonfile = os.path.join(self.data_process_output, '{}.json'.format(each))
            if os.path.exists(process_jsonfile): os.remove(process_jsonfile)
            fw = open(process_jsonfile, 'a', encoding='utf-8')
            each_file = os.path.join(self.data_dir, '{}.txt'.format(each))
            df = pd.read_csv(each_file, sep='\t', nrows=1000,names=['dialogid', 'userutt', 'answerid', 'resp', 'label'])
            for val in df[['dialogid', 'userutt', 'answerid', 'resp', 'label']].values:
                dialogid = val[0]
                answerid = val[2]
                user_utterance = val[1]
                sys_resp = val[3]
                label = val[4]
                each_data = {
                    'dialogid':dialogid,
                    'answerid':answerid,
                    'user_utterance':user_utterance,
                    'sys_resp':sys_resp,
                    'label':label
                }
                json.dump(each_data, fw, ensure_ascii=False)
                fw.writelines('\n')
            fw.close()
    def load_data(self, data_path):
        data = []
        with open(data_path, encoding='utf-8') as inf:
            for index, line in enumerate(inf):
                # if index == 50 :break
                each_data = json.loads(line.strip())
                data.append(each_data)
        return data

    def convert_lines_to_features(self, data, max_seq_length, bert_tokenizer):
        features = []
        for i, each_line in enumerate(data):
            # 外部信息
            file_name = each_line['dialogid']
            # 0
            turn = each_line['answerid']
            text_additional = ''
            text_user = each_line['user_utterance']
            text_sys = each_line['sys_resp']
            label = each_line['label']

            tokens_user = bert_tokenizer.tokenize(text_user)
            tokens_sys = bert_tokenizer.tokenize(text_sys)

            tokens = ['[CLS]'] + tokens_user + ['[SEP]']
            segment_ids = [0] * (len(tokens_user) + 2)
            assert len(tokens) == len(segment_ids)
            tokens += tokens_sys + ['[SEP]']
            segment_ids += [1] * (len(tokens_sys) + 1)


            if len(tokens)>max_seq_length:
                tokens = tokens[:max_seq_length-1]+['[SEP]']
                segment_ids = segment_ids[:max_seq_length]

            input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # padding
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                DSTC8Features(
                    filename=file_name,
                    turn=turn,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label
                )
            )
        return features


if __name__ == "__main__":
    a = DSTC8Data()
    a.process_data()
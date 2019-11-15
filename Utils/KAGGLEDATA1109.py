import os
import sys
import json
import re
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

class DATAKAGGLE:
    def __init__(self,data_dir,out_file):
        self.data_dir = data_dir
        self.train_file = os.path.join(self.data_dir,'simplified-nq-train.jsonl')
        self.test_file = os.path.join(self.data_dir,'simplified-nq-test.jsonl')

        self.out_file = out_file
        # assert len(id) == len(utterances) == len(response) == len(label)


        # return data
    
    def process_data(self,filename):
        count = 0
        with open(self.train_file, 'rt') as f:
            example_id,query,long_ans_start,long_ans_end,long_answer,short_ans,label = [],[],[],[],[],[],[]
            lines = f.readlines()
            for eachline in lines:
                count = count + 1 
                print(count)
                data = json.loads(eachline)

                each_document_text = data['document_text'].split(' ')
                each_question  = data['question_text']
                each_example_id = data['example_id']

                long_ans_dict = {}
                for each_anno in data['annotations']:
                    short_ans_text = ''

                    for each_shortans in each_anno['short_answers']:

                        each_short_ans_start = each_shortans['start_token']
                        each_short_ans_end = each_shortans['end_token']
                        each_short_ans_text = ' '.join(each_document_text[each_short_ans_start:each_short_ans_end])
                        short_ans_text += each_short_ans_text
                    each_long_ans_start = each_anno['long_answer']['start_token']
                    each_long_ans_end = each_anno['long_answer']['end_token']
                    long_ans_dict[str(each_long_ans_start)+'-'+str(each_long_ans_end)] = short_ans_text


                for each_candidate in data['long_answer_candidates']:
                    start = each_candidate['start_token']
                    end = each_candidate['end_token']
                    if str(start)+'-'+str(end) in long_ans_dict:
                        long_ans_start.append(start)
                        long_ans_end.append(end)
                        long_answer.append(' '.join(each_document_text[start:end]))
                        label.append('1')
                        short_ans.append(long_ans_dict[str(start)+'-'+str(end)])
                        query.append(each_question)
                        example_id.append(each_example_id)
                    else:
                        long_ans_start.append(start)
                        long_ans_end.append(end)
                        long_answer.append(' '.join(each_document_text[start:end]))
                        label.append('0')
                        short_ans.append('no ans')
                        query.append(each_question)
                        example_id.append(each_example_id)
            
            assert len(example_id) == len(query) ==  len(long_ans_start) == len(long_ans_end) == len(long_answer) == len(short_ans) == len(label)

            data = pd.DataFrame({
                'id':[x for x in range(len(example_id))],
                'example_id': example_id,
                'query': query,
                'long_answer': long_answer,
                'short_ans': short_ans,
                'long_ans_start':long_ans_start,
                'long_ans_end':long_ans_end,
                'label':label
                })

        return data


    def read_data(self):

        train_df = self.process_data(self.train_file)
        # test_data = self.process_data(self.test_file)
        train_data = train_df.sample(frac=0.95, random_state=0, axis=0)
        dev_data = train_df[~train_df.index.isin(train_data.index)]


        print('dev',dev_data.shape)
        # print('test',test_data.shape)
        print('train', train_data.shape)

        if not os.path.exists(self.out_file):os.makedirs(self.out_file)
        train_data.to_csv(os.path.join(self.out_file, "train.csv"), index=False, header=False)
        dev_data.to_csv(os.path.join(self.out_file, "dev.csv"), index=False, header=False)
        # test_data.to_csv(os.path.join(self.out_file, "test.csv"), index=False, header=False)


if __name__ == "__main__":
    a = DATAKAGGLE(
        data_dir='/home/lsy2018/TextClassification/DATA/DATA_KAGGLE/',
        out_file='/home/lsy2018/TextClassification/DATA/DATA_KAGGLE/data_1109_full/')
    a.read_data()

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
# from Utils.Logger import logger

class DATADOUBAN:
    def __init__(self,data_dir,out_file):
        self.data_dir = data_dir
        self.vocab_file = os.path.join(self.data_dir,'vocab.txt')
        self.response_file = os.path.join(self.data_dir,'responses.txt')
        self.train_file = os.path.join(self.data_dir,'train.txt')
        self.dev_file = os.path.join(self.data_dir,'valid.txt')
        self.test_file = os.path.join(self.data_dir,'test.txt')

        self.out_file = out_file
        self.max_utterance_num = 10
        self.response = self.read_response()
    def process_data(self,file):
        id = []
        utterance = []
        response = []
        label = []
        count = 0
        with open(file,'rt') as f:
            print(count)
            count += 1
            for line in f:
                fields = line.strip().split('\t')
                us_id = fields[0]
                context = fields[1]
                # 不加-1会有空字符
                utterances = (context + ' ').split(' _EOS_ ')[:-1]
                utterances = utterances[-self.max_utterance_num:]

                for each_utterance_tokenlist in utterances:
                    each_utterance = ''.join(each_utterance_tokenlist.split(' '))


                    # 负样例
                    if fields[3] != 'NA':
                        negative_id = [id for id in fields[3].split('|')]
                        for each_neg in negative_id:
                            negative_text = self.response[each_neg]
                            id.append(us_id)
                            utterances.append(each_utterance)
                            response.append(negative_text)
                            label.append('0')
                    if fields[2] != 'NA':
                        positive_id = [id for id in fields[2].split('|')]
                        for each_pos in positive_id:
                            positive_text = self.response[each_pos]
                            id.append(us_id)
                            utterances.append(each_utterance)
                            response.append(positive_text)
                            label.append('0')
        assert len(id) == len(utterance) == len(response) == len(label)
        data = pd.DataFrame({
            'id': id,
            'utterance': utterance,
            'response': response,
            'label': label})

        return data

    def process_data_(self,file):
        id = []
        utterances = []
        response = []
        label = []

        df = pd.read_csv(file, sep='\t')
        for index, row in df.iterrows():
            print(index)
            us_id = row[0]
            context = row[1]
            # 不加-1会有空字符
            utterances_ = (context + ' ').split(' _EOS_ ')[:-1]
            utterance_tokens = ''.join(utterances_)
            utterance = ''.join(utterance_tokens.split(' '))


            # 负样例
            try:
                if row[3] != 'NA' or 'nan':
                    negative_id = [id for id in str(row[3]).split('|')]
                    for each_neg in negative_id:
                        if '.' in each_neg:each_neg = each_neg[0]
                        negative_text = self.response[each_neg]
                        id.append(us_id)
                        utterances.append(utterance)
                        response.append(negative_text)
                        label.append('0')
                if row[2] != 'NA' or 'nan':
                    positive_id = [id for id in str(row[2]).split('|')]
                    for each_pos in positive_id:
                        positive_text = self.response[each_pos]
                        id.append(us_id)
                        utterances.append(utterance)
                        response.append(positive_text)
                        label.append('1')
            except:
                continue
        assert len(id) == len(utterances) == len(response) == len(label)
        data = pd.DataFrame({
            'id': id,
            'utterance': utterances,
            'response': response,
            'label': label})

        return data

    def read_response(self):
        response_dict = {}
        with open(self.response_file,'rt') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) != 2: response_dict[fields[0]]  = '_OOV_'
                response_dict[fields[0]] = fields[1]
        print('response 加载完成')
        return response_dict
    def read_data(self):

        dev_data = self.process_data_(self.dev_file)
        test_data = self.process_data_(self.test_file)

        print('dev',dev_data.shape)
        print('test',test_data.shape)
        train_data = self.process_data_(self.train_file)
        print('train', train_data.shape)

        train_data.to_csv(os.path.join(self.out_file, "train.csv"), index=False, header=False)
        dev_data.to_csv(os.path.join(self.out_file, "dev.csv"), index=False, header=False)
        test_data.to_csv(os.path.join(self.out_file, "test.csv"), index=False, header=False)


if __name__ == "__main__":
    a = DATADOUBAN(
        data_dir='/home/lsy2018/TextClassification/DATA/DATA_DOUBAN/Douban_Corpus/',
        out_file='/home/lsy2018/TextClassification/DATA/DATA_DOUBAN/data_1024/')
    a.read_data()

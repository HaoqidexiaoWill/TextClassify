import os
import pandas as pd
import numpy as np

import pandas_profiling
class DATADOUBAN:
    def __init__(self,data_dir,out_file):
        self.data_dir = data_dir
        self.vocab_file = os.path.join(self.data_dir,'vocab.txt')
        self.response_file = os.path.join(self.data_dir,'responses.txt')
        self.train_file = os.path.join(self.data_dir,'train.txt')
        self.dev_file = os.path.join(self.data_dir,'valid.txt')
        self.test_file = os.path.join(self.data_dir,'test.txt')

        self.out_file = out_file
        self.response = self.read_response()
    def process_data(self,file):
        id = []
        history = []
        utterance = []
        response = []
        label = []
        seq_length_history = []
        seq_length_utterance = []
        seq_length_response = []

        df = pd.read_csv(file, sep='\t')
        for index, row in df.iterrows():
            print(index)
            us_id = row[0]
            context = row[1]
            # 不加-1会有空字符
            history_ = (context + ' ').split(' _EOS_ ')[:-1]

            history_tokens = ''.join(history_[:-1])
            utterance_tokens = history_[-1]
            eachturn_history = ''.join(history_tokens.split(' '))
            eachturn_utterance = ''.join(utterance_tokens.split(' '))


            # 正负样例
            if not row[3] or row[3] == 'NA' or row[3] == 'nan' : continue
            negative_id = [id for id in str(row[3]).split('|')]
            for each_neg in negative_id:
                if '.' in each_neg:each_neg = each_neg.split('.')[0]
                elif each_neg == 'nan': continue
                negative_text = self.response[each_neg]
                if len(negative_text) == 0 or len(eachturn_history) == 0 or len(eachturn_utterance) == 0:
                    continue
                id.append(us_id)
                history.append(eachturn_history)
                utterance.append(eachturn_utterance)
                response.append(negative_text)
                label.append('0')

                seq_length_history.append(len(eachturn_history))
                seq_length_utterance.append(len(eachturn_utterance))
                seq_length_response.append(len(negative_text))


            if not row[2] or row[2] == 'NA' or row[2] == 'nan' : continue
            positive_id = [id for id in str(row[2]).split('|')]
            for each_pos in positive_id:
                if each_pos == 'nan':continue
                positive_text = self.response[each_pos]
                if len(positive_text) == 0 or len(eachturn_history) == 0 or len(eachturn_utterance) == 0:
                    continue
                id.append(us_id)
                history.append(eachturn_history)
                utterance.append(eachturn_utterance)
                response.append(positive_text)
                label.append('1')

                seq_length_history.append(len(eachturn_history))
                seq_length_utterance.append(len(eachturn_utterance))
                seq_length_response.append(len(positive_text))

        assert len(id) == len(utterance) == len(response) == len(label) == len(history)
        assert len(id) == len(seq_length_response) == len(seq_length_utterance) == len(seq_length_history)
        data = pd.DataFrame({
            'id': id,
            'history':history,
            'utterance': utterance,
            'response': response,
            'label': label,
            'seq_length_history':seq_length_history,
            'seq_length_utterance':seq_length_utterance,
            'seq_length_response':seq_length_response
        })

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

        dev_data = self.process_data(self.dev_file)
        test_data = self.process_data(self.test_file)

        print('dev',dev_data.shape)
        print('test',test_data.shape)
        train_data = self.process_data(self.train_file)
        print('train', train_data.shape)
        print('his',train_data['seq_length_history'].mean())
        print('utt',train_data['seq_length_utterance'].mean())
        print('resp', train_data['seq_length_response'].mean())

        if not os.path.exists(self.out_file):os.makedirs(self.out_file)
        train_data.to_csv(os.path.join(self.out_file, "train.csv"), index=False, header=False)
        dev_data.to_csv(os.path.join(self.out_file, "dev.csv"), index=False, header=False)
        test_data.to_csv(os.path.join(self.out_file, "test.csv"), index=False, header=False)


if __name__ == "__main__":
    a = DATADOUBAN(
        data_dir='/home/lsy2018/TextClassification/DATA/DATA_DOUBAN/Douban_Corpus/',
        out_file='/home/lsy2018/TextClassification/DATA/DATA_DOUBAN/data_1112/')
    # a.read_data()
    data = pd.read_csv('/home/lsy2018/TextClassification/DATA/DATA_DOUBAN/data_1112/train.csv')
    profile = pandas_profiling.ProfileReport(data)
    profile.to_file("output_file.html")


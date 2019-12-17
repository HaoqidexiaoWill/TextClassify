import os
import re
import numpy as np
import random
import pandas as pd
import torch
from collections import defaultdict


class DataLoader():
    def __init__(self):
        train_path = os.path.join(os.getcwd(), 'origin', 'train.txt')
        test_path = os.path.join(os.getcwd(), 'origin', 'test.txt')
        valid_path = os.path.join(os.getcwd(), 'origin', 'valid.txt')
        responses_path = os.path.join(os.getcwd(), 'origin', 'responses.txt')
        self.train = self.load_data(train_path)
        self.test = self.load_data(test_path)
        self.valid = self.load_data(valid_path)
        self.response = self.load_responses(responses_path)
        

    def load_data(self, file_path, rows=None):
        data = pd.read_csv(file_path, header=None, usecols=[0,1,2,3], nrows=rows, names=['context_id', 'context', 'answer_idx', 'candidate_idx'], sep='\t')
        return data

    def load_responses(self, file_path):
        responses = pd.read_csv(file_path, sep='\t', header=None, index_col=0, names=['response'])
        problem_lines = []
        for index ,row in responses.iterrows():
            row_split = row['response'].split('\n')
            if len(row_split) != 1:
                row = row['response'].split('\n')
                problem_lines.append([index,row.pop(0)])
                problem_lines.extend([line.split('\t') for line in row])
        for line in problem_lines:
            responses.loc[int(line[0])] = line[1]
        return responses

    def look_up(self, dataset):
        new_data = []
        for index ,row in dataset.iterrows():            
            context_id = row['context_id']
            context = row['context']
            context = context.strip().strip('__EOS__')
            context = context.split('__EOS__')
            query = context[-1].strip()
            context = '__EOS__'.join(context[:-1])
            # NoneType process
            if not context:
                context = ' '
            if not query:
                query = ' '
            if str(row['answer_idx'])=='nan':
                answer = []
            else:
                answer = [int(i) for i in str(row['answer_idx']).split('|')] 
            answer = [self.response.loc[i]['response'].strip() for i in answer]  
            if str(row['candidate_idx'])=='nan':
                candidates = []
            else:
                candidates = [int(float(i)) for i in str(row['candidate_idx']).split('|')]
            candidates = [self.response.loc[i]['response'].strip() for i in candidates]

            row_data = []
            for i in answer:
                row_data.append([context_id, context, query, i, 1])
            for i in candidates:
                row_data.append([context_id, context, query, i, 0])
            np.random.shuffle(row_data)
            new_data.extend(row_data)
        new_df = pd.DataFrame(data = new_data, columns=['context_id', 'context', 'query', 'response', 'label'])
        return new_df



if __name__ == "__main__":
    data_loader = DataLoader()
    train = data_loader.look_up(data_loader.train)
    test = data_loader.look_up(data_loader.test)
    valid = data_loader.look_up(data_loader.valid)
    train.to_csv('processed/train.csv', index=None, header=None)
    test.to_csv('processed/test.csv', index=None, header=None)
    valid.to_csv('processed/valid.csv', index=None, header=None)

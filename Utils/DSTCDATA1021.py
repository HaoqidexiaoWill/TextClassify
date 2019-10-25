import os
import sys
import json
import re
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np


def read_data(input_file, output_file):
    train_file = os.path.join(input_file, 'train.txt')
    test_file = os.path.join(input_file, 'dev.txt')

    train_df = pd.read_csv(train_file, sep='\t',names=['dialogid', 'userutt', 'answerid', 'resp', 'label'])
    test_df = pd.read_csv(test_file, sep='\t',names=['dialogid', 'userutt', 'answerid', 'resp', 'label'])


    train_data = process_data(train_df)
    test_data  = process_data(test_df)

    print('traindata:',train_data.shape)
    print('testdata:',test_data.shape)

    if not os.path.exists(output_file) : os.makedirs(output_file)
    train_data.to_csv(os.path.join(output_file, "train.csv"), index=False, header=False)
    test_data.to_csv(os.path.join(output_file, "test.csv"), index=False, header=False)


def process_data(df):
    df.dropna(how='any', inplace=True)
    new_df = df.drop(columns=['answerid'],axis = 1)
    return new_df




if __name__ == "__main__":
    read_data(
        '/home/lsy2018/TextClassification/DATA/DATA_DSTC/datapro/ubuntu/',
        '/home/lsy2018/TextClassification/DATA/DATA_DSTC/data_1021/')

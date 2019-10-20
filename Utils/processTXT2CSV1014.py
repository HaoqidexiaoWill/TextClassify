import pandas as pd
import numpy as np
import os


def process(input_file, mode):
    df = pd.read_csv(os.path.join(input_file,'{}.txt'.format(mode)), sep='\t', nrows=1000, names=['dialogid', 'userutt', 'answerid', 'resp', 'label'])
    df['text'] = df['userutt']+df['resp']
    data = pd.DataFrame(df,columns = ['dialogid','text','label'])
    # df['dialogid','text','label'].to_csv('{}.csv'.format(mode),header=0,index=0) #不保存列名
    data.to_csv('{}.csv'.format(mode), header=0, index=0)  # 不保存列名
for each in ['train','dev']:
    process('/home/lsy2018/文本匹配/datapro/ubuntu',each)
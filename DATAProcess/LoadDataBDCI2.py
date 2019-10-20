import os
import sys
import json
import re
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from Utils.Logger import logger
class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DATABDCI:
    def __init__(self,debug,data_dir,data_process_output):
        self.split_num = 3
        self.debug = debug
        self.data_dir = data_dir
        if not os.path.exists(data_process_output):
            os.makedirs(data_process_output)
        self.data_process_output = data_process_output
        print('数据清洗')
        # self.process1014()
        # self.process1016()
        # self.process1019()

    @staticmethod
    def load_data(input_file,n_splits = 5):
        df = pd.read_csv(input_file, sep=',', names=['id', 'content', 'title', 'label'])
        print('行列数', df.shape)
        train_features = df.drop(['label'], axis = 1)
        train_label = df['label']
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True).split(train_features,train_label))
        data_splitList = []
        for i, (train_idx, valid_idx) in enumerate(splits):

            current_train_data = df.ix[train_idx.astype(int)]
            current_valid_data = df.ix[valid_idx.astype(int)]
            print('训练集和测试集的维度',current_train_data.shape,current_valid_data.shape)
            data_splitList.append((current_train_data,current_valid_data))

        return data_splitList

    def data_clean(self,text='【本报讯】，2018年4月10日这位副厅长每年每次  受贿30万\n后都--__打借条，担任嘉陵区委书记的3.3年间，？；？，，收受67%人民币……,▶全南'):
        text = re.sub(r"\s+", "", text)
        text = text.replace('！', '!')
        text = text.replace('？', '?')
        text = text.replace('，', ',')
        pattern = "[\u4e00-\u9fa5]+|[,。?!]+"  # 中文正则表达式
        regex = re.compile(pattern)  # 生成正则对象
        results = regex.findall(text)
        text = ''.join(results)
        # 删除常见计量单位
        text = re.sub(r'[年月日个十百千万亿]+', '', text)
        # 删除无意义的单词
        stopwords = ['查看更多', '返回搜狐', '责任编辑', '本报讯', '来自客户端', '复制到浏览器打开']
        for each in stopwords:
            text = re.sub(each, '', text)
        if not text: text = '无'
        return text

    def process1019(self):
        # 加载数据
        train_df = pd.read_csv(os.path.join(self.data_dir, "Train_DataSet.csv"), na_values="")
        train_label_df = pd.read_csv(os.path.join(self.data_dir, "Train_DataSet_Label.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "Test_DataSet.csv"))

        # 数据合并
        train_df = train_df.merge(train_label_df, on='id', how='left')
        train_df['label'] = train_df['label'].fillna(-1)
        train_df = train_df[train_df['label'] != -1]
        train_df['label'] = train_df['label'].astype(int)
        test_df['label'] = 0

        # 数据清洗
        test_df['content'] = test_df['content'].fillna('无')
        train_df['content'] = train_df['content'].fillna('无')
        test_df['title'] = test_df['title'].fillna('无')
        train_df['title'] = train_df['title'].fillna('无')
        test_df['label'] = 0

        ##生成数据处理结果
        new_title, new_context, new_label = [], [], []
        for index, row in train_df.iterrows():

            skip_len = len(row['content']) // self.split_num
            if skip_len == 0: continue
            for i in range(self.split_num):
                new_title.append(row['title'])
                new_label.append(row['label'])
                new_context.append(row['content'][int(i * skip_len):int((i + 1) * skip_len)])
        assert len(new_title) == len(new_context) == len(new_label)

        train_data = pd.DataFrame({
            'id': [x for x in range(len(new_title))],
            'title': new_title,
            'content': new_context,
            'label': new_label})
        print('检查训练集是否有空值')
        for col in train_data.columns:
            print("col:", col, "missing:", sum(train_df[col].isnull()))

        train_data.to_csv(os.path.join(self.data_process_output, "train.csv"), index=False, header=False)
        test_df.to_csv(os.path.join(self.data_process_output, "test.csv"), index=False, header=False)


    def process1016(self):
        # 加载数据
        train_df = pd.read_csv(os.path.join(self.data_dir, "Train_DataSet.csv"), na_values="")
        train_label_df = pd.read_csv(os.path.join(self.data_dir, "Train_DataSet_Label.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "Test_DataSet.csv"))

        # 数据合并
        train_df = train_df.merge(train_label_df, on='id', how='left')
        train_df['label'] = train_df['label'].fillna(-1)
        train_df = train_df[train_df['label'] != -1]
        train_df['label'] = train_df['label'].astype(int)
        test_df['label'] = 0

        # 数据清洗
        test_df['content'] = test_df['content'].fillna('无').apply(self.data_clean)
        train_df['content'] = train_df['content'].fillna('无').apply(self.data_clean)
        test_df['title'] = test_df['title'].fillna('无').apply(self.data_clean)
        train_df['title'] = train_df['title'].fillna('无').apply(self.data_clean)
        test_df['label'] = 0

        ##生成数据处理结果
        new_title, new_context, new_label = [], [], []
        for index, row in train_df.iterrows():

            skip_len = len(row['content']) // self.split_num
            if skip_len == 0: continue
            for i in range(self.split_num):
                new_title.append(row['title'])
                new_label.append(row['label'])
                new_context.append(row['content'][int(i * skip_len):int((i + 1) * skip_len)])
        assert len(new_title) == len(new_context) == len(new_label)

        train_data = pd.DataFrame({
            'id': [x for x in range(len(new_title))],
            'title': new_title,
            'content': new_context,
            'label': new_label})
        print('检查训练集是否有空值')
        for col in train_data.columns:
            print("col:", col, "missing:", sum(train_df[col].isnull()))

        train_data.to_csv(os.path.join(self.data_process_output, "train.csv"), index=False, header=False)
        test_df.to_csv(os.path.join(self.data_process_output, "test.csv"), index=False, header=False)

    def process1014(self):
        # 加载数据
        train_df = pd.read_csv(os.path.join(self.data_dir, "Train_DataSet.csv"), na_values="")
        train_label_df = pd.read_csv(os.path.join(self.data_dir, "Train_DataSet_Label.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "Test_DataSet.csv"))

        # 数据合并
        train_df = train_df.merge(train_label_df, on='id', how='left')
        train_df['label'] = train_df['label'].fillna(-1)
        train_df = train_df[train_df['label'] != -1]
        train_df['label'] = train_df['label'].astype(int)
        test_df['label'] = 0

        # 数据清洗
        test_df['content'] = test_df['content'].fillna('无').apply(self.data_clean)
        train_df['content'] = train_df['content'].fillna('无').apply(self.data_clean)
        test_df['title'] = test_df['title'].fillna('无').apply(self.data_clean)
        train_df['title'] = train_df['title'].fillna('无').apply(self.data_clean)

        # self.genenrat_profiling((train_df))

        ##生成数据处理结果
        train_data = train_df.sample(frac=0.8, random_state=0, axis=0)
        dev_data = train_df[~train_df.index.isin(train_data.index)]
        new_title, new_context, new_label = [], [], []
        for index, row in train_data.iterrows():

            skip_len = len(row['content']) // self.split_num
            if skip_len == 0: continue
            for i in range(self.split_num):
                new_title.append(row['title'])
                new_label.append(row['label'])
                new_context.append(row['content'][int(i * skip_len):int((i + 1) * skip_len)])
        assert len(new_title) == len(new_context) == len(new_label)

        train_data = pd.DataFrame({
            'id': [x for x in range(len(new_title))],
            'title': new_title,
            'content': new_context,
            'label': new_label})
        print('检查训练集是否有空值')
        for col in train_data.columns:
            print("col:", col, "missing:", sum(train_df[col].isnull()))

        train_data.to_csv(os.path.join(self.data_process_output, "train.csv"), index=False, header=False)
        dev_data.to_csv(os.path.join(self.data_process_output, "dev.csv"), index=False, header=False)
        test_df.to_csv(os.path.join(self.data_process_output, "test.csv"), index=False, header=False)

    def read_examples(self,input_file, is_training):
        df = pd.read_csv(input_file, sep=',',names = ['id', 'content', 'title', 'label'])
        print('行数',df.shape[0])
        examples = []
        for index, row in df.iterrows():
            examples.append(InputExample(
                guid=row[0],
                text_a=row[1],
                text_b=row[2],
                label=row[3]
            ))
        return examples
    def read_examples1016(self,df):
        print('行数',df.shape[0])
        examples = []
        for index, row in df.iterrows():
            examples.append(InputExample(
                guid=row[0],
                text_a=row[1],
                text_b=row[2],
                label=row[3]
            ))
        return examples


    def convert_examples_to_features(self,examples, tokenizer, max_seq_length):
        features = []
        '''
        对每一个例子进行处理
        '''
        for example_index, example in enumerate(examples):

            context_tokens = tokenizer.tokenize(example.text_a)
            ending_tokens = tokenizer.tokenize(example.text_b)

            choices_features = []
            self._truncate_seq_pair(context_tokens, ending_tokens, max_seq_length - 3)
            tokens = ["[CLS]"] + ending_tokens + ["[SEP]"] + context_tokens + ["[SEP]"]
            segment_ids = [0] * (len(ending_tokens) + 2) + [1] * (len(context_tokens) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_ids += ([0] * padding_length)
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

            label = example.label
            if example_index <3:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("guid: {}".format(example.guid))
                logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
                logger.info("label: {}".format(label))
            features.append(
                InputFeatures(
                    example_id=example.guid,
                    choices_features=choices_features,
                    label=label
                )
            )
        return features


    def _truncate_seq_pair(self,tokens_a, tokens_b, max_length):

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


    def select_field(self,features, field):
        return [
            [
                choice[field]
                for choice in feature.choices_features
            ]
            for feature in features
        ]


if __name__ == "__main__":
    a = DATABDCI(
        debug=False,
        data_dir='/home/lsy2018/文本匹配/DATA/DATA_BDCI/',
        data_process_output='/home/lsy2018/文本匹配/DATA/DATA_BDCI/data_1014/')
    print('开始处理')
    a.process1019()
    # a.data_clean()



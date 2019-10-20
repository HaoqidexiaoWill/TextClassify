from __future__ import absolute_import
import logging
import os
import sys
import random
import time
import pandas as pd
import numpy as np
import gc
from itertools import cycle
from sklearn.metrics import f1_score
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.modeling_bertOrigin import BertForSequenceClassification, BertConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

from args import args

def get_train_logger(log_path):
    logger = logging.getLogger('train-{}'.format(__name__))
    logger.setLevel(logging.INFO)
    #控制台
    handler_stream = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_stream)

    #日志文件
    handler_file = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    return logger

logger = get_train_logger('log.txt')
os.environ["CUDA_VISIBLE_DEVICES"]='0'
class InputExample(object):
    """A single training/test example for simple sequence classification."""

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
class Trainer:
    def __init__(self,args):
        self.valid_step = 1
        self.warmup_steps = 0
        self.adam_epsilon = 1e-8
        self.data_dir = './datapro/ubuntu/'
        self.model_name_or_path = './uncased_L-12_H-768_A-12/'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path, do_lower_case=True)
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.train_steps = 10
        self.device = torch.device('cuda')
        self.debug_mode = False
        self.model_name = 'bert'
        self.seed = 2019
        self.seed_everything()
        self.max_len = 128
        self.epochs = 5
        self.batch_size = 16
        self.num_labels = 2
        self.args = args

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def read_examples_(self,input_file):
        print('file',input_file)
        examples = []
        with open(input_file, 'r') as f:
            lines = f.readlines()[:100]
            for each_line in lines:
                each_line = each_line.split('\t')
                examples.append(InputExample(
                    guid = each_line[0],
                    text_a = each_line[1],
                    text_b = each_line[3],
                    label = int(each_line[4]))
                )
        print(len(examples))
        return examples
    def read_examples(self,input_file):
        df = pd.read_csv(input_file,sep = '\t',nrows =1000,names = ['dialogid', 'userutt', 'answerid','resp', 'label'])
        examples = []
        for val in df[['dialogid', 'userutt', 'answerid','resp', 'label']].values:
            examples.append(InputExample(guid=val[0], text_a=val[1], text_b=val[3], label=val[4]))
        return examples

    def convert_examples_to_features(self,examples, tokenizer, max_seq_length):
        features = []
        for example_index, example in enumerate(examples):
            context_tokens = tokenizer.tokenize(example.text_a)
            if example.text_b == 'nan':example.text_b = 'no content'
            # print(example.text_b)
            ending_tokens = tokenizer.tokenize(example.text_b)

            choices_features = []
            context_tokens_choice = context_tokens
            self._truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            tokens = ["[CLS]"] + ending_tokens + ["[SEP]"] + context_tokens_choice + ["[SEP]"]
            segment_ids = [0] * (len(ending_tokens) + 2) + [1] * (len(context_tokens_choice) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_ids += ([0] * padding_length)
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

            label = example.label

            features.append(
                InputFeatures(
                    example_id=example.guid,
                    choices_features=choices_features,
                    label=label
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
    def _truncate_seq_pair(self,tokens_a, tokens_b, max_length):

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    def accuracy(self,out, labels):

        outputs = np.argmax(out, axis=1)
        # return f1_score(labels, outputs, labels=[0, 1, 2], average='macro')
        return f1_score(labels, outputs, labels=[0, 1], average='macro')
    def create_dataloader(self):
        # 创建batch
        train_examples = self.read_examples(os.path.join(self.data_dir,'train.txt'))
        train_examples_length = len(train_examples)
        train_features= self.convert_examples_to_features(train_examples,self.tokenizer,self.max_len)
        all_input_ids = torch.tensor(self.select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(self.select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(self.select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # train_sampler = DistributedSampler(train_data)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,batch_size=self.batch_size)
        num_train_optimization_steps = self.train_steps

        valid_examples = self.read_examples(os.path.join(self.data_dir, 'dev.txt'))
        valid_examples_length = len(valid_examples)
        valid_features = self.convert_examples_to_features(valid_examples, self.tokenizer, self.max_len)
        all_input_ids = torch.tensor(self.select_field(valid_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(self.select_field(valid_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(self.select_field(valid_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in valid_features], dtype=torch.long)

        valid_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.batch_size)

        return train_dataloader,valid_dataloader,train_examples_length,valid_examples_length, valid_features

    def train(self):
        if self.debug_mode: self.epochs = 2
        print('加载dataloader')
        # train_loader, valid_loader = self.create_dataloader()
        train_dataloader, eval_dataloader, train_examples_length, valid_examples_length, eval_features = self.create_dataloader()
        print('准备模型')
        config = BertConfig.from_pretrained(self.model_name_or_path, num_labels=2)
        model = BertForSequenceClassification.from_pretrained(self.model_name_or_path,self.args, config=config)
        model.to(self.device)
        print('准备优化器')

        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.warmup_steps, t_total=self.train_steps)
        print('开始训练')
        global_step = 0
        best_acc = 0
        train_loss = 0

        num_train_optimization_steps = self.train_steps

        for epoch in range(num_train_optimization_steps):
            nb_tr_examples, nb_tr_steps = 0, 0
            for step,batch in enumerate(train_dataloader):
                print('epoch:', epoch, 'batchIndex:', step)
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,labels=label_ids)


                loss.backward()
                train_loss += loss.item()
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            if(epoch+1)%self.valid_step == 0:
                logger.info("***** Report result *****")
                logger.info("  %s = %s", 'global_step', str(global_step))
                logger.info("  %s = %s", 'train loss', str(train_loss))

                model.eval()
                inference_labels = []
                gold_labels = []
                inference_logits = []
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps = 0
                for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                    input_ids = input_ids.to(self.device)
                    input_mask = input_mask.to(self.device)
                    segment_ids = segment_ids.to(self.device)
                    label_ids = label_ids.to(self.device)

                    with torch.no_grad():
                        tmp_eval_loss = model(input_ids=input_ids, token_type_ids=segment_ids,attention_mask=input_mask, labels=label_ids)
                        logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    inference_labels.append(np.argmax(logits, axis=1))
                    gold_labels.append(label_ids)
                    inference_logits.append(logits)
                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1

                gold_labels = np.concatenate(gold_labels, 0)
                inference_logits = np.concatenate(inference_logits, 0)
                model.train()
                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = self.accuracy(inference_logits, gold_labels)

                result = {'eval_loss': eval_loss,
                          'eval_F1': eval_accuracy,
                          'global_step': global_step,
                          'loss': train_loss}
                if eval_accuracy > best_acc :
                    logger.info("=" * 80)
                    logger.info("Best F1", eval_accuracy)
                    logger.info("Saving Model......")
                    best_acc = eval_accuracy
                    # Save a trained model
                    # model_to_save = model.module if hasattr(model,'module') else model  # Only save the model it-self
                    # output_model_file = os.path.join(self.output_dir, "pytorch_model.bin")
                    # torch.save(model_to_save.state_dict(), self.output_model_file)
                    print("=" * 80)
                else:
                    print("=" * 80)
                model.train()



if __name__ == "__main__":

    # if do_train:
    # data_dir = "/home/lsy2018/DST/data/data_pro/multiwoz/"
    # trainer = Trainer(data_dir, "model_name", epochs=5, batch_size=16, base_batch_size=16, max_len=128,debug_mode=False,num_labels=35,
    #                     learning_rate=5e-5, warmup_proportion=0.1, gradient_accumulation_steps=1, period=50, test_set="dev", load_step=0)
    trainer = Trainer(args)
    time1 = time.time()
    trainer.train()
    print("训练时间: %d min" % int((time.time() - time1) / 60))






from __future__ import absolute_import

import argparse
import logging
import os
import random
import sys
from io import open
import pandas as pd
import numpy as np
import torch
import gc

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

# from tqdm import tqdm, trange

from sklearn.metrics import f1_score

from pytorch_transformers.modeling_bertLSTM import BertForSequenceClassification, BertConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

from itertools import cycle

from Config.argsBDCI import args
from Utils.Logger import logger

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
        self.output_dir = args.output_dir
        self.do_eval = args.do_eval
        self.eval_steps = args.eval_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.warmup_steps = args.warmup_steps
        self.learning_rate = args.learning_rate
        self.adam_epsilon = args.adam_epsilon
        self.train_steps = args.train_steps
        self.data_dir = args.data_dir
        self.per_gpu_eval_batch_size = args.per_gpu_eval_batch_size
        self.per_gpu_train_batch_size = args.per_gpu_train_batch_size
        self.do_lower_case = args.do_lower_case
        self.model_name_or_path = args.model_name_or_path
        self.max_seq_length = args.max_seq_length
        self.seed = args.seed
        self.seed_everything()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def read_examples(self,input_file, is_training):
        df = pd.read_csv(input_file, sep=',',names = ['id', 'content', 'title', 'label'])
        print('行数',df.shape[0])
        print('检查数据中是否有空值')
        # for col in df.columns:
        #     print("col:", col, "missing:", sum(df[col].isnull()))
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
            if example_index < 10:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("guid: {}".format(example.guid))
                logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
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
        """Truncates a sequence pair in place to the maximum length."""

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
        return f1_score(labels, outputs, labels=[0, 1, 2], average='macro')


    def select_field(self,features, field):
        return [
            [
                choice[field]
                for choice in feature.choices_features
            ]
            for feature in features
        ]


    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True


    def train(self):


        try:
            os.makedirs(args.output_dir)
        except:
            pass

        tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path, do_lower_case=self.do_lower_case)
        config = BertConfig.from_pretrained(self.model_name_or_path, num_labels=3)

        # Prepare model
        model = BertForSequenceClassification.from_pretrained(self.model_name_or_path, args, config=config)
        model.to(self.device)

        train_batch_size = self.per_gpu_train_batch_size
        eval_batch_size = self.per_gpu_eval_batch_size
        for i in range(1):

            # Prepare data loader

            train_examples = self.read_examples(os.path.join(self.data_dir, 'train.csv'), is_training=True)
            train_features = self.convert_examples_to_features(
                train_examples, tokenizer, self.max_seq_length)
            all_input_ids = torch.tensor(self.select_field(train_features, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(self.select_field(train_features, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(self.select_field(train_features, 'segment_ids'), dtype=torch.long)
            all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler,batch_size=train_batch_size )

            num_train_optimization_steps = self.train_steps

            # Prepare optimizer

            param_optimizer = list(model.named_parameters())

            # hack to remove pooler, which is not used
            # thus it produce None grad that break apex
            param_optimizer = [n for n in param_optimizer]

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.warmup_steps, t_total=self.train_steps)

            global_step = 0

            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

            best_acc = 0
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            train_dataloader = cycle(train_dataloader)

            for step in range(num_train_optimization_steps):
                batch = next(train_dataloader)
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                tr_loss += loss.item()
                train_loss = round(tr_loss / (nb_tr_steps + 1), 4)

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                loss.backward()

                if (nb_tr_steps + 1) % self.gradient_accumulation_steps == 0:

                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (step + 1) % (self.eval_steps * self.gradient_accumulation_steps) == 0:
                    tr_loss = 0
                    nb_tr_examples, nb_tr_steps = 0, 0
                    logger.info("***** Report result *****")
                    logger.info("  %s = %s", 'global_step', str(global_step))
                    logger.info("  %s = %s", 'train loss', str(train_loss))

                if self.do_eval and (step + 1) % (self.eval_steps * self.gradient_accumulation_steps) == 0:
                    for file in ['dev.csv']:
                        inference_labels = []
                        gold_labels = []
                        inference_logits = []
                        eval_examples = self.read_examples(os.path.join(self.data_dir, file), is_training=True)
                        eval_features = self.convert_examples_to_features(eval_examples, tokenizer, self.max_seq_length)
                        all_input_ids = torch.tensor(self.select_field(eval_features, 'input_ids'), dtype=torch.long)
                        all_input_mask = torch.tensor(self.select_field(eval_features, 'input_mask'), dtype=torch.long)
                        all_segment_ids = torch.tensor(self.select_field(eval_features, 'segment_ids'), dtype=torch.long)
                        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

                        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

                        logger.info("***** Running evaluation *****")
                        logger.info("  Num examples = %d", len(eval_examples))
                        logger.info("  Batch size = %d", eval_batch_size)

                        # Run prediction for full data
                        eval_sampler = SequentialSampler(eval_data)
                        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                        model.eval()
                        eval_loss, eval_accuracy = 0, 0
                        nb_eval_steps, nb_eval_examples = 0, 0
                        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                            input_ids = input_ids.to(self.device)
                            input_mask = input_mask.to(self.device)
                            segment_ids = segment_ids.to(self.device)
                            label_ids = label_ids.to(self.device)

                            with torch.no_grad():
                                tmp_eval_loss = model(input_ids=input_ids, token_type_ids=segment_ids,
                                                      attention_mask=input_mask, labels=label_ids)
                                logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                            logits = logits.detach().cpu().numpy()
                            label_ids = label_ids.to('cpu').numpy()
                            inference_labels.append(np.argmax(logits, axis=1))
                            gold_labels.append(label_ids)
                            inference_logits.append(logits)
                            eval_loss += tmp_eval_loss.mean().item()
                            nb_eval_examples += input_ids.size(0)
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

                        output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
                        with open(output_eval_file, "a") as writer:
                            for key in sorted(result.keys()):
                                logger.info("  %s = %s", key, str(result[key]))
                                writer.write("%s = %s\n" % (key, str(result[key])))
                            writer.write('*' * 80)
                            writer.write('\n')
                        if eval_accuracy > best_acc and 'dev' in file:
                            print("=" * 80)
                            print("Best F1", eval_accuracy)
                            print("Saving Model......")
                            best_acc = eval_accuracy
                            # Save a trained model
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            print("=" * 80)
                        else:
                            print("=" * 80)
        if args.do_test:
            del model
            gc.collect()
            args.do_train = False
            model = BertForSequenceClassification.from_pretrained(os.path.join(args.output_dir, "pytorch_model.bin"), args,
                                                                  config=config)
            if args.fp16:
                model.half()
            model.to(self.device)
            if args.local_rank != -1:
                try:
                    from apex.parallel import DistributedDataParallel as DDP
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

                model = DDP(model)
            elif args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            for file, flag in [('dev.csv', 'dev'), ('test.csv', 'test')]:
            # for file, flag in [ ('test.csv', 'test')]:
                inference_labels = []
                gold_labels = []
                eval_examples = self.read_examples(os.path.join(args.data_dir, file), is_training=False)
                print('exa',len(eval_examples))
                # exit()
                eval_features = self.convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length)
                all_input_ids = torch.tensor(self.select_field(eval_features, 'input_ids'), dtype=torch.long)
                all_input_mask = torch.tensor(self.select_field(eval_features, 'input_mask'), dtype=torch.long)
                all_segment_ids = torch.tensor(self.select_field(eval_features, 'segment_ids'), dtype=torch.long)
                all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                    input_ids = input_ids.to(self.device)
                    input_mask = input_mask.to(self.device)
                    segment_ids = segment_ids.to(self.device)
                    label_ids = label_ids.to(self.device)

                    with torch.no_grad():
                        logits = model(input_ids=input_ids, token_type_ids=segment_ids,
                                       attention_mask=input_mask).detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    inference_labels.append(logits)
                    gold_labels.append(label_ids)
                gold_labels = np.concatenate(gold_labels, 0)
                logits = np.concatenate(inference_labels, 0)
                if flag == 'dev':
                    print(flag,self.accuracy(logits, gold_labels))
                if flag == 'test':
                    df = pd.read_csv(os.path.join(args.data_dir, file),names = ['id', 'content', 'title', 'label'])
                    predict = np.argmax(logits, axis=1).tolist()
                    print(df.shape[0])
                    print(len(predict))
                    df['labelpre'] = predict
                    df[['id','labelpre']].to_csv(os.path.join(args.output_dir, "sub.csv"),index=False,header = False)


if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()

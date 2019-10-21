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
from pytorch_transformers.modeling_bertLSTM import BertForSequenceClassification, BertConfig
# from pytorch_transformers.modeling_bert import BertForSequenceClassification, BertConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

from itertools import cycle

from Config.argsBDCI import args
from Utils.Logger import logger
from DATAProcess.LoadDataCQA import DATACQA
from metric import accuracyCQA,compute_MRR_CQA,compute_5R20
os.environ["CUDA_VISIBLE_DEVICES"]='0'

class Trainer:
    def __init__(self,data_dir,output_dir,num_labels,args):

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_labels = num_labels


        self.weight_decay = args.weight_decay

        self.eval_steps = args.eval_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.warmup_steps = args.warmup_steps
        self.learning_rate = args.learning_rate
        self.adam_epsilon = args.adam_epsilon
        self.train_steps = args.train_steps
        self.per_gpu_eval_batch_size = args.per_gpu_eval_batch_size
        self.train_batch_size = args.per_gpu_train_batch_size
        self.eval_batch_size = self.per_gpu_eval_batch_size
        self.do_lower_case = args.do_lower_case
        self.model_name_or_path = args.model_name_or_path
        self.max_seq_length = args.max_seq_length
        self.seed = args.seed
        self.seed_everything()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path, do_lower_case=self.do_lower_case)

        self.do_test = args.do_test
        self.do_eval = True
        self.args = args

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def create_dataloader(self,examples_):
        data = DATACQA(
            debug = False,
            data_dir= self.data_dir,
        )
        train_examples = data.read_examples(examples_[0])
        train_features = data.convert_examples_to_features(train_examples, self.tokenizer, self.max_seq_length)
        all_input_ids = torch.tensor(data.select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(data.select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(data.select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)

        eval_examples = data.read_examples(examples_[1])
        eval_features = data.convert_examples_to_features(eval_examples, self.tokenizer, self.max_seq_length)
        all_input_ids = torch.tensor(data.select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(data.select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(data.select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)

        return train_dataloader,eval_dataloader,train_examples,eval_examples

    def train(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        data_splitList = DATACQA.load_data(os.path.join(self.data_dir, 'train.csv'),n_splits=5)
        for split_index ,each_data in enumerate(data_splitList):
            # Prepare model
            config = BertConfig.from_pretrained(self.model_name_or_path, num_labels=self.num_labels)
            model = BertForSequenceClassification.from_pretrained(self.model_name_or_path, self.args, config=config)
            model.to(self.device)

            logger.info(f'Fold {split_index + 1}')
            train_dataloader, eval_dataloader, train_examples, eval_examples = self.create_dataloader(each_data)

            num_train_optimization_steps = self.train_steps

            # Prepare optimizer

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

            global_step = 0

            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", self.train_batch_size)
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
                        scores = []
                        questions = [x.text_a for x in eval_examples]

                        logger.info("***** Running evaluation *****")
                        logger.info("  Num examples = %d", len(eval_examples))
                        logger.info("  Batch size = %d", self.eval_batch_size)

                        # Run prediction for full data

                        model.eval()
                        eval_loss, eval_accuracy = 0, 0
                        nb_eval_steps, nb_eval_examples = 0, 0
                        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                            input_ids = input_ids.to(self.device)
                            input_mask = input_mask.to(self.device)
                            segment_ids = segment_ids.to(self.device)
                            label_ids = label_ids.to(self.device)

                            with torch.no_grad():
                                tmp_eval_loss = model(
                                    input_ids=input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=input_mask,
                                    labels=label_ids)
                                logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                            logits = logits.detach().cpu().numpy()
                            label_ids = label_ids.to('cpu').numpy()
                            inference_labels.append(np.argmax(logits, axis=1))
                            scores.append(logits)
                            gold_labels.append(label_ids)
                            inference_logits.append(logits)
                            eval_loss += tmp_eval_loss.mean().item()
                            nb_eval_examples += input_ids.size(0)
                            nb_eval_steps += 1

                        gold_labels = np.concatenate(gold_labels, 0)
                        inference_logits = np.concatenate(inference_logits, 0)
                        scores = np.concatenate(scores, 0)
                        model.train()
                        eval_loss = eval_loss / nb_eval_steps
                        eval_accuracy = accuracyCQA(inference_logits, gold_labels)
                        eval_mrr = compute_MRR_CQA(scores,gold_labels,questions)
                        eval_5R20 = compute_5R20(scores,gold_labels,questions)

                        result = {'eval_loss': eval_loss,
                                  'eval_F1': eval_accuracy,
                                  'eval_MRR':eval_mrr,
                                  'eval_5R20':eval_5R20,
                                  'global_step': global_step,
                                  'loss': train_loss}

                        output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
                        with open(output_eval_file, "a") as writer:
                            for key in sorted(result.keys()):
                                logger.info("  %s = %s", key, str(result[key]))
                                writer.write("%s = %s\n" % (key, str(result[key])))
                            writer.write('*' * 80)
                            writer.write('\n')
                        if eval_accuracy > best_acc :
                            print("=" * 80)
                            print("Best F1", eval_accuracy)
                            print("Saving Model......")
                            best_acc = eval_accuracy
                            # Save a trained model
                            model_to_save = model.module if hasattr(model,'module') else model
                            output_model_file = os.path.join(self.output_dir, "pytorch_model_{}.bin".format(split_index))
                            torch.save(model_to_save.state_dict(), output_model_file)
                            print("=" * 80)
                        else:
                            print("=" * 80)

            del model
            gc.collect()
    def test(self):
        data = DATACQA(
            debug=False,
            data_dir=self.data_dir
        )
        test_examples = data.read_examples(os.path.join(self.data_dir, 'test.csv'))
        print('eval_examples的数量', len(test_examples))
        prediction = np.zeros((len(test_examples),3))
        gold_labels_  = np.zeros((len(test_examples),3))
        logits_ = np.zeros((len(test_examples),3))
        questions = [x.text_a for x in test_examples]
        test_features = data.convert_examples_to_features(test_examples, self.tokenizer, self.max_seq_length)
        all_input_ids = torch.tensor(data.select_field(test_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(data.select_field(test_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(data.select_field(test_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in test_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)

        for i in range(5):

            config = BertConfig.from_pretrained(self.model_name_or_path, num_labels=self.num_labels)
            model = BertForSequenceClassification.from_pretrained(
                os.path.join(self.output_dir, "pytorch_model_{}.bin".format(i)), self.args, config=config)
            model.to(self.device)
            model.eval()

            inference_labels = []
            gold_labels = []

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
            gold_labels_ = gold_labels
            logits = np.concatenate(inference_labels, 0)
            print(logits.shape)
            print(prediction.shape)

            prediction += logits/5



        test_id = [x.guid for x in test_examples]
        assert len(test_id) == len(prediction)
        # print(accuracyCQA(prediction, gold_labels_))
        # print(compute_MRR_CQA(questions))
        logits_ = np.argmax(prediction, axis=1)

        submission = pd.DataFrame({
            'id':test_id,
            'predict':logits_
        })
        submission.to_csv(os.path.join(self.output_dir, "sub.csv"),index=False,header = False)


if __name__ == "__main__":
    trainer = Trainer(
        data_dir = '/home/lsy2018/TextClassification/DATA/DATA_CQA/data_1020/',
        output_dir = './model_BertLSTM_CQA',
        num_labels= 3,
        args = args)
    trainer.train()
    trainer.test()

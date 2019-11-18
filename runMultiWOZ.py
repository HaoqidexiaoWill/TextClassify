from __future__ import absolute_import

import os
import random
import time
from io import open
import numpy as np
import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from pytorch_transformers.modeling_bert import BertConfig,BertForSequenceClassification
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

from itertools import cycle

from Config.argsMultiWOZ import args
from Utils.Logger import logger
from DATAProcess.LoadDATAMultiWOZ import DATAMultiWOZ
from Metric.ComputeMultiWOZ import accuracyF1
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
        self.model_name_or_path = '/home/lsy2018/TextClassification/PreTraining/uncased_L-12_H-768_A-12/'
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
    def create_dataloader(self):
        data = DATAMultiWOZ(
            debug = False,
            data_dir= self.data_dir,
        )
        train_examples = data.read_examples(os.path.join(self.data_dir,'train.tsv'))
        train_features = data.convert_examples_to_features(train_examples, self.tokenizer, self.max_seq_length)
        all_input_ids = torch.tensor(data.select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(data.select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(data.select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)

        eval_examples = data.read_examples(os.path.join(self.data_dir, 'test.tsv'))
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


        # logger.info(f'Fold {split_index + 1}')
        train_dataloader, eval_dataloader, train_examples, eval_examples = self.create_dataloader()

        num_train_optimization_steps = self.train_steps

        # Prepare model
        config = BertConfig.from_pretrained(self.model_name_or_path, num_labels=self.num_labels)
        model = BertForSequenceClassification.from_pretrained(self.model_name_or_path,self.args, config=config)
        model.to(self.device)
        model.train()
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
        best_MRR = 0
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_dataloader = cycle(train_dataloader)

        for step in range(num_train_optimization_steps):
            batch = next(train_dataloader)
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=label_ids)
            tr_loss += loss.item()
            train_loss = round(tr_loss / (nb_tr_steps + 1), 4)

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            loss.backward()
            if (nb_tr_steps + 1) % self.gradient_accumulation_steps == 0:

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
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
                    ID = [x.guid for x in eval_examples]

                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", self.eval_batch_size)

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    for input_ids, input_mask, segment_ids,label_ids in eval_dataloader:
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
                            logits = model(
                                input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask
                            )

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
                    eval_accuracy = accuracyF1(inference_logits, gold_labels)
                    print(
                        'eval_F1',eval_accuracy,
                        'global_step',global_step,
                        'loss',train_loss
                    )
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
                    if eval_accuracy > best_acc :
                        print("=" * 80)
                        print("Best F1", eval_accuracy)
                        print("Saving Model......")
                        # best_acc = eval_accuracy
                        best_acc = eval_accuracy
                        # Save a trained model
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_model_file = os.path.join(self.output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("=" * 80)
                    else:
                        print("=" * 80)

    def test_eval(self):
        data = DATAMultiWOZ(
            debug=False,
            data_dir=self.data_dir
        )
        test_examples = data.read_examples(os.path.join(self.data_dir, 'test.tsv'))
        print('eval_examples的数量', len(test_examples))

        ID = [x.guid for x in test_examples]

        test_features = data.convert_examples_to_features(test_examples, self.tokenizer, self.max_seq_length)
        all_input_ids = torch.tensor(data.select_field(test_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(data.select_field(test_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(data.select_field(test_features, 'segment_ids'), dtype=torch.long)
        all_utterance_mask = torch.tensor(data.select_field(test_features, 'utterance_mask'), dtype=torch.long)
        all_response_mask = torch.tensor(data.select_field(test_features, 'response_mask'), dtype=torch.long)
        all_history_mask = torch.tensor(data.select_field(test_features, 'history_mask'), dtype=torch.long)

        all_label = torch.tensor([f.label for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_utterance_mask,all_response_mask,all_history_mask, all_label)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.eval_batch_size)



        config = BertConfig.from_pretrained(self.model_name_or_path, num_labels=self.num_labels)
        model = BertForSequenceClassification.from_pretrained(
            os.path.join(self.output_dir, "pytorch_model.bin"), self.args, config=config)
        model.to(self.device)
        model.eval()

        inference_labels = []
        gold_labels = []
        scores = []

        for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)

            with torch.no_grad():
                logits = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                ).detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            scores.append(logits)
            inference_labels.append(np.argmax(logits, axis=1))
            gold_labels.append(label_ids)
        gold_labels = np.concatenate(gold_labels, 0)
        scores = np.concatenate(scores, 0)
        logits = np.concatenate(inference_labels, 0)

        # 计算评价指标
        assert  len(ID) == scores.shape[0]== scores.shape[0]
        eval_accuracy = accuracyF1(logits, gold_labels)

        # eval_DOUBAN_MRR,eval_DOUBAN_mrr,eval_DOUBAN_MAP,eval_Precision1 = compute_DOUBAN(ID,scores,gold_labels)
        # print(
        #     'eval_MRR',eval_DOUBAN_MRR,eval_DOUBAN_mrr,
        #     'eval_MAP',eval_DOUBAN_MAP,
        #     'eval_Precision1',eval_Precision1)
        print('F1',eval_accuracy)


if __name__ == "__main__":

    trainer = Trainer(
        data_dir = '/home/lsy2018/graphDialog/data/data/',
        output_dir = './model_MultiWOZ_1',
        # DOUBAN 是二分类
        num_labels= 2,
        args = args)
    trainer.train()
    time_start = time.time()
    # trainer.test_eval()
    print('1000条测试运行时间',time.time()-time_start,'s')
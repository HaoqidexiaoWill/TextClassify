from __future__ import absolute_import

import os
import random
import time
from io import open
import numpy as np
import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from pytorch_transformers.modeling_bertSeqMatrix4 import BertForTokenClassification
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

from itertools import cycle

from Config.argsMultiWOZ import args
from Utils.Logger import logger
from DATAProcess.LoadDATAMultiWOZ5 import DATAMultiWOZ
from Metric.ComputeMultiWOZ import accuracyF1,compute_jointGoal_domainslot
os.environ["CUDA_VISIBLE_DEVICES"]='0'
class Trainer:
    def __init__(self,data_dir,output_dir,num_labels_value,num_labels_domainslot,args):

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_labels_value= num_labels_value
        self.num_labels_domainslot = num_labels_domainslot


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
        train_examples = data.read_examples(os.path.join(self.data_dir,'train.json'))
        train_features = data.convert_examples_to_features(train_examples, self.tokenizer, self.max_seq_length)
        all_input_ids = torch.tensor(data.select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(data.select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(data.select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_utterance_mask = torch.tensor(data.select_field(train_features, 'utt_mask'), dtype=torch.long)
        all_domain_mask = torch.tensor(data.select_field(train_features, 'domain_mask'), dtype=torch.long)
        all_slot_mask = torch.tensor(data.select_field(train_features, 'slot_mask'), dtype=torch.long)
        all_hist_mask = torch.tensor(data.select_field(train_features, 'hist_mask'), dtype=torch.long)
        all_label_value_start = torch.tensor([f.label_value_start for f in train_features], dtype=torch.long)
        all_label_value_end = torch.tensor([f.label_value_end for f in train_features], dtype=torch.long)
        all_label_domainslot = torch.tensor([f.label_domainslot for f in train_features], dtype=torch.long)
        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_utterance_mask,
            all_domain_mask,
            all_slot_mask,
            all_hist_mask,
            all_label_value_start,
            all_label_value_end,
            all_label_domainslot
        )

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)

        eval_examples = data.read_examples(os.path.join(self.data_dir, 'test.json'))
        eval_features = data.convert_examples_to_features(eval_examples, self.tokenizer, self.max_seq_length)
        eval_input_ids = torch.tensor(data.select_field(eval_features, 'input_ids'), dtype=torch.long)
        eval_input_mask = torch.tensor(data.select_field(eval_features, 'input_mask'), dtype=torch.long)
        eval_segment_ids = torch.tensor(data.select_field(eval_features, 'segment_ids'), dtype=torch.long)
        eval_utterance_mask = torch.tensor(data.select_field(eval_features, 'utt_mask'), dtype=torch.long)
        eval_domain_mask = torch.tensor(data.select_field(eval_features, 'domain_mask'), dtype=torch.long)
        eval_slot_mask = torch.tensor(data.select_field(eval_features, 'slot_mask'), dtype=torch.long)
        eval_hist_mask = torch.tensor(data.select_field(eval_features, 'hist_mask'), dtype=torch.long)
        eval_label_value_start = torch.tensor([f.label_value_start for f in eval_features], dtype=torch.long)
        eval_label_value_end = torch.tensor([f.label_value_end for f in eval_features], dtype=torch.long)
        eval_label_domainslot = torch.tensor([f.label_domainslot for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(
            eval_input_ids,
            eval_input_mask,
            eval_segment_ids,
            eval_utterance_mask,
            eval_domain_mask,
            eval_slot_mask,
            eval_hist_mask,
            eval_label_value_start,
            eval_label_value_end,
            eval_label_domainslot

        )
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
        config = BertConfig.from_pretrained(self.model_name_or_path)
        model = BertForTokenClassification.from_pretrained(self.model_name_or_path,self.args, config=config)
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
            input_ids,input_mask,segment_ids,\
            utterance_mask,domain_mask, \
            slot_mask,hist_mask,\
            label_value_start,label_value_end,\
            label_domainslot = batch

            loss_tokenstart,loss_tokenend,loss_domainslot = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                utterance_mask = utterance_mask,
                domain_mask = domain_mask,
                slot_mask = slot_mask,
                hist_mask = hist_mask,
                label_value_start=label_value_start,
                label_value_end = label_value_end,
                label_domainslot = label_domainslot
            )
            loss = loss_tokenstart + loss_tokenend + loss_domainslot
            # loss = loss_domainslot
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
                for file in ['de.csv']:
                    gold_value_start = []
                    gold_value_end = []
                    gold_domainslot = []
                    scores_value_start = []
                    scores_value_end = []
                    scores_domainslot = []
                    dialogueID = [x.guid for x in eval_examples]
                    utterance_text = [x.text_eachturn for x in eval_examples]
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", self.eval_batch_size)

                    model.eval()
                    eval_loss_tokens_start,eval_loss_tokens_end,eval_loss_domainslot = 0,0,0
                    eval_F1_tokens_start,eval_F1_tokens_end = 0,0
                    eval_F1_sentence_domainslot,eval_F1_tokens_domainslot = 0,0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    for input_ids,input_mask, segment_ids,\
                        utterance_mask,domain_mask, \
                        slot_mask,hist_mask,\
                        label_value_start,label_value_end,\
                        label_domainslot in eval_dataloader:
                        input_ids = input_ids.to(self.device)
                        input_mask = input_mask.to(self.device)
                        segment_ids = segment_ids.to(self.device)
                        utterance_mask = utterance_mask.to(self.device)
                        domain_mask = domain_mask.to(self.device)
                        slot_mask = slot_mask.to(self.device)
                        hist_mask = hist_mask.to(self.device)
                        label_value_start = label_value_start.to(self.device)
                        label_value_end = label_value_end.to(self.device)
                        label_domainslot = label_domainslot.to(self.device)


                        with torch.no_grad():
                            batch_eval_loss_value_start,batch_eval_loss_value_end,batch_eval_loss_domainslot = model(
                                input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                utterance_mask = utterance_mask,
                                domain_mask = domain_mask,
                                slot_mask = slot_mask,
                                hist_mask = hist_mask,
                                label_value_start = label_value_start,
                                label_value_end=label_value_end,
                                label_domainslot=label_domainslot
                            )
                            logits_value_start,logits_value_end,logits_domainslot = model(
                                input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                utterance_mask = utterance_mask,
                                domain_mask = domain_mask,
                                slot_mask = slot_mask,
                                hist_mask = hist_mask,
                            )
                        logits_value_start = logits_value_start.cpu().numpy()
                        logits_value_end = logits_value_end.cpu().numpy()
                        logits_domainslot = logits_domainslot.cpu().numpy()

                        label_value_start = label_value_start.to('cpu').numpy()
                        label_value_end = label_value_end.to('cpu').numpy()
                        label_domainslot = label_domainslot.to('cpu').numpy()

                        scores_value_start.append(logits_value_start)
                        scores_value_end.append(logits_value_end)
                        scores_domainslot.append(logits_domainslot)

                        gold_value_start.append(label_value_start)
                        gold_value_end.append(label_value_end)
                        gold_domainslot.append(label_domainslot)

                        eval_loss_tokens_start += batch_eval_loss_value_start.mean().item()
                        eval_loss_tokens_end += batch_eval_loss_value_end.mean().item()
                        eval_loss_domainslot += batch_eval_loss_domainslot.mean().item()

                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    gold_value_start = np.concatenate(gold_value_start,0)
                    gold_value_end = np.concatenate(gold_value_end,0)
                    gold_domainslot = np.concatenate(gold_domainslot,0)

                    scores_value_start = np.concatenate(scores_value_start, 0)
                    scores_value_end = np.concatenate(scores_value_end, 0)
                    scores_domainslot = np.concatenate(scores_domainslot,0)

                    model.train()
                    eval_loss_tokens_start = eval_loss_tokens_start/nb_eval_steps
                    eval_loss_tokens_end = eval_loss_tokens_end / nb_eval_steps
                    eval_loss_domainslot = eval_loss_domainslot /nb_eval_steps

                    # print(scores_domainslot.shape)
                    # print(gold_labels_domainslot.shape)
                    # print(scores_domainslot)
                    # print(gold_labels_domainslot)
                    # exit()
                    # eval_accuracy_token_start = accuracyF1(scores_domain, gold_labels_domain,mode='domain')
                    # eval_accuracy_token_end = accuracyF1(scores_dependcy, gold_labels_dependcy ,mode= 'dependcy')

                    eval_F1_valuestart,eval_F1_valueend,F1_domainslot = compute_jointGoal_domainslot(
                        dialogueID,
                        utterance_text,
                        scores_value_start,
                        scores_value_end,
                        scores_domainslot,
                        gold_value_start,
                        gold_value_end,
                        gold_domainslot
                    )


                    print(
                        'F1_domainslot',F1_domainslot,
                        'eval_F1_valuestart',eval_F1_valuestart,
                        'eval_F1_valueend', eval_F1_valueend,
                        'global_step',global_step,
                        'loss',train_loss
                    )
                    result = {

                        'eval_loss_tokens_start':eval_loss_tokens_start,
                        'eval_loss_tokens_end': eval_loss_tokens_end,
                        'eval_loss_domainslot':eval_loss_domainslot,

                        'F1_domainslot': F1_domainslot,
                        'eval_F1_valuestart': eval_F1_valuestart,
                        'eval_F1_valueend': eval_F1_valueend,
                        'global_step': global_step,
                        'loss': train_loss}

                    output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
                        writer.write('*' * 80)
                        writer.write('\n')
                    if eval_F1_valuestart > best_acc :
                        print("=" * 80)
                        print("Best jointGoal", eval_F1_valuestart)
                        print("Saving Model......")
                        # best_acc = eval_accuracy
                        best_acc = eval_F1_valuestart
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
        test_examples = data.read_examples(os.path.join(self.data_dir, 'test.json'))
        print('eval_examples的数量', len(test_examples))

        dialogueID = [x.guid for x in test_examples]
        utterance_text = [x.text_history for x in test_examples]

        test_features = data.convert_examples_to_features(test_examples, self.tokenizer, self.max_seq_length)
        all_input_ids = torch.tensor(data.select_field(test_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(data.select_field(test_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(data.select_field(test_features, 'segment_ids'), dtype=torch.long)
        eval_labels_domainslot = torch.tensor([f.labels_domainslot for f in test_features], dtype=torch.float)
        eval_labels_domain = torch.tensor([f.labels_domain for f in test_features], dtype=torch.long)
        eval_labels_dependcy = torch.tensor([f.labels_dependcy for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,eval_labels_domainslot,eval_labels_domain,eval_labels_dependcy)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.eval_batch_size)



        config = BertConfig.from_pretrained(self.model_name_or_path)
        model = BertForTokenClassification.from_pretrained(
            os.path.join(self.output_dir, "pytorch_model.bin"), self.args, config=config)
        model.to(self.device)
        model.eval()

        inference_labels = []
        gold_labels_domain = []
        gold_labels_dependcy = []
        gold_labels_domainslot = []
        scores_domainslot = []
        scores_domain = []
        scores_dependcy = []

        for input_ids, input_mask, segment_ids,label_domainslot,label_domain,label_dependcy in test_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_domainslot = label_domainslot.to(self.device)
            label_domain = label_domain.to(self.device)
            label_dependcy = label_dependcy.to(self.device)

            with torch.no_grad():
                logits_domainslot,logits_domain,logits_dependcy = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask
                )
            logits_domainslot = torch.sigmoid(logits_domainslot)
            logits_domainslot = (logits_domainslot > 0.4).float()
            logits_domainslot = logits_domainslot.cpu().long().numpy()


            logits_domain = logits_domain.view(-1, self.num_labels_domain).cpu().numpy()
            logits_dependcy = logits_dependcy.view(-1, self.num_labels_dependcy).cpu().numpy()


            label_domainslot = label_domainslot.to('cpu').numpy()
            label_domain = label_domain.view(-1).to('cpu').numpy()
            label_dependcy = label_dependcy.view(-1).to('cpu').numpy()

            scores_domainslot.append(logits_domainslot)
            scores_domain.append(logits_domain)
            scores_dependcy.append(logits_dependcy)

            gold_labels_domainslot.append(label_domainslot)
            gold_labels_domain.append(label_domain)
            gold_labels_dependcy.append(label_dependcy)

        gold_labels_domainslot = np.concatenate(gold_labels_domainslot, 0)
        gold_labels_domain = np.concatenate(gold_labels_domain, 0)
        gold_labels_dependcy = np.concatenate(gold_labels_dependcy, 0)
        scores_domainslot = np.concatenate(scores_domainslot, 0)
        scores_domain = np.concatenate(scores_domain, 0)
        scores_dependcy = np.concatenate(scores_dependcy, 0)

        # 计算评价指标
        assert   scores_domain.shape[0] == scores_dependcy.shape[0] == gold_labels_domain.shape[0] == gold_labels_dependcy.shape[0]
        eval_accuracy_domain = accuracyF1(scores_domain, gold_labels_domain,mode='domain',report=True)
        eval_accuracy_dependcy = accuracyF1(scores_dependcy, gold_labels_dependcy,mode='dependcy',report=True)
        eval_jointGoal = compute_jointGoal_domainslot(
            dialogueID,
            utterance_text,
            scores_domainslot,
            gold_labels_domainslot,
            scores_domain,
            gold_labels_domain,
            scores_dependcy,
            gold_labels_dependcy
        )
        print('eval_accuracy_domain',eval_accuracy_domain)
        print('eval_accuracy_dependcy', eval_accuracy_dependcy)
        print('eval_jointGoal', eval_jointGoal)


if __name__ == "__main__":

    trainer = Trainer(
        data_dir = '/home/lsy2018/TextClassification/DATA/DATA_MultiWOZ/data_1203/',
        output_dir = './model_MultiWOZ_1204',
        # DOUBAN 是二分类
        num_labels_value = 2,
        num_labels_domainslot= 2,
        args = args)
    trainer.train()
    time_start = time.time()
    # trainer.test_eval()
    print('1000条测试运行时间',time.time()-time_start,'s')
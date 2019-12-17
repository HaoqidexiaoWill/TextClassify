from __future__ import absolute_import

import os
import random
import time
from io import open
import numpy as np
import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from pytorch_transformers.modeling_Matrix2 import BertForTokenClassification
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

from itertools import cycle

from Config.argsMultiWOZ import args
from Utils.Logger import logger
from DATAProcess.LoadDATAMultiWOZ7 import DATAMultiWOZ
from Metric.ComputeMultiWOZ3 import compute_jointGoal_domainslot_1_
os.environ["CUDA_VISIBLE_DEVICES"]='1'
class Trainer:
    def __init__(self,data_dir,output_dir,num_labels_domain,num_labels_dependcy,args):

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_labels_domain = num_labels_domain
        self.num_labels_dependcy = num_labels_dependcy


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
        all_utterance_mask = torch.tensor(data.select_field(train_features, 'utterance_mask'), dtype=torch.long)
        all_domainslot_mask = torch.tensor(data.select_field(train_features, 'domainslot_mask'), dtype=torch.long)
        all_label_tokens_start = torch.tensor([f.label_tokens_start for f in train_features], dtype=torch.long)
        all_label_tokens_end = torch.tensor([f.label_tokens_end for f in train_features], dtype=torch.long)
        all_label_sentence_domainslot = torch.tensor([f.label_sentence_domainslot for f in train_features], dtype=torch.long)
        all_label_tokens_domainslot = torch.tensor([f.label_tokens_domainslot for f in train_features], dtype=torch.long)

        all_hist_tokens = [f.hist_token for f in train_features]

        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_utterance_mask,
            all_domainslot_mask,
            all_label_tokens_start,
            all_label_tokens_end,
            all_label_sentence_domainslot,
            all_label_tokens_domainslot
        )

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)

        eval_examples = data.read_examples(os.path.join(self.data_dir, 'test.json'))
        eval_features = data.convert_examples_to_features(eval_examples, self.tokenizer, self.max_seq_length)
        eval_input_ids = torch.tensor(data.select_field(eval_features, 'input_ids'), dtype=torch.long)
        eval_input_mask = torch.tensor(data.select_field(eval_features, 'input_mask'), dtype=torch.long)
        eval_segment_ids = torch.tensor(data.select_field(eval_features, 'segment_ids'), dtype=torch.long)
        eval_utterance_mask = torch.tensor(data.select_field(eval_features, 'utterance_mask'), dtype=torch.long)
        eval_domainslot_mask = torch.tensor(data.select_field(eval_features, 'domainslot_mask'), dtype=torch.long)
        eval_label_tokens_start = torch.tensor([f.label_tokens_start for f in eval_features], dtype=torch.long)
        eval_label_tokens_end = torch.tensor([f.label_tokens_end for f in eval_features], dtype=torch.long)
        eval_label_sentence_domainslot = torch.tensor([f.label_sentence_domainslot for f in eval_features], dtype=torch.long)
        eval_label_tokens_domainslot = torch.tensor([f.label_tokens_domainslot for f in eval_features], dtype=torch.long)

        eval_hist_tokens = [f.hist_token for f in eval_features]


        eval_data = TensorDataset(
            eval_input_ids,
            eval_input_mask,
            eval_segment_ids,
            eval_utterance_mask,
            eval_domainslot_mask,
            eval_label_tokens_start,
            eval_label_tokens_end,
            eval_label_sentence_domainslot,
            eval_label_tokens_domainslot

        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)

        return train_dataloader,eval_dataloader,train_examples,eval_examples,all_hist_tokens,eval_hist_tokens

    def train(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


        # logger.info(f'Fold {split_index + 1}')
        train_dataloader, eval_dataloader, train_examples, eval_examples,all_hist_tokens,eval_hist_tokens = self.create_dataloader()

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
            utterance_mask,domainslot_mask,\
            label_tokens_start,label_tokens_end,\
            label_sentence_domainslot,label_tokens_domainslot = batch

            loss_tokenstart,loss_tokenend,loss_sentence_domainslot,loss_token_domainslot = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                utterance_mask = utterance_mask,
                domainslot_mask = domainslot_mask,
                label_tokens_start = label_tokens_start,
                label_tokens_end=label_tokens_end,
                label_sentence_domainslot = label_sentence_domainslot,
                label_tokens_domainslot = label_tokens_domainslot
            )
            # loss = loss_tokenstart+loss_tokenend+loss_sentence_domainslot+loss_token_domainslot
            loss = loss_tokenstart + loss_tokenend + loss_sentence_domainslot+loss_token_domainslot
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
                    gold_labels_tokens_start = []
                    gold_labels_tokens_end = []
                    gold_label_sentence_domainslot = []
                    gold_label_tokens_domainslot = []
                    scores_tokens_start = []
                    scores_tokens_end = []
                    scores_sentence_domainslot = []
                    scores_tokens_domainslot = []
                    # ID = [x.guid for x in eval_examples]
                    dialogueID = [x.guid for x in eval_examples]

                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", self.eval_batch_size)

                    model.eval()
                    eval_loss_tokens_start,eval_loss_tokens_end,eval_loss_sentence_domainslot,eval_loss_tokens_domainslot = 0,0,0,0
                    eval_F1_tokens_start,eval_F1_tokens_end = 0,0
                    eval_F1_sentence_domainslot,eval_F1_tokens_domainslot = 0,0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    for input_ids,input_mask, segment_ids,\
                        utterance_mask,domainslot_mask,\
                        label_tokens_start,label_tokens_end,\
                        label_sentence_domainslot,label_tokens_domainslot in eval_dataloader:
                        input_ids = input_ids.to(self.device)
                        input_mask = input_mask.to(self.device)
                        segment_ids = segment_ids.to(self.device)
                        utterance_mask = utterance_mask.to(self.device)
                        domainslot_mask = domainslot_mask.to(self.device)
                        label_tokens_start = label_tokens_start.to(self.device)
                        label_tokens_end = label_tokens_end.to(self.device)
                        label_sentence_domainslot = label_sentence_domainslot.to(self.device)
                        # print(label_sentence_domainslot.size())
                        # print(label_sentence_domainslot)
                        label_tokens_domainslot = label_tokens_domainslot.to(self.device)


                        with torch.no_grad():
                            batch_eval_loss_tokens_start,batch_eval_loss_tokens_end,batch_eval_loss_sentence_domainslot,batch_eval_loss_tokens_domainslot = model(
                                input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                utterance_mask = utterance_mask,
                                domainslot_mask = domainslot_mask,
                                label_tokens_start = label_tokens_start,
                                label_tokens_end=label_tokens_end,
                                label_sentence_domainslot=label_sentence_domainslot,
                                label_tokens_domainslot = label_tokens_domainslot
                            )
                            logits_tokens_start,logits_tokens_end,logits_sentence_domainslot,logits_tokens_domainslot = model(
                                input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                utterance_mask = utterance_mask,
                                domainslot_mask = domainslot_mask

                            )

                        logits_tokens_start = logits_tokens_start.view(-1, 2).cpu().numpy()
                        logits_tokens_end = logits_tokens_end.view(-1,2).cpu().numpy()
                        logits_tokens_domainslot = logits_tokens_domainslot.view(-1,2).detach().cpu().numpy()
                        logits_sentence_domainslot = logits_sentence_domainslot.view(-1,2).cpu().numpy()


                        label_tokens_start = label_tokens_start.view(-1).to('cpu').numpy()
                        label_tokens_end = label_tokens_end.view(-1).to('cpu').numpy()
                        label_sentence_domainslot = label_sentence_domainslot.to('cpu').numpy()
                        label_tokens_domainslot = label_tokens_domainslot.to('cpu').numpy()

                        scores_tokens_start.append(logits_tokens_start)
                        scores_tokens_end.append(logits_tokens_end)
                        scores_sentence_domainslot.append(logits_sentence_domainslot)
                        scores_tokens_domainslot.append(logits_tokens_domainslot)

                        gold_labels_tokens_start.append(label_tokens_start)
                        gold_labels_tokens_end.append(label_tokens_end)
                        gold_label_sentence_domainslot.append(label_sentence_domainslot)
                        gold_label_tokens_domainslot.append(label_tokens_domainslot)

                        eval_loss_tokens_start += batch_eval_loss_tokens_start.mean().item()
                        eval_loss_tokens_end += batch_eval_loss_tokens_end.mean().item()
                        eval_loss_sentence_domainslot += batch_eval_loss_sentence_domainslot.mean().item()
                        eval_loss_tokens_domainslot += batch_eval_loss_tokens_domainslot.mean().item()

                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    gold_labels_tokens_start = np.concatenate(gold_labels_tokens_start,0)
                    gold_labels_tokens_end = np.concatenate(gold_labels_tokens_end,0)
                    gold_label_sentence_domainslot = np.concatenate(gold_label_sentence_domainslot,0)
                    gold_label_tokens_domainslot = np.concatenate(gold_label_tokens_domainslot,0)

                    scores_tokens_start = np.concatenate(scores_tokens_start, 0)
                    scores_tokens_end = np.concatenate(scores_tokens_end, 0)
                    scores_sentence_domainslot = np.concatenate(scores_sentence_domainslot, 0)
                    scores_tokens_domainslot = np.concatenate(scores_tokens_domainslot,0)

                    model.train()
                    eval_loss_tokens_start = eval_loss_tokens_start/nb_eval_steps
                    eval_loss_tokens_end = eval_loss_tokens_end / nb_eval_steps
                    eval_loss_sentence_domainslot = eval_loss_sentence_domainslot / nb_eval_steps
                    eval_loss_tokens_domainslot = eval_loss_tokens_domainslot /nb_eval_steps


                    eval_F1_tokenstart,eval_F1_tokenend,F1_sentence_domainslot,F1_token_domainslot = compute_jointGoal_domainslot_1_(
                        dialogueID,
                        eval_hist_tokens,
                        scores_tokens_start,
                        scores_tokens_end,
                        scores_sentence_domainslot,
                        scores_tokens_domainslot,
                        gold_labels_tokens_start,
                        gold_labels_tokens_end,
                        gold_label_sentence_domainslot,
                        gold_label_tokens_domainslot
                    )


                    print(
                        'F1_token_domainslot',F1_token_domainslot,
                        'F1_sentence_domainslot',F1_sentence_domainslot,
                        'eval_F1_tokenstart',eval_F1_tokenstart,
                        'eval_F1_tokenend', eval_F1_tokenend,
                        'global_step',global_step,
                        'loss',train_loss
                    )
                    result = {

                        'eval_loss_tokens_start':eval_loss_tokens_start,
                        'eval_loss_tokens_end': eval_loss_tokens_end,
                        'eval_loss_sentence_domainslot':eval_loss_sentence_domainslot,
                        'eval_loss_tokens_domainslot':eval_loss_tokens_domainslot,

                        'F1_token_domainslot':F1_token_domainslot,
                        'eval_F1_sentence_domainslot': F1_sentence_domainslot,
                        'eval_F1_tokenstart': eval_F1_tokenstart,
                        'eval_F1_tokenend': eval_F1_tokenend,
                        'global_step': global_step,
                        'loss': train_loss}

                    output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
                        writer.write('*' * 80)
                        writer.write('\n')
                    if eval_F1_tokenstart > best_acc :
                        print("=" * 80)
                        print("Best jointGoal", eval_F1_tokenstart)
                        print("Saving Model......")
                        # best_acc = eval_accuracy
                        best_acc = eval_F1_tokenstart
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

        test_features = data.convert_examples_to_features(test_examples, self.tokenizer, self.max_seq_length)
        test_input_ids = torch.tensor(data.select_field(test_features, 'input_ids'), dtype=torch.long)
        test_input_mask = torch.tensor(data.select_field(test_features, 'input_mask'), dtype=torch.long)
        test_segment_ids = torch.tensor(data.select_field(test_features, 'segment_ids'), dtype=torch.long)
        test_utterance_mask = torch.tensor(data.select_field(test_features, 'utterance_mask'), dtype=torch.long)
        test_domainslot_mask = torch.tensor(data.select_field(test_features, 'domainslot_mask'), dtype=torch.long)
        test_label_tokens_start = torch.tensor([f.label_tokens_start for f in test_features], dtype=torch.long)
        test_label_tokens_end = torch.tensor([f.label_tokens_end for f in test_features], dtype=torch.long)
        test_label_sentence_domainslot = torch.tensor([f.label_sentence_domainslot for f in test_features], dtype=torch.long)
        test_label_tokens_domainslot = torch.tensor([f.label_tokens_domainslot for f in test_features], dtype=torch.long)

        test_hist_tokens = [f.hist_token for f in test_features]

        test_data = TensorDataset(
            test_input_ids,
            test_input_mask,
            test_segment_ids,
            test_utterance_mask,
            test_domainslot_mask,
            test_label_tokens_start,
            test_label_tokens_end,
            test_label_sentence_domainslot,
            test_label_tokens_domainslot
        )
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.eval_batch_size)



        config = BertConfig.from_pretrained(self.model_name_or_path)
        model = BertForTokenClassification.from_pretrained(
            os.path.join(self.output_dir, "pytorch_model.bin"), self.args, config=config)
        model.to(self.device)
        model.eval()

        gold_labels_tokens_start = []
        gold_labels_tokens_end = []
        gold_label_sentence_domainslot = []
        gold_label_tokens_domainslot = []
        scores_tokens_start = []
        scores_tokens_end = []
        scores_sentence_domainslot = []
        scores_tokens_domainslot = []

        for input_ids, input_mask, segment_ids, \
            utterance_mask, domainslot_mask, \
            label_tokens_start, label_tokens_end, \
            label_sentence_domainslot, label_tokens_domainslot in test_dataloader:

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            utterance_mask = utterance_mask.to(self.device)
            domainslot_mask = domainslot_mask.to(self.device)
            label_tokens_start = label_tokens_start.to(self.device)
            label_tokens_end = label_tokens_end.to(self.device)
            label_sentence_domainslot = label_sentence_domainslot.to(self.device)
            # print(label_sentence_domainslot.size())
            # print(label_sentence_domainslot)
            label_tokens_domainslot = label_tokens_domainslot.to(self.device)

            with torch.no_grad():
                batch_eval_loss_tokens_start, batch_eval_loss_tokens_end, batch_eval_loss_sentence_domainslot, batch_eval_loss_tokens_domainslot = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    utterance_mask=utterance_mask,
                    domainslot_mask=domainslot_mask,
                    label_tokens_start=label_tokens_start,
                    label_tokens_end=label_tokens_end,
                    label_sentence_domainslot=label_sentence_domainslot,
                    label_tokens_domainslot=label_tokens_domainslot
                )
                logits_tokens_start, logits_tokens_end, logits_sentence_domainslot, logits_tokens_domainslot = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    utterance_mask=utterance_mask,
                    domainslot_mask=domainslot_mask

                )

            logits_tokens_start = logits_tokens_start.view(-1, 2).cpu().numpy()
            logits_tokens_end = logits_tokens_end.view(-1, 2).cpu().numpy()
            logits_tokens_domainslot = logits_tokens_domainslot.view(-1, 2).detach().cpu().numpy()
            logits_sentence_domainslot = logits_sentence_domainslot.view(-1, 2).cpu().numpy()

            label_tokens_start = label_tokens_start.view(-1).to('cpu').numpy()
            label_tokens_end = label_tokens_end.view(-1).to('cpu').numpy()
            label_sentence_domainslot = label_sentence_domainslot.to('cpu').numpy()
            label_tokens_domainslot = label_tokens_domainslot.to('cpu').numpy()

            scores_tokens_start.append(logits_tokens_start)
            scores_tokens_end.append(logits_tokens_end)
            scores_sentence_domainslot.append(logits_sentence_domainslot)
            scores_tokens_domainslot.append(logits_tokens_domainslot)

            gold_labels_tokens_start.append(label_tokens_start)
            gold_labels_tokens_end.append(label_tokens_end)
            gold_label_sentence_domainslot.append(label_sentence_domainslot)
            gold_label_tokens_domainslot.append(label_tokens_domainslot)

        gold_labels_tokens_start = np.concatenate(gold_labels_tokens_start, 0)
        gold_labels_tokens_end = np.concatenate(gold_labels_tokens_end, 0)
        gold_label_sentence_domainslot = np.concatenate(gold_label_sentence_domainslot, 0)
        gold_label_tokens_domainslot = np.concatenate(gold_label_tokens_domainslot, 0)

        scores_tokens_start = np.concatenate(scores_tokens_start, 0)
        scores_tokens_end = np.concatenate(scores_tokens_end, 0)
        scores_sentence_domainslot = np.concatenate(scores_sentence_domainslot, 0)
        scores_tokens_domainslot = np.concatenate(scores_tokens_domainslot, 0)

        # 计算评价指标
        # eval_accuracy_domain = accuracyF1(scores_domain, gold_labels_domain,mode='domain',report=True)
        # eval_accuracy_dependcy = accuracyF1(scores_dependcy, gold_labels_dependcy,mode='dependcy',report=True)
        eval_F1_tokenstart, eval_F1_tokenend, F1_sentence_domainslot, F1_token_domainslot = compute_jointGoal_domainslot_1_(
            dialogueID,
            test_hist_tokens,
            scores_tokens_start,
            scores_tokens_end,
            scores_sentence_domainslot,
            scores_tokens_domainslot,
            gold_labels_tokens_start,
            gold_labels_tokens_end,
            gold_label_sentence_domainslot,
            gold_label_tokens_domainslot
        )

        print(
            'F1_token_domainslot', F1_token_domainslot,
            'F1_sentence_domainslot', F1_sentence_domainslot,
            'eval_F1_tokenstart', eval_F1_tokenstart,
            'eval_F1_tokenend', eval_F1_tokenend
        )


if __name__ == "__main__":

    trainer = Trainer(
        data_dir = '/home/lsy2018/TextClassification/DATA/DATA_MultiWOZ/data_1135/',
        output_dir = './model_MultiWOZ_1208',
        # DOUBAN 是二分类
        num_labels_domain = 32,
        num_labels_dependcy = 4,
        args = args)
    trainer.train()
    time_start = time.time()
    # trainer.test_eval()
    print('1000条测试运行时间',time.time()-time_start,'s')
from __future__ import absolute_import
import random
from io import open
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

from pytorch_transformers.modeling_bert import BertForSequenceClassification, BertConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from itertools import cycle

from metric import accuracy
from oldcode.args import args
from oldcode.LoadDataBDCI import *

class Trainer:
    def __init__(self,debug,args,data_dir,data_process_output):

        self.eval_steps = args.eval_steps
        self.adam_epsilon = args.adam_epsilon
        self.warmup_steps = args.warmup_steps
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.device = torch.device('cuda')
        self.debug = debug
        self.seed = 2019
        self.args = args
        self.data_dir = args.data_dir
        self.max_seq_length = args.max_seq_length
        self.batch_size = args.per_gpu_train_batch_size
        self.train_steps = args.train_steps
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        self.config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=3)
        self.seed_everything()
        self.do_eval = True

        self.data_dir = data_dir
        self.data_process_output = data_process_output
        self.output_dir = './'
    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
    def train(self):
        model = BertForSequenceClassification.from_pretrained(
            self.args.model_name_or_path, self.args, config=self.config)
        model.to(self.device)

        logger.info('准备数据')
        data = DATABDCI(
        debug=False,
        data_dir='/home/lsy2018/文本匹配/DATA/DATA_BDCI/',
        data_process_output='/home/lsy2018/文本匹配/DATA/DATA_BDCI/data_1014/')

        train_examples = data.read_examples(os.path.join(self.data_process_output, 'train.csv'))
        train_features = data.convert_examples_to_features(
            train_examples, self.tokenizer, self.max_seq_length)
        all_input_ids = torch.tensor(data.select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(data.select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(data.select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

        # 这步干嘛的？
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=self.batch_size // self.gradient_accumulation_steps)

        # Prepare optimizer

        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.warmup_steps, t_total=self.train_steps)

        best_acc = 0
        global_step = 0
        model.train()
        train_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        bar = tqdm(range(self.train_steps), total=self.train_steps)
        train_dataloader = cycle(train_dataloader)

        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)

            train_loss += loss.item()
            train_loss = round(train_loss * self.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % self.gradient_accumulation_steps == 0:
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if self.do_eval and (step + 1) % (self.eval_steps * self.gradient_accumulation_steps) == 0:
                inference_labels = []
                scores = []
                gold_labels = []
                inference_logits = []
                eval_examples = data.read_examples(os.path.join(self.data_process_output, 'dev.csv'))
                eval_features = data.convert_examples_to_features(
                    eval_examples, self.tokenizer, self.max_seq_length)
                ID1 = [x.sentence_ID1 for x in eval_examples]
                ID2 = [x.sentence_ID2 for x in eval_examples]

                all_input_ids = torch.tensor(data.select_field(eval_features, 'input_ids'), dtype=torch.long)
                all_input_mask = torch.tensor(data.select_field(eval_features, 'input_mask'), dtype=torch.long)
                all_segment_ids = torch.tensor(data.select_field(eval_features, 'segment_ids'), dtype=torch.long)
                all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", self.batch_size)

                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                count = 0

                for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                    # ID1_list_eachbatch = ID1[count*args.eval_batch_size:(count+1)*args.eval_batch_size]
                    # ID2_list_eachbatch = ID2[count * args.eval_batch_size:(count + 1) * args.eval_batch_size]
                    input_ids = input_ids.to(self.device)
                    input_mask = input_mask.to(self.device)
                    segment_ids = segment_ids.to(self.device)
                    label_ids = label_ids.to(self.device)


                    with torch.no_grad():
                        tmp_eval_loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                        logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)


                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        inference_labels.append(np.argmax(logits, axis=1))
                        # scores.append(logits)
                        gold_labels.append(label_ids)
                        inference_logits.append(logits)
                        eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    gold_labels = np.concatenate(gold_labels, 0)
                    inference_logits = np.concatenate(inference_logits, 0)
                    # scores = np.concatenate(scores, 0)
                    model.train()
                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = accuracy(inference_logits, gold_labels)
                    # eval_mrr = compute_MRR(scores, gold_labels, ID1, ID2)


                    result = {'eval_loss': eval_loss,
                              'eval_F1': eval_accuracy,
                              'global_step': global_step,
                              # 'mrr':eval_mrr,
                              'loss': train_loss}

                    output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
                        writer.write('*' * 80)
                        writer.write('\n')
                    if eval_accuracy > best_acc:
                        print("=" * 80)
                        print("Best F1", eval_accuracy)
                        print("Saving Model......")
                        best_acc = eval_accuracy
                        # Save a trained model
                        model_to_save = model.module if hasattr(model,'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(self.output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("=" * 80)
                    else:
                        print("=" * 80)
if __name__ == "__main__":
    trainer = Trainer(
        debug=False,args = args,
        data_dir='/home/lsy2018/文本匹配/DATA/DATA_BDCI/',
        data_process_output='/home/lsy2018/文本匹配/DATA/DATA_BDCI/data_1014/'
    )
    trainer.train()


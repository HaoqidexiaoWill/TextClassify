import torch
from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer, BertAdam
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
# from modeling import GraphMultiHeadAttention
from torch.nn import BCEWithLogitsLoss
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from metric import obtain_TP_TN_FN_FP
import json
import os
import random
import numpy as np
import time
import sys
import logging

from DATAProcess.LoadDataDSTC8 import DSTC8Data

do_train = True


# do_eval=True

def get_train_logger(log_path):
    logger = logging.getLogger('train-{}'.format(__name__))
    logger.setLevel(logging.INFO)
    # 控制台
    handler_stream = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_stream)

    # 日志文件
    handler_file = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    return logger


logger = get_train_logger('log.txt')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Trainer:
    def __init__(self, data_dir, model_name, epochs=1, batch_size=64, base_batch_size=32, max_len=120, seed=1234,
                 debug_mode=False, num_labels=44,
                 learning_rate=5e-5, warmup_proportion=0.1, gradient_accumulation_steps=1, period=500, test_set="dev",
                 load_step=0):
        self.device = torch.device('cuda')
        self.data_dir = data_dir
        self.debug_mode = debug_mode
        self.model_name = model_name
        self.seed = seed
        self.seed_everything()
        self.max_len = max_len
        self.epochs = epochs
        self.base_batch_size = base_batch_size
        self.batch_size = batch_size
        self.num_labels = num_labels

        self.learning_rate = float(learning_rate)
        self.warmup_proportion = warmup_proportion
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.period = period

        self.test_set = test_set
        self.load_step = load_step

        if not os.path.exists(self.data_dir):
            raise NotImplementedError()
        else:
            self.train_data_path = os.path.join(self.data_dir, "train.json")
            self.valid_data_path = os.path.join(self.data_dir, "dev.json")
            # self.test_data_path = os.path.join(self.data_dir, "va.json")
            self.bert_model_path = "/home/lsy2018/BERT文本相似度/uncased_L-12_H-768_A-12/"
            self.pytorch_bert_path = "/home/lsy2018/BERT文本相似度/uncased_L-12_H-768_A-12/pytorch_model.bin"
            self.bert_config = BertConfig("/home/lsy2018/BERT文本相似度/uncased_L-12_H-768_A-12/bert_config.json")

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def create_dataloader(self):
        # 创建 batch
        # 将 dataset 转成 dataloader
        multiwoz = DSTC8Data()
        train_data_list = multiwoz.load_data(self.train_data_path)
        train_examples_length = len(train_data_list)
        valid_data_list = multiwoz.load_data(self.valid_data_path)
        valid_examples_length = len(valid_data_list)
        # test_data_list = self.load_data(self.test_data_path)
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)

        train_features = multiwoz.convert_lines_to_features(train_data_list, 256, bert_tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        valid_features = multiwoz.convert_lines_to_features(valid_data_list, 256, bert_tokenizer)
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.float)

        train_data = data.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.base_batch_size)
        eval_data = data.TensorDataset(valid_all_input_ids, valid_all_input_mask, valid_all_segment_ids,
                                       valid_all_label_ids)
        eval_dataloader = torch.utils.data.DataLoader(eval_data, batch_size=self.base_batch_size)

        # return train_dataloader,valid_loader
        return train_dataloader, eval_dataloader, train_examples_length, valid_examples_length, valid_features

    def train(self):
        device = torch.device("cuda:0")
        # pdb.set_trace()
        if self.debug_mode: self.epochs = 2
        print('加载dataloader')
        # train_loader, valid_loader = self.create_dataloader()
        train_dataloader, eval_dataloader, train_examples_length, valid_examples_length, eval_features = self.create_dataloader()
        print('开始训练')

        num_train_optimization_steps = None
        if do_train:
            num_train_optimization_steps = int(
                train_examples_length / self.batch_size / self.gradient_accumulation_steps) * self.epochs
        model = BertForSequenceClassification.from_pretrained(self.bert_model_path, cache_dir=None,
                                                              num_labels=self.num_labels).cuda()
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.learning_rate,
                             warmup=self.warmup_proportion,
                             t_total=num_train_optimization_steps)

        global_step = 0
        tr_loss = 0
        best_F1 = 0

        tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)

        model.train()
        for epoch in range(int(self.epochs)):
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                print('epoch:', epoch, 'batchIndex:', step)
                batch = tuple(t.to(device) for t in batch)
                # pdb.set_trace()
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = model(input_ids.cuda(), segment_ids.cuda(), input_mask.cuda(), labels=None).cuda()
                loss_fct = BCEWithLogitsLoss()
                label_ids = label_ids.cuda()
                loss = loss_fct(logits.view(-1, 1), label_ids.view(-1, 1))

                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    model.zero_grad()
                    global_step += 1

                if (step + 1) % self.period == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model

                    model.eval()
                    torch.set_grad_enabled(False)

                    # 开始验证
                    idx = 0
                    TP, TN, FN, FP = 0, 0, 0, 0
                    output = {}
                    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        batch_size = input_ids.size(0)
                        with torch.no_grad():
                            logits = model(input_ids, segment_ids, input_mask, labels=None)
                            logits = torch.sigmoid(logits)
                        preds = (logits > 0.4).float()
                        preds_numpy = preds.cpu().long().data.numpy()
                        for i in range(idx, idx + batch_size):
                            if eval_features[i].file not in output:
                                output[eval_features[i].file] = {}
                            output[eval_features[i].file][eval_features[i].turn] = preds_numpy[i - idx].tolist()
                        TP, TN, FN, FP = obtain_TP_TN_FN_FP(preds, label_ids, TP, TN, FN, FP)
                        idx += batch_size

                    with open("data/BERT_{}_prediction.json".format(self.test_set), 'w') as f:
                        json.dump(output, f)

                    precision = TP / (TP + FP + 0.001)
                    recall = TP / (TP + FN + 0.001)
                    F1 = 2 * precision * recall / (precision + recall + 0.001)
                    logger.info(
                        "epoch is {} step is {} precision is {} recall is {} F1 is {} best_F1 is {}".format(epoch, step,
                                                                                                            precision,
                                                                                                            recall, F1,
                                                                                                            best_F1))

                    # F1 = evaluate(args, model, device, processor, label_list, num_labels, tokenizer, output_mode)

                    if F1 > best_F1:
                        output_dir = os.path.join("checkpoints/predictor/", 'save_step_{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(output_dir)

                        best_F1 = F1

                    model.train()  # turn on train mode
                    torch.set_grad_enabled(True)  # start gradient tracking
                    tr_loss = 0


if __name__ == "__main__":
    # if do_train:
    data_dir = "/home/lsy2018/文本匹配/datapro/data_ubuntu"
    trainer = Trainer(data_dir, "model_name", epochs=5, batch_size=16, base_batch_size=16, max_len=128,
                      debug_mode=False, num_labels=35,
                      learning_rate=5e-5, warmup_proportion=0.1, gradient_accumulation_steps=1, period=50,
                      test_set="dev", load_step=0)
    time1 = time.time()
    trainer.train()
    print("训练时间: %d min" % int((time.time() - time1) / 60))
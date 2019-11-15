from DATAProcess.LoadDataDoubanDGL2 import DataDOUBAN
from itertools import cycle
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import operator
import time
import random
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from Utils.Logger import logger
from metric import compute_DOUBAN
from Model.GCN import Classifier

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Trainer:
    def __init__(self, data_dir, output_dir):
        self.eval_steps = 200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_steps = 5000
        self.train_batchsize = 32
        self.eval_batchsize = 4
        self.seed = 2019
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        self.output_dir = output_dir
        self.seed_everything()

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels).to(self.device)

    def create_dataloader(self):
        trainset = DataDOUBAN(mode='train')
        testset = DataDOUBAN(mode='test')
        train_dataloader = DataLoader(
            trainset,
            batch_size=self.train_batchsize,
            shuffle=True,
            collate_fn=self.collate)
        test_dataloader = DataLoader(
            testset,
            batch_size=4,
            shuffle=False,
            collate_fn=self.collate)
        return trainset, train_dataloader, testset, test_dataloader

    def train(self):
        trainset, train_dataloader, testset, test_dataloader = self.create_dataloader()
        model = Classifier(1, 256, trainset.num_classes).to(self.device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(trainset.ids))
        logger.info("  Batch size = %d", self.train_batchsize)
        logger.info("  Num steps = %d", self.train_steps)

        global_step, nb_tr_steps, tr_loss = 0, 0, 0
        best_MRR = 0
        train_dataloader = cycle(train_dataloader)

        for each_step in range(self.train_steps):
            bg, label = next(train_dataloader)
            prediction = model(bg)
            loss = loss_func(prediction, label)
            tr_loss += loss.item()
            train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
            nb_tr_steps += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if (each_step + 1) % (self.eval_steps) == 0:
                tr_loss = 0
                nb_tr_steps = 0
                logger.info("***** Report result *****")
                logger.info("  %s = %s", 'global_step', str(global_step))
                logger.info("  %s = %s", 'train loss', str(train_loss))
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(testset.ids))
                logger.info("  Batch size = %d", self.eval_batchsize)

                scores = []
                labels = []
                ids = testset.ids
                model.eval()

                for iter, (bg, label) in enumerate(test_dataloader):
                    with torch.no_grad():
                        logits = model(bg).detach().cpu().numpy()
                    label = label.detach().cpu().numpy()
                    scores.append(logits)
                    labels.append(label)
                scores = np.concatenate(scores, 0)
                labels = np.concatenate(labels, 0)
                model.train()

                assert len(ids) == len(scores) == len(labels)
                eval_DOUBAN_MRR, eval_DOUBAN_mrr, eval_DOUBAN_MAP, eval_Precision1 = compute_DOUBAN(ids, scores, labels)
                print(
                    'eval_MRR', eval_DOUBAN_MRR, eval_DOUBAN_mrr,
                    'eval_MAP', eval_DOUBAN_MAP,
                    'eval_Precision1', eval_Precision1)
                result = {'eval_MRR': eval_DOUBAN_MRR,
                          'eval_MAP': eval_DOUBAN_MAP,
                          'eval_Precision1': eval_Precision1,
                          'global_step': global_step,
                          'loss': train_loss}
                output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
                    writer.write('*' * 80)
                    writer.write('\n')
                if eval_DOUBAN_MRR > best_MRR:
                    print("=" * 80)
                    print("Best MRR", eval_DOUBAN_MRR)
                    print("Saving Model......")
                    best_MRR = eval_DOUBAN_MRR
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(self.output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    print("=" * 80)
                else:
                    print("=" * 80)


if __name__ == "__main__":
    trainer = Trainer(
        data_dir='/home/lsy2018/TextClassification/DATA/DATA_DOUBAN/data_1102/',
        output_dir='./model_DOUBAN_DGL3')
    trainer.train()
    time_start = time.time()
    # trainer.test_eval()
    print('测试运行时间', time.time() - time_start, 's')
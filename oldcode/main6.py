import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torchtext
from torchtext import data
from oldcode.ModelLSTM import Net
from sklearn.metrics import f1_score

class Trainer:
    def __init__(self):
        self.Max_Len = 200  # 句子的最大长度
        self.EPOCHS = 400  # epoch的个数
        self.BATCH_SIZE = 64  # batch大小
        self.HIDDEN_DIM = 128  # 隐藏层长度
        self.LR = 5e-5  # 优化器学习率
        self.WEIGHT_DECAY = 1e-2  # 优化器衰减系数
        self.BEST_ACC = 0  # 最优的准确率
        self.BEST_EPOCH = 0  # 最优的epoch
        self.device = torch.device('cuda')
    def tokenizer_id(self,text):
        # 处理ID字段
        return text
    def tokenizer_label(self,id):
        # 处理标签字段
        return int(id)
    def tokenizer_text(self,text):
        # 处理文本字段
        return text.split()
    def load_embedding(self):
        IDX = data.Field(
            sequential=False, tokenize=self.tokenizer_id, use_vocab=True)  # sequential=False就不用拆分该字段内容，保留整体。
        TEXT = data.Field(
            sequential=True, tokenize=self.tokenizer_text, use_vocab=True, fix_length=self.Max_Len)
        LABEL = data.Field(
            sequential=False, tokenize=self.tokenizer_label, use_vocab=False, dtype=torch.long)
        '''
        train, val, test = data.TabularDataset.splits(path='/home/lsy2018/文本匹配/datapro/data1014/', train='train.csv', validation='val.csv',
                                                      test='test.csv', format='csv', skip_header=True,
                                                      fields=[('id', IDX), ('text', TEXT), ('label', LABEL)])
        '''
        train, val= data.TabularDataset.splits(
            path='/home/lsy2018/文本匹配/datapro/data1014/', train='train.csv', validation='dev.csv',format='csv',
            skip_header=True,fields=[('id', IDX), ('text', TEXT), ('label', LABEL)])
        vectors = torchtext.vocab.Vectors(
            name='/home/lsy2018/文本匹配/Embedding/glove_42B_300d_vec_plus_word2vec_100.txt',cache='cachefiles/')
        TEXT.build_vocab(train, vectors=vectors)
        # LABEL.build_vocab(train,)
        IDX.build_vocab(train)


        # print(type(TEXT))
        # print(type(TEXT.vocab))
        # print(TEXT.vocab.itos[:200])  # 显示前200个词语
        # print(TEXT.vocab.vectors[200])  # 显示'德福'的词向量


        return IDX, TEXT, train, val

    def accrate_(self,pre, y):
        """
        求分类准确率。
        :param pre:
        :param y:
        :return:
        """
        pre = pre.argmax(dim=1)
        correct = torch.eq(pre, y).sum().float().item()
        acc = correct / float(y.size(0))
        return acc
    def accrate(self,pre, y):
        """
        求分类准确率。
        :param pre:
        :param y:
        :return:
        """
        logits = pre.detach().cpu().numpy()
        label_ids = y.to('cpu').numpy()
        inference_labels = np.argmax(logits, axis=1)
        F1 = f1_score(label_ids, inference_labels, labels=[0, 1], average='macro')
        # pre = pre.argmax(dim=1)
        # correct = torch.eq(pre, y).sum().float().item()
        # acc = correct / float(y.size(0))
        return F1
    def evalute(self,model, val_iter):
        """
        计算验证集的准确率。
        :param net:
        :param val_iter:
        :return:
        """
        model.eval()

        avg_acc = []
        for step, batch in enumerate(val_iter):
            batch.text = batch.text.to(self.device)  # GPU加速
            batch.label = batch.label.to(self.device)

            with torch.no_grad():
                pre = model(batch.text)
                acc = self.accrate(pre, batch.label)

                avg_acc.append(acc)

        avg_acc = np.array(avg_acc).mean()

        return avg_acc
    def train(self):
        IDX, TEXT, train, val = self.load_embedding()
        # create dataloader
        train_iter, val_iter = data.BucketIterator.splits(
            (train, val, ),
            batch_size=self.BATCH_SIZE,
            device=self.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=False)
        model = Net(len(TEXT.vocab),400,self.HIDDEN_DIM).to(self.device)
        pretrained_embedding = TEXT.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embedding)
        optimizer = optim.Adam(model.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        BEST_ACC = 0
        BEST_EPOCH = 0
        for epoch in range(self.EPOCHS):
            avg_acc = []
            for step,batch in enumerate(train_iter):
                batch.text = batch.text.to(self.device)
                batch.label = batch.label.to(self.device)
                model.train()
                logit = model(batch.text)
                # print(batch.text.size())
                # print(logit.size(), batch.label.size())
                loss = criterion(logit,batch.label)

                acc = self.accrate(logit,batch.label)
                avg_acc.append(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 1 == 0:
                    print('epoch:{},batch:{},acc:{},loss:{}'.format(epoch, step, acc, loss))
            print('第{}个epoch的平均准确率为：{}'.format(epoch, np.array(avg_acc).mean()))
            print('第{}个epoch的平均准确率为：{}'.format(epoch, np.array(avg_acc).mean()))

            if epoch % 1 == 0:
                val_acc = self.evalute(model, val_iter)
                if val_acc > BEST_ACC:
                    BEST_EPOCH = epoch
                    BEST_ACC = val_acc

                    torch.save(model.state_dict(), 'best_par.pt')

        print('验证集最优的准确率为：{}'.format(BEST_ACC))
        print('验证集最优的epoch为：{}'.format(BEST_EPOCH))

        # model.load_state_dict(torch.load('best_par.pt'))
        # print('加载最优模型参数完成...')
        #
        # test(model, test_iter, IDX)
        # print('分类完成，请查看输出文件！')

    def test(self,model, test_iter, IDX):
        """
        对官方的测试集进行测试。
        :param net:
        :param test_iter:
        :param IDX:
        :return:
        """
        temp = []
        id = []
        label = []
        for step, batch in enumerate(test_iter):
            id.extend(batch.id)
            batch.text = batch.text.to(self.device)  # GPU加速
            with torch.no_grad():
                pre = model(batch.text)
                pre = pre.argmax(dim=1)  # 示例tensor([0,1,2,1,1,...,])
                pre = pre.cpu().data.numpy()
                pre = pre.tolist()  # 示例[0,1,2,1,1,...,]
                label.extend(pre)

        id = np.array(id)  # array([1,2,5,3,...])
        id = id.tolist()  # [1,2,5,3,...]

        for item in id:  # 将id中的数值转换为真实的id文本
            temp.append(IDX.vocab.itos[item])

        dataframe = pd.DataFrame({'id': temp, 'label': label})
        dataframe.to_csv('endclass.csv', index=False)
        return label



if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
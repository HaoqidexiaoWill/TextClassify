import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

class GraphAttention(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(GraphAttention, self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)


    def forward(self, x):
        print(x.size())
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
sentences = []
labels = []

df = pd.read_csv('输入.csv')
label = df.ix[:,[-1]]
label_list = np.array(label).tolist()
for each in label_list:
    labels.append(each[0])
print(labels)
print(len(labels))
sentences = [x for x in range(len(labels))]
print(sentences)

x = torch.tensor([sentences],dtype=torch.float).transpose(0,1)
y = torch.tensor([labels],dtype=torch.float).transpose(0,1)


import torch
from torch.autograd import Variable  # 使用variable包住数据

# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
print(x.size())
# y = x.pow(2) + 0.2 * torch.rand(x.size())
print(y.size())
# exit()
x, y = Variable(x), Variable(y)



import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):

        super(Net, self).__init__()  # 继承 __init__ 功能

        self.hidden = torch.nn.Linear(n_feature,
                                      n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)


    def forward(self, x):  # 这同时也是 Module 中的 forward 功能

        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)




print(net)

from torch.nn import CrossEntropyLoss
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(1,loss.item())



torch.save(net.state_dict(), "./gat.pt")
import os

# model = torch.load('demo.pt')
print(os.listdir("./"))
__labels__ = net(x).to('cpu').detach().numpy()
# 生成最后的文件

df['predict'] = labels
df.to_csv('输出.csv',index = False)






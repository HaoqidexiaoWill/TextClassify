import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        scale = q.size(-1) ** -0.5

        attention = torch.bmm(q, k.transpose(1, 2))* scale
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context
class TransformerEncoder(nn.Module):
    def __init__(self,hidden_size):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = 8
        self.dim_per_head = hidden_size // self.num_heads
        self.attention_net = ScaledDotProductAttention()
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.linear_k = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.linear_q = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.linear_v = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.final_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self,sequence_output):
        # 残差
        residual = sequence_output

        # batchsize
        batch_size = sequence_output.size(0)

        # linear projection
        key = self.linear_k(sequence_output)
        query = self.linear_q(sequence_output)
        value = self.linear_v(sequence_output)

        # spilt heads 64维
        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)

        # 送入attention_net
        outputs = self.attention_net(query, key, value)

        # 拼接
        outputs = outputs.view(batch_size, -1, self.dim_per_head * self.num_heads)

        # 送入线性层
        outputs = self.final_linear(outputs)
        # dropout
        outputs = self.dropout(outputs)

        # 加上残差项，再layer norm
        outputs = self.layer_norm(outputs + residual)
        return outputs
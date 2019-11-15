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

class TripleAttention(nn.Module):
    def __init__(self,hidden_size):
        super(TripleAttention, self).__init__()
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.FeedForward2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.attention = ScaledDotProductAttention()

    def forward(self, history,utterance,response):
        # history 对 utt的 增强
        context_his2utt = self.attention(utterance,history,history)
        context_his2utt = self.FeedForward2(context_his2utt)
        # utt 对 his 的增强
        context_utt2his = self.attention(history,utterance,utterance)
        context_utt2his = self.FeedForward2(context_utt2his)
        # hist 对 resp的 交互
        context_his2resp = self.attention(response,history,history)
        context_his2resp = self.FeedForward2(context_his2resp)
        # resp 对 hist 的增强
        context_resp2hist = self.attention(history,response,response)
        context_resp2hist = self.FeedForward2(context_resp2hist)
        # utt 对 resp 的增强
        context_utt2resp = self.attention(utterance,response,response)
        context_utt2resp = self.FeedForward2(context_utt2resp)
        # resp 对 utt 的增强
        context_resp2utt = self.attention(response,utterance,utterance)
        context_resp2utt = self.FeedForward2(context_resp2utt)

        # 层归一化
        context_hist = self.layer_norm(context_utt2his+context_resp2hist)
        context_resp = self.layer_norm(context_utt2resp+context_his2resp)
        context_utt = self.layer_norm(context_resp2utt+context_his2utt)

        return context_hist,context_resp,context_utt

class Highway(nn.Module):

    def __init__(self, layer_num, dim=600):
        super(Highway, self).__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x

class TripleAttentionHighWay(nn.Module):
    def __init__(self,hidden_size):
        super(TripleAttentionHighWay, self).__init__()
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.FeedForward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.LayerNorm(self.hidden_size)
        )
        self.FeedForward2 = nn.Sequential(
            nn.Linear(self.hidden_size*4, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.LayerNorm(self.hidden_size)
        )
        self.highway = Highway(layer_num=2,dim=self.hidden_size*4)
        self.attention = ScaledDotProductAttention()

    def forward(self, history,utterance,response):
        # history 对 utt的 增强
        context_his2utt = self.attention(utterance,history,history)
        context_his2utt = self.FeedForward(context_his2utt)
        # utt 对 his 的增强
        context_utt2his = self.attention(history,utterance,utterance)
        context_utt2his = self.FeedForward(context_utt2his)
        # hist 对 resp的 交互
        context_his2resp = self.attention(response,history,history)
        context_his2resp = self.FeedForward(context_his2resp)
        # resp 对 hist 的增强
        context_resp2hist = self.attention(history,response,response)
        context_resp2hist = self.FeedForward(context_resp2hist)
        # utt 对 resp 的增强
        context_utt2resp = self.attention(utterance,response,response)
        context_utt2resp = self.FeedForward(context_utt2resp)
        # resp 对 utt 的增强
        context_resp2utt = self.attention(response,utterance,utterance)
        context_resp2utt = self.FeedForward(context_resp2utt)

        # selfatt
        selfatt_history = self.attention(history,history,history)
        selfatt_utterance = self.attention(utterance,utterance,utterance)
        selfatt_response = self.attention(response,response,response)

        # highway
        highway_history = self.highway(torch.cat([
            history,context_utt2his,context_resp2hist,selfatt_history], dim=-1))
        highway_utterance = self.highway(torch.cat([
            utterance,context_his2utt,context_resp2utt,selfatt_utterance],dim =-1))
        highway_response = self.highway(torch.cat([
            response,context_his2resp,context_utt2resp,selfatt_response],dim = -1))

        # 整合
        context_hist = self.FeedForward2(highway_history)
        context_resp = self.FeedForward2(highway_utterance)
        context_utt = self.FeedForward2(highway_response)

        return context_hist,context_resp,context_utt


class KMaxPooling1D(nn.Module):
    def __init__(self, k):
        super(KMaxPooling1D, self).__init__()
        self.k = k

    def kmax_pooling(self,x, dim, k):
        index = x.topk(min(x.size(dim), k), dim=dim)[1].sort(dim=dim)[0]
        x = x.gather(dim, index)
        if x.size(dim) < k:
            # x = pad_tensor(x, k, dim=dim)
            raise NotImplementedError
        return x
    def forward(self, x):
        # B x L x E
        return self.kmax_pooling(x, 1, self.k)

# 原始版本
'''
class TextCNN1D(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(TextCNN1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DOU_KER_SIZE = [(1, 3), (3, 5), (3, 3), (5, 5)]

        # self.convs = nn.ModuleList([
        #     nn.Conv1d(in_channels=in_channels,
        #               out_channels=out_channels,
        #               kernel_size=fs)
        #     for fs in filter_sizes
        # ])
        self.conv_single = [nn.Sequential(
                nn.Conv1d(in_channels=self.in_channels,
                          out_channels=self.out_channels,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(self.out_channels),
                nn.ReLU(inplace=True),
            )for kernel_size in filter_sizes]
        self.conv_double = [nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=kernel_size[0]),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=self.out_channels,
                      out_channels=self.out_channels,
                      kernel_size=kernel_size[1]),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True),
        ) for kernel_size in self.DOU_KER_SIZE]
        self.convs_single = nn.ModuleList(self.conv_single)
        self.convs_double = nn.ModuleList(self.conv_double)
        # self.init_params()
    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [conv(x) for conv in self.convs_single],[conv(x) for conv in self.convs_double]
'''

class ResnetCNN1D(nn.Module):
    def __init__(self, channel_size):
        super(ResnetCNN1D, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x

class TextCNN1D(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(TextCNN1D, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        self.init_params()
    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [nn.functional.relu(conv(x)) for conv in self.convs]

# 进阶版本
class TextDoubleCNN1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TextDoubleCNN1D, self).__init__()
        self.EMBEDDING_DIM = in_channels
        self.TITLE_DIM = out_channels
        self.SIN_KER_SIZE = [1, 1, 3, 3]
        self.DOU_KER_SIZE = [(1, 3), (3, 5), (3, 3), (5, 5)]

        question_convs1 = [nn.Sequential(
                nn.Conv1d(in_channels=self.EMBEDDING_DIM,
                          out_channels=self.TITLE_DIM,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(self.TITLE_DIM),
                nn.ReLU(inplace=True),

                # nn.MaxPool1d(kernel_size=(self.SENT_LEN - kernel_size + 1))
            )for kernel_size in self.SIN_KER_SIZE]

        question_convs2 = [nn.Sequential(
                nn.Conv1d(in_channels=self.EMBEDDING_DIM,
                          out_channels=self.TITLE_DIM,
                          kernel_size=kernel_size[0]),
                nn.BatchNorm1d(self.TITLE_DIM),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=self.TITLE_DIM,
                          out_channels=self.TITLE_DIM,
                          kernel_size=kernel_size[1]),
                nn.BatchNorm1d(self.TITLE_DIM),
                nn.ReLU(inplace=True),
                # nn.MaxPool1d(kernel_size=(self.SENT_LEN - kernel_size[0] - kernel_size[1] + 2))
            )for kernel_size in self.DOU_KER_SIZE]

        question_convs = question_convs1
        # question_convs.extend(question_convs2)
        self.question_convs = nn.ModuleList(question_convs)

        def forward(self, question):
            # permute 的作用是交换维度，因为词嵌入的维度200要作为后面conv1的输入的channel，所以第二和三维交换
            x = [question_conv(question.permute(0, 2, 1))for question_conv in self.question_convs]

class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x

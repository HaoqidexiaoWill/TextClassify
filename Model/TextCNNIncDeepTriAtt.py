import torch
import torch.nn as nn
import copy
from Model.BasicModel import KMaxPooling1D,TripleAttentionHighWay,ScaledDotProductAttention
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
        # self.triattention = ScaledDotProductAttention()
        self.triattention = TripleAttentionHighWay(self.channel_size)


    def forward(self,input):
        hist, utt, resp = input
        hist_shortcut = self.maxpool(hist)
        hist = self.conv(hist_shortcut)
        hist = hist + hist_shortcut
        hist = hist.permute(0, 2, 1)

        utt_shortcut = self.maxpool(utt)
        utt = self.conv(utt_shortcut)
        utt = utt + utt_shortcut
        utt = utt.permute(0, 2, 1)

        resp_shortcut = self.maxpool(resp)
        resp = self.conv(resp_shortcut)
        resp = resp + resp_shortcut
        resp = resp.permute(0, 2, 1)

        # print(hist.size())
        # his_att = self.triattention(hist,utt,utt)
        # utt_att = self.triattention(utt,resp,resp)
        # resp_att = self.triattention(resp,utt,utt)
        his_att,utt_att,resp_att = self.triattention(hist,utt,resp)


        return his_att.permute(0, 2, 1),utt_att.permute(0, 2, 1),resp_att.permute(0, 2, 1)



class TextCNNIncDeepTriAtt(nn.Module):
    def __init__(self, input_dim, output_dim, max_len,batch_size):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.SENT_LEN = max_len
        self.BATCH_SIZE = batch_size
        self.NUM_ID_FEATURE_MAP = 512
        self.SIN_KER_SIZE = [1, 1, 3, 3]                      # single convolution kernel

        self.DOU_KER_SIZE = [(1, 3), (3, 5), (3, 3), (5, 5)]  # double convolution kernel

        self.k_maxpooling = KMaxPooling1D(max_len)
        self.triattention = TripleAttentionHighWay(self.output_dim)
        convs_single = [nn.Sequential(
                nn.Conv1d(in_channels=self.input_dim,
                          out_channels=self.output_dim,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(self.output_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=(self.SENT_LEN - kernel_size + 1))
            )for kernel_size in self.SIN_KER_SIZE]


        convs_double = [nn.Sequential(
                nn.Conv1d(in_channels=self.input_dim,
                          out_channels=self.output_dim,
                          kernel_size=kernel_size[0]),
                nn.BatchNorm1d(self.output_dim),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=self.output_dim,
                          out_channels=self.output_dim,
                          kernel_size=kernel_size[1]),
                nn.BatchNorm1d(self.output_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=(self.SENT_LEN - kernel_size[0] - kernel_size[1] + 2))
            )for kernel_size in self.DOU_KER_SIZE]

        convs_ = convs_single
        convs_.extend(convs_double)
        self.convs_hist = nn.ModuleList(convs_)
        self.convs_utt = copy.deepcopy(nn.ModuleList(convs_))
        self.convs_resp = copy.deepcopy(nn.ModuleList(convs_))

        self.seq_lenght = len(self.DOU_KER_SIZE) + len(self.SIN_KER_SIZE)

        resnet_block_list = []
        while (self.seq_lenght > 2):
            resnet_block_list.append(ResnetBlock(self.NUM_ID_FEATURE_MAP))
            self.seq_lenght = self.seq_lenght // 2

        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.change_dim_conv  = nn.Conv1d(self.output_dim, self.NUM_ID_FEATURE_MAP, kernel_size=1, stride=1)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.output_dim),
            nn.ReLU(),
            nn.Conv1d(self.output_dim, self.NUM_ID_FEATURE_MAP,kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.NUM_ID_FEATURE_MAP),
            nn.ReLU(),
            nn.Conv1d(self.NUM_ID_FEATURE_MAP, self.NUM_ID_FEATURE_MAP,kernel_size=3, padding=1)
        )


    def forward(self, history,utterance,response):
        # permute 的作用是交换维度，因为词嵌入的维度200要作为后面conv1的输入的channel，所以第二和三维交换
        hist_list = [history_conv(history.permute(0, 2, 1)) for history_conv in self.convs_hist]
        hist_cat = torch.cat(hist_list, dim=2)

        utt_list = [utterance_conv(history.permute(0, 2, 1)) for utterance_conv in self.convs_utt]
        utt_cat = torch.cat(utt_list, dim=2)

        resp_list = [response_conv(history.permute(0, 2, 1)) for response_conv in self.convs_resp]
        resp_cat = torch.cat(resp_list, dim=2)

        hist_cat_resnet = hist_cat
        hist_cat_resnet = self.change_dim_conv(hist_cat_resnet)
        hist_cat = self.conv(hist_cat)
        hist_cat = hist_cat+hist_cat_resnet
        hist_cat = hist_cat.permute(0, 2, 1)

        utt_cat_resnet = utt_cat
        utt_cat_resnet = self.change_dim_conv(utt_cat_resnet)
        utt_cat = self.conv(utt_cat)
        utt_cat = utt_cat+utt_cat_resnet
        utt_cat = utt_cat.permute(0, 2, 1)

        resp_cat_resnet = resp_cat
        resp_cat_resnet = self.change_dim_conv(resp_cat_resnet)
        resp_cat = self.conv(resp_cat)
        resp_cat = resp_cat+resp_cat_resnet
        resp_cat = resp_cat.permute(0, 2, 1)


        his_att1,utt_att1,resp_att1 = self.triattention(hist_cat,utt_cat,resp_cat)
        hist = hist_cat+his_att1
        utt = utt_cat+utt_att1
        resp = resp_cat+resp_att1

        hist_res,utt_res,resp_res = self.resnet_layer((hist.permute(0, 2, 1),utt.permute(0, 2, 1),resp.permute(0, 2, 1)))

        hist_res = hist_res.contiguous().view(history.size(0), -1)
        utt_res = utt_res.contiguous().view(utterance.size(0), -1)
        resp_res = resp_res.contiguous().view(response.size(0),-1)
        return hist_res,utt_res,resp_res
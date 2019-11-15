import torch
import torch.nn as nn
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


class TextCNNIncDeep(nn.Module):
    def __init__(self, input_dim, output_dim, max_len,batch_size):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.SENT_LEN = max_len
        self.BATCH_SIZE = batch_size
        self.NUM_ID_FEATURE_MAP = 512
        self.SIN_KER_SIZE = [1, 1, 3, 3]                      # single convolution kernel
        # self.SIN_KER_SIZE = [1,2,3,4,5,6,7,8,9,10]                      # single convolution kernel

        self.DOU_KER_SIZE = [(1, 3), (3, 5), (3, 3), (5, 5)]  # double convolution kernel
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
        self.convs = nn.ModuleList(convs_)

        self.seq_lenght = len(self.DOU_KER_SIZE) + len(self.SIN_KER_SIZE)
        self.change_dim_conv  = nn.Conv1d(self.output_dim, self.NUM_ID_FEATURE_MAP, kernel_size=1, stride=1)


        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.output_dim),
            nn.ReLU(),
            nn.Conv1d(self.output_dim, self.NUM_ID_FEATURE_MAP,kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.NUM_ID_FEATURE_MAP),
            nn.ReLU(),
            nn.Conv1d(self.NUM_ID_FEATURE_MAP, self.NUM_ID_FEATURE_MAP,kernel_size=3, padding=1)
        )
        resnet_block_list = []
        while (self.seq_lenght > 2):
            resnet_block_list.append(ResnetBlock(self.NUM_ID_FEATURE_MAP))
            self.seq_lenght = self.seq_lenght // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        # self.fc = nn.Sequential(
        #     nn.Linear(opt.NUM_ID_FEATURE_MAP*self.num_seq, opt.NUM_CLASSES),
        #     nn.BatchNorm1d(opt.NUM_CLASSES),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(opt.NUM_CLASSES, opt.NUM_CLASSES)
        # )

    def forward(self, question):
        # permute 的作用是交换维度，因为词嵌入的维度200要作为后面conv1的输入的channel，所以第二和三维交换
        x = [question_conv(question.permute(0, 2, 1))
             for question_conv in self.convs]
        x = torch.cat(x, dim=2)
        xp = x
        print(xp.size())
        exit()
        xp = self.change_dim_conv(xp)
        x = self.conv(x)
        x = x+xp
        x = self.resnet_layer(x)
        x = x.view(question.size(0), -1)
        # print(x.size())
        # exit()
        # x = self.fc(x)
        return x
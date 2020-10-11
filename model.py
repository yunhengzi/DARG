
import torch.nn as nn
import torch.nn
from torch.autograd import Variable
import torch.nn.functional as F


class DANet(nn.Module):
    def __init__(self, n_feats=32):
        super(DANet, self).__init__()

        kernel_size = 3
        self.n_feats = n_feats
        pad_1 = int(2 * (kernel_size - 1) / 2)
        pad_2 = int(5 * (kernel_size - 1) / 2)
        self.conv_1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=1, dilation=1)
        self.conv_2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=pad_1, dilation=2)
        self.conv_3 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=pad_2, dilation=5)
        self.PR = nn.PReLU()

        self.conv_11 = nn.Conv2d(n_feats * 2, n_feats, 1)
        self.Ne_conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1, groups=n_feats),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        res_x = x
        channel_out = self.Ne_conv(x)
        x1 = self.conv_1(x)
        x2 = self.PR(self.conv_2(x))
        x3 = self.PR(self.conv_3(x))
        a = torch.cat([x1, x2], 1)
        b = self.PR(self.conv_11(a))
        c = torch.cat([b, x3], 1)
        d = self.conv_11(c)

        output = d * channel_out
        output = output + res_x

        return output


class RG(nn.Module):
    def __init__(self, input_channel=32):
        super(RG, self).__init__()
        ksize = 3
        self.block = nn.Sequential(
            DANet(n_feats=32),
            nn.PReLU(),
            DANet(n_feats=32),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, ksize, padding=ksize // 2, stride=1, bias=False),
            nn.PReLU())

    def forward(self, x):
        res = x
        x2 = self.block(x)
        x3 = self.conv2(x2)
        out = res + x3
        return out


class NET(nn.Module):
    def __init__(self, input_channel=32, use_GPU=True):
        super(NET, self).__init__()

        n_feats = 3
        self.use_GPU = use_GPU
        ksize = 3
        self.PR = nn.PReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_feats, input_channel, ksize, padding=ksize // 2, stride=1, bias=False),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, ksize, padding=ksize // 2, stride=1, bias=False),
            nn.PReLU(),
        )
        self.block1 = nn.Sequential(RG(input_channel=32))
        self.block2 = nn.Sequential(RG(input_channel=32))
        self.block3 = nn.Sequential(RG(input_channel=32))
        self.block4 = nn.Sequential(RG(input_channel=32))
        self.block5 = nn.Sequential(RG(input_channel=32))
        self.conv3 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, ksize, padding=ksize // 2, stride=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(input_channel, input_channel, ksize, padding=ksize // 2, stride=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(input_channel, 3, ksize, padding=ksize // 2, stride=1, bias=False)
        )

    def forward(self, inpiut):
        x = self.conv1(inpiut)
        res = x
        x1 = self.conv2(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)
        x6 = self.block5(x5)
        out = res + x6
        out = self.conv3(out)
        return out

#     def print_network(net):
#         num_params = 0
#         for param in net.parameters():
#             num_params += param.numel()
#         print(net)
#         print('Total number of parameters: %d' % num_params)
#
#
# model = NET(input_channel=32)
# print(model.print_network())
#
#
# def test():
#     net = NET(input_channel=32)
#     fms = net(Variable(torch.randn(1, 3, 32, 32)))
#     for fm in fms:
#         print(fm.size())
#
#
# test()

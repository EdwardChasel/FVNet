# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:56:51 2022

@author: yimen
"""


import torch
from torch import nn
# import FCNN_3D as fn
from networks import FCNN_3D as fn

class ConvBlock_head(nn.Module):
    def __init__(self, sizeP, n_filters_in, n_filters_out, tranNum, normalization='none'):
        super(ConvBlock_head, self).__init__()
        input_channel = n_filters_in
        ops = []
        ops.append(fn.Fconv_3D_PCA(sizeP = sizeP ,inNum = input_channel, outNum = n_filters_out, tranNum=tranNum, inP = None, padding=2, ifIni=1, bias=True))
                  
        if normalization == 'batchnorm':
            ops.append(fn.F_BN_3D(n_filters_out,tranNum=tranNum))
        elif normalization == 'groupnorm':
            ops.append(fn.F_GroupNorm_3D(groups=16, channels=n_filters_out,tranNum=tranNum))
        elif normalization == 'instancenorm':
            ops.append(fn.F_InstanceNorm_3D(channels=n_filters_out,tranNum=tranNum))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class ConvBlock_body(nn.Module):
    def __init__(self, n_stages, sizeP, n_filters_in, n_filters_out, tranNum, normalization='none'):
        super(ConvBlock_body, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(fn.Fconv_3D_PCA(sizeP = sizeP ,inNum = input_channel, outNum = n_filters_out, tranNum=tranNum, inP = None, padding=2, ifIni=0, bias=True))
            if normalization == 'batchnorm':
                ops.append(fn.F_BN_3D(n_filters_out,tranNum=tranNum))
            elif normalization == 'groupnorm':
                ops.append(fn.F_GroupNorm_3D(groups=16, channels=n_filters_out,tranNum=tranNum))
            elif normalization == 'instancenorm':
                ops.append(fn.F_InstanceNorm_3D(channels=n_filters_out,tranNum=tranNum))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x



class ResidualConvBlock(nn.Module):
    #第一层不能用
    def __init__(self, n_stages, sizeP, n_filters_in, n_filters_out, tranNum, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(fn.Fconv_3D_PCA(sizeP = sizeP ,inNum = input_channel, outNum = n_filters_out, tranNum=tranNum, inP = None, padding=2, ifIni=0, bias=True))
            if normalization == 'batchnorm':
                ops.append(fn.F_BN_3D(n_filters_out,tranNum=tranNum))
            elif normalization == 'groupnorm':
                ops.append(fn.F_GroupNorm_3D(groups=16, channels=n_filters_out,tranNum=tranNum))
            elif normalization == 'instancenorm':
                ops.append(fn.F_InstanceNorm_3D(channels=n_filters_out,tranNum=tranNum))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x
    
class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, tranNum, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(fn.Fconv_3D(sizeP = stride+2 ,inNum = n_filters_in, outNum = n_filters_out, tranNum=tranNum, inP = None, padding=1, stride=stride, ifIni=0, bias=True))
            if normalization == 'batchnorm':
                ops.append(fn.F_BN_3D(n_filters_out,tranNum=tranNum))
            elif normalization == 'groupnorm':
                ops.append(fn.F_GroupNorm_3D(groups=16, channels=n_filters_out,tranNum=tranNum))
            elif normalization == 'instancenorm':
                ops.append(fn.F_InstanceNorm_3D(channels=n_filters_out,tranNum=tranNum))
            else:
                assert False
        else:
            ops.append(fn.Fconv_3D(sizeP = stride+2 ,inNum = n_filters_in, outNum = n_filters_out, tranNum=tranNum, inP = None, padding=1, stride=stride, ifIni=0, bias=True))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, tranNum, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(fn.FConvTranspose_3D(sizeP = stride+2,inNum = n_filters_in, outNum = n_filters_out, tranNum=tranNum, inP = None, padding=1, stride=stride, ifIni=0, bias=True))
            if normalization == 'batchnorm':
                ops.append(fn.F_BN_3D(n_filters_out,tranNum=tranNum))
            elif normalization == 'groupnorm':
                ops.append(fn.F_GroupNorm_3D(groups=16, channels=n_filters_out,tranNum=tranNum))
            elif normalization == 'instancenorm':
                ops.append(fn.F_InstanceNorm_3D(channels=n_filters_out,tranNum=tranNum))
            else:
                assert False
        else:
            ops.append(fn.FConvTranspose_3D(sizeP = stride+2,inNum = n_filters_in, outNum = n_filters_out, tranNum=tranNum, inP = None, padding=1, stride=stride, ifIni=0, bias=True))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class FVNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16,tranNum=8, normalization='none', has_dropout=False):
        super(FVNet, self).__init__()
        self.has_dropout = has_dropout
# sizeP=3, n_filters_in=3, n_filters_out=16, tranNum=8, normalization='none'
        self.block_one = ConvBlock_head(5, n_channels, n_filters, tranNum, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, tranNum, normalization=normalization)
# n_stages=2, sizeP=3, n_filters_in=16, n_filters_out=32, tranNum=8, normalization='none'
        self.block_two = ConvBlock_body(2, 5, n_filters * 2, n_filters * 2, tranNum, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, tranNum, normalization=normalization)

        self.block_three = ConvBlock_body(3, 5, n_filters * 4, n_filters * 4, tranNum, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, tranNum, normalization=normalization)

        self.block_four = ConvBlock_body(3, 5, n_filters * 8, n_filters * 8, tranNum, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, tranNum, normalization=normalization)

        self.block_five = ConvBlock_body(3, 5, n_filters * 16, n_filters * 16, tranNum, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, tranNum, normalization=normalization)

        self.block_six = ConvBlock_body(3, 5, n_filters * 8, n_filters * 8, tranNum, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, tranNum, normalization=normalization)

        self.block_seven = ConvBlock_body(3, 5, n_filters * 4, n_filters * 4, tranNum, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, tranNum, normalization=normalization)

        self.block_eight = ConvBlock_body(2, 5, n_filters * 2, n_filters * 2, tranNum, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, tranNum, normalization=normalization)

        self.block_nine = ConvBlock_body(1, 5, n_filters, n_filters, tranNum, normalization=normalization)
        self.out_conv = fn.Fconv_1X1X1_out(n_filters, n_classes, tranNum)

        self.dropout = fn.F_Dropout_3D(zero_prob=0.5, tranNum=tranNum, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out

# net = FVNet(n_channels=1, n_classes=2, tranNum=8, normalization='batchnorm', has_dropout=True)
# net = net.cuda()
# block_one = ConvBlock_head(sizeP=5, n_filters_in=3, n_filters_out=16, tranNum=8, normalization='none')
# block_one_dw = DownsamplingConvBlock(n_filters_in=16, n_filters_out=16*2, tranNum=8, normalization='none')
# block_one_up = UpsamplingDeconvBlock(n_filters_in=16*2, n_filters_out=16, tranNum=8, normalization='none')
# # block_one1 = ResidualConvBlock(n_stages=1, sizeP=3, n_filters_in=16, n_filters_out=16, tranNum=8, normalization='none')
# X = torch.randn([1,3,29,29,29])#.cuda()
# X = block_one(X)
# print(X.shape)
# X = block_one_dw(X)
# print(X.shape)
# X = block_one_up(X)
# # print(X.shape)
# # X = block_one1(X)
# print(X.shape)
# block_two = ConvBlock_body(n_stages=2, sizeP=5, n_filters_in=16, n_filters_out=32, tranNum=8, normalization='none')
# X = block_two(X)
# print(X.shape)
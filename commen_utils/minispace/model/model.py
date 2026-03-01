import numpy as np
import torch.nn as nn
import torch
from model.modules import ResidualAttentionNet,conv_block

#########################################
import os
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2
class AVRA_rnn(nn.Module):
    '''
    典型的CNN+LSTM
    A recurrent convolutional neural network to predict either PA, GCA-F or MTA of an input brain volume.
    Combines the CNN part (a residual attention network) w/ the recurrent NN component (LSTM)
    Args:
        input_dims = [h,w,c] - list or tuple containing dimensions of each input slice
    '''
    def __init__(self, input_dims):
        super(AVRA_rnn, self).__init__()
        #CNN處理
        self.features = ResidualAttentionNet()
        # print("self.features",self.features)
        input_size = [1,input_dims[0],input_dims[1]]
        self.l = self.get_flat_fts(input_size, self.features)

        self.hs = 256
#LSTM输入数据格式：输入维度 50，隐层100维，两层 lstm_seq = nn.LSTM(50, 100, num_layers=2)
        self.rnn = nn.LSTM(
            input_size=self.l,
            hidden_size=self.hs,
            num_layers=2,
            batch_first=True,
            bidirectional=False
            )

        f = 1
        if self.rnn.bidirectional:
            f = 2
        self.linear = nn.Linear(self.hs*f,1)

    def get_flat_fts(self, in_size, fts):
        # Calculate output dimensions for feature exctration in each plane (with varying dimensions)
        f = fts(torch.Tensor(torch.ones(1,*in_size)))
        return int(np.prod(f.size()[1:]))

    #CNN的输入数据的形状为(batch_size，height，width，depth)，
    #其中第一维表示图像的batch大小，其他三个维表示图像的各个属性，即高度，宽度和深度。
    #深度就是色彩通道的数量。例如，RGB图像的深度为3，而灰度图像的深度为1。

    #ResNet的输入形状，在PyTorch中，ResNet模型的输入通常是一个四维张量（tensor），形状为 (batch_size, channels, height, width)。
    #其中：batch_size：批次大小，表示一次前向传播中处理的样本数。这个值在训练时可以设置，通常取决于可用内存和计算资源。
    #channels：图像通道数。对于彩色图像，通常是3（对应RGB三个通道）；对于灰度图像，是1。
    def forward(self, x,return_r_out=False):
        batch_size, timesteps, C, H, W = x.size()  # 1，22，1，128，128 C:图像深度即通道数为1
        c_in = x.view(batch_size * timesteps, C, H, W)  # 22，1，128，128
        c_out = self.features(c_in) #22，512  图片处理
        r_in = c_out.view(batch_size, timesteps, -1)#1,22，512

        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out_last = self.linear(r_out[:, -1, :])#1
        # print("r_out_last",r_out_last)
        if return_r_out: # features extracted from Residual Attention network, input to LSTM
            return r_out_last,r_in.view(batch_size,-1)
        else:
            return r_out_last

class VGG_bl(nn.Module): 
    ''' 
    VGG16_bl model used as baseline comparison in paper
    '''
    def __init__(self, input_dims):
        super().__init__()
        x,y,z = input_dims
        self.num_filters = [64,128,256,512,512]
        self.convxd = nn.Conv2d
        self.pooling = nn.MaxPool2d
        self.norm = nn.BatchNorm2d
        self.relu = nn.LeakyReLU
        
        self.features = nn.Sequential(conv_block(z,self.num_filters[0], False,self.convxd ,self.norm, self.pooling,relu=self.relu),
                                      conv_block(self.num_filters[0],self.num_filters[1], False,self.convxd ,self.norm, self.pooling,relu=self.relu),
                                      conv_block(self.num_filters[1],self.num_filters[2], True,self.convxd ,self.norm, self.pooling,relu=self.relu),
                                      conv_block(self.num_filters[2],self.num_filters[3], True,self.convxd ,self.norm, self.pooling,relu=self.relu),
                                      conv_block(self.num_filters[3],self.num_filters[4], True,self.convxd ,self.norm, self.pooling,relu=self.relu),
                                      )
        
        a = (x//(2**np.shape(self.num_filters)[0]))*(y//(2**np.shape(self.num_filters)[0])) *self.num_filters[-1]
        a = int(a)
        N= 4096
        
        self.fc1 = nn.Sequential(
            nn.Linear(a,N),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(N, 1))
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x= self.fc1(x)
        return x
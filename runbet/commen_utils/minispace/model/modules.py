#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual Attention Network module - feature extractor in AVRA
"""
import torch.nn as nn
import torch.nn.functional as F

########################################
import os
import random
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2

################################################
class FeatureExtractor(nn.Module):
    def __init__(self,submodule,extracted_layer):
        super(FeatureExtractor,self).__init__()
        self.submodule=submodule
        self.extracted_layer=extracted_layer
    def forward(self, x):
        outputs=[]
        for name,module in self.submodule._modules.items():
            x=module(x) #更新迭代x
            outputs.append(x)  #保存每一层的结果
            # if name in self.extracted_layer:
            #     outputs.append(x)
        return outputs

# extracted_layer=['0','2','3','5']
extracted_layer= None

def Feature_visual(outputs):
    moudlename = str(random.randint(100000,999999))
    for i in range(len(outputs)):#经历多少步处理，当前共12步；self.features=nn.Sequential( conv1,maxpool, resblock1,attention_module1,resblock2,attention_module2,resblock3,attention_module3,resblock4,resblock5,resblock6,        avgpoolblock
        stepSavPth = "./ProgResults/"+moudlename+"/step_"+str(i)

        if not os.path.exists(stepSavPth):
            os.makedirs(stepSavPth)
        out=outputs[i].data.squeeze().numpy()
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.35, hspace=0.1)

        if out.ndim ==3:  #单张图
            for count in range(out.shape[0]):
                feature_img = out[count, :, :].squeeze()   #选择第一个特征图进行可视化
                feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
                picture_num = math.ceil(math.sqrt(out.shape[0]+1))
                plt.subplot(picture_num, picture_num, count + 1)
                plt.axis('off')
                plt.imshow(feature_img)
                # plt.imshow(feature_img, cmap="viridis")
                plt.title(f"3dChnel {count + 1}", fontsize=10)

            fig.savefig(stepSavPth +".jpg", dpi=100)
            # fig.clf()
            # plt.close()
                # plt.imsave(stepSavPth +"/3channel_"+str(count)+".png", feature_img)
        elif out.ndim==4:  #多张图
            for count in range(out.shape[0]):#当前图
                slice = count
                for count1 in range(out.shape[1]):#当前通道
                    print("layer: ",stepSavPth,"slice: ",slice,"Channel: ",count1)
                    feature_img = out[slice,count1, :, :].squeeze()   #选择第一个特征图进行可视化
                    feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
                    picture_num = math.ceil(math.sqrt(out.shape[1]+1))
                    plt.subplot(picture_num, picture_num, count1 + 1)
                    plt.imshow(feature_img)
                    # plt.imshow(feature_img, cmap="viridis")
                    plt.title(f"4dChnel {count1 + 1}", fontsize=10)
                    plt.axis('off')
                fig.savefig(stepSavPth +"/slice_"+str(slice)+".jpg", dpi=100)
        fig.clf()
        plt.close()

                    # plt.imsave(stepSavPth +"/slice_"+str(slice)+"_channel_"+str(count1)+".png", feature_img)


            # plt.savefig(feature_img,)
            # plt.imshow(feature_img,cmap='gray')
            # plt.show()
########################################


class ResidualAttentionNet(nn.Module):
    '''
    Convolutional part of AVRA, that inputs a single 2D MRI slice and outputs a 
    flattened vector with relevant features extracted.
    
    Architecture based on "Residual Attention Network for Image Classification" by Wang et al. (2017)
    https://arxiv.org/abs/1704.06904
    '''
    def __init__(self, z=1):
        super(ResidualAttentionNet, self).__init__()
        
        # number of output filters from each block
        num_filters = [8,16,32,64,128]

        k=0 # block number counter


        ### 每一层处理步骤
        ##开始阶段
        # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1)
        conv1 = nn.Sequential(
            nn.Conv2d(z, num_filters[k], kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(num_filters[k]),
            nn.ReLU(inplace=True)
        )#卷积处理

        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #池化层

        resblock1 = ResidualModule(num_filters[k], num_filters[k+1])#循环锁定
        k+=1
        ##注意力循环处理阶段
        attention_module1 = AttentionModule(num_filters[k], num_filters[k], stage=1)

        resblock2 = ResidualModule(num_filters[k], num_filters[k+1], stride=2)
        k+=1
        attention_module2 = AttentionModule(num_filters[k], num_filters[k], stage=2)

        resblock3 = ResidualModule(num_filters[k], num_filters[k+1], stride=2)
        k+=1
        attention_module3 = AttentionModule(num_filters[k], num_filters[k], stage=3)

        resblock4 = ResidualModule(num_filters[k], num_filters[k+1], stride=2)
        k+=1
        resblock5= ResidualModule(num_filters[k], num_filters[k])
        resblock6 = ResidualModule(num_filters[k], num_filters[k])
        avgpoolblock = nn.Sequential(
            nn.BatchNorm2d(num_filters[k]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=1)
        )

        self.features=nn.Sequential(
                conv1,maxpool,
                resblock1,attention_module1,
                resblock2,attention_module2,
                resblock3,attention_module3,
                resblock4,resblock5,resblock6,
                avgpoolblock
                      )
        
    def forward(self, x):
        # extracted_layers = []  #yang
        # extract_feature = FeatureExtractor(self.features,extracted_layers)(x)
        # Feature_visual(extract_feature)
        out = self.features(x)
        # print("out",out)
        
        # Flatten before passing to RNN 
        out = out.view(out.size(0), -1)
        # print("out",out)#yang
        return out


class ResidualModule(nn.Module):
    '''
    A residual module, used in the Residual Attention Network.
    '''
    def __init__(self, inplanes, planes, stride=1):
        super(ResidualModule, self).__init__()
        
        planes_4 = int(planes/4) # bottlenecking
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inplanes,planes_4, kernel_size=1, stride=1, bias = False)
        
        self.bn2 = nn.BatchNorm2d(planes_4)
        self.relu2 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(planes_4, planes_4, kernel_size=3, stride=stride, padding = 1, bias = False)
        
        self.bn3 = nn.BatchNorm2d(planes_4)
        self.relu3 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(planes_4, planes, kernel_size=1, stride=1, bias = False)
        
        self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias = False)
        # downsampling?
        self.downsample = (self.inplanes != self.planes) or (self.stride !=1 )
    def forward(self, x):

        residual = x
        out = self.bn1(x)
        out1 = self.relu1(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.conv4(out1)
        out += residual
        return out
    
class AttentionModule(nn.Module):
    '''
    Code for generating Attention Module stage 1, 2, or 3
    '''
    def __init__(self, in_planes, out_planes, stage=1):
        super(AttentionModule, self).__init__()
        # p=1, r=1,t=2, as defined in original paper by Wang et al.
        self.stage=stage # 1,2,3, TODO: assert
        
        self.res1 = ResidualModule(in_planes, out_planes)

        self.trunk_branch = nn.Sequential(# i.e. "t=2"
            ResidualModule(in_planes, out_planes),
            ResidualModule(in_planes, out_planes)
         )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.stage<3:
            self.block1 = ResidualModule(in_planes, out_planes)
            self.skip1 = ResidualModule(in_planes, out_planes)
            self.block5 = ResidualModule(in_planes, out_planes)
            if self.stage==1:
                self.block2 = ResidualModule(in_planes, out_planes)
                self.skip2 = ResidualModule(in_planes, out_planes)
            
                self.block4 = ResidualModule(in_planes, out_planes)

        self.block3 = nn.Sequential( # middle block, included in all stages
            ResidualModule(in_planes, out_planes),
            ResidualModule(in_planes, out_planes)
        )
  
        self.block_sigmoid = nn.Sequential( # conv1x1 + sigmoid
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes , kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )

        self.block6 = ResidualModule(in_planes, out_planes)
    def upsample(self,x):
        # upsample tensor with a factor 2
        return F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
    def forward(self, x):
        x = self.res1(x) # H
        trunk_branch = self.trunk_branch(x) # H

        # SOFT MASK BRANCH
        if self.stage<3:
            # downsample 1
            x = self.maxpool(x) # H/2        
            x = self.block1(x)# H/2
            skip1 = self.skip1(x) # H/2, passes through 1 res unit        
            if self.stage==1:
                # downsample 2
                x = self.maxpool(x) # H/4
                x = self.block2(x) # H/4
                skip2 = self.skip2(x) # H/4, passes through 1 res unit
        
        # downsample 3
        x = self.maxpool(x) # H/8
        x = self.block3(x) # H/8
        
        if self.stage==1:
            # upsample 1
            x = self.upsample(x)
            x = x + skip2 # H/4
            x = self.block4(x) # H/4
        if self.stage<3:
            # upsample 2
            x = self.upsample(x)
            x = x + skip1 # H/2
            x = self.block5(x) # H/2
        
        # upsample 3
        x = self.upsample(x)
        mask = self.block_sigmoid(x) # H
        
        # merging trunk branc and soft mask branch
        x = (1 + mask) * trunk_branch # H
        x = x + trunk_branch # H
                
        out_last = self.block6(x) # H
        # print("out_last:",type(out_last),out_last)#yang
        # print("out_last:",type(out_last))#yang
        return out_last

def conv_block(in_planes, out_planes, bigblock,convxd,norm,pooling,fs=3,stride=1,relu=nn.ReLU):
    # conv3->bn3->relu->conv3->bn3->relu->maxpool3
    if bigblock:
        block = nn.Sequential(
            convxd(in_planes, out_planes, fs, 1, int(fs/2)),
            #norm(out_planes),
            relu(True),
            norm(out_planes),
            convxd(out_planes, out_planes, fs, 1, int(fs/2)),
            #norm(out_planes),
            relu(True),
            norm(out_planes),
            convxd(out_planes, out_planes, fs, stride, int(fs/2)),
            #norm(out_planes),
            relu(True),
            norm(out_planes),
            pooling(2, 2)
            )
    else:
        block = nn.Sequential(
            convxd(in_planes, out_planes, fs, 1, int(fs/2)),
            #norm(out_planes),
            relu(True),
            norm(out_planes),
            convxd(out_planes, out_planes, fs, stride, int(fs/2)),
            #norm(out_planes),
            relu(True),
            norm(out_planes),
            pooling(2, 2)
            )
        
    return block

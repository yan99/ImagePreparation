import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import utils

class Intensity_MFG(nn.Module):
    def __init__(self, ch_n = 3):
        super(Intensity_MFG, self).__init__()
        # filter size: [96, 48, 1]
        self.cnn = nn.Sequential(
            # 1st layer
            nn.Conv2d(3, 96, 9, padding = 4, groups = 1),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            # 2nd layer
            nn.Conv2d(96, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            # 3rd layer
            nn.Conv2d(48, 1, 5, padding = 2),
            nn.BatchNorm2d(1)
        )
    def forward(self, rgb):
        out = self.cnn(rgb)
        return out
    
class MultiDepth_MFG(nn.Module):
    def __init__(self, ch_n = 4, pretrain = False):
        super(MultiDepth_MFG, self).__init__()
        # filter size: [96, 48, 24, 1]
        self.ch_n = ch_n
        self.scaling_branch_1 = nn.Sequential(
            # 1st layer
            nn.Conv2d(4, 96, 11, padding = 5, groups = 4),
            nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)),
            nn.ConvTranspose2d(96, 96, 4, stride = 2, padding = 1),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            # 2nd layer
            nn.Conv2d(96, 48, 7, padding = 3),
            nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)),
            nn.ConvTranspose2d(48, 48, 4, stride = 2, padding = 1),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            # 3rd layer
            nn.Conv2d(48, 24, 1),
            nn.BatchNorm2d(24)
        )
        self.relu = nn.ReLU(True)
        self.scaling_brach_2_pretrain = nn.Sequential(
            # Pretrain layer 1
            nn.Conv2d(24, 1, 5, padding = 2),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )
        # Local Branch
        # 1st layer, depth map input
        self.lb_conv1 =  nn.Sequential(
            nn.Conv2d(1, 96, 9, padding = 4),
            nn.BatchNorm2d(96),
            nn.ReLU(True)
        )
        # 2nd & 3rd layer, multi-scale depth + depth input
        self.lb_conv2_pretrain =  nn.Sequential(
            nn.Conv2d(96+1, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            nn.Conv2d(48, 1, 5, padding = 2),
            nn.BatchNorm2d(1)
        )
        # 2nd & 3rd layer, multi-scale depth + depth input
        self.lb_conv2 =  nn.Sequential(
            nn.Conv2d(96+24, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            nn.Conv2d(48, 1, 5, padding = 2),
            nn.BatchNorm2d(1)
        )
        
    def forward(self, rgb, depth, pretrain):
        x = torch.cat((rgb,depth), 1)
        scaling_branch = self.scaling_branch_1(x)
        if(pretrain):
            scaling_branch = self.relu(scaling_branch)
            scaling_branch = self.scaling_brach_2_pretrain(scaling_branch)
        
        local_branch = self.lb_conv1(depth)
        depth_mutual_feature = torch.cat((local_branch, scaling_branch), 1)
        if(pretrain):
            depth_mutual_feature = self.lb_conv2_pretrain(depth_mutual_feature)
        else:
            depth_mutual_feature = self.lb_conv2(depth_mutual_feature)
        return depth_mutual_feature
    
class MSMF_CNN(nn.Module):
    def __init__(self, pretrain):
        super(MSMF_CNN, self).__init__()
        # filter size: [128, 64, 1]
        self.intensity_MFG = Intensity_MFG(ch_n=3)
        self.multiDepth_MFG = MultiDepth_MFG(ch_n=4,pretrain=pretrain)
        self.cnn = nn.Sequential(
            # 1st layer
            nn.Conv2d(2, 128, 9, padding = 4),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 2nd layer
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 3rd layer
            nn.Conv2d(64, 1, 5, padding = 2),
            nn.BatchNorm2d(1)
        )

    def forward(self, rgb, depth, pretrain=False):
        intensity_feature = self.intensity_MFG(rgb)
        depth_mutual_feature = self.multiDepth_MFG(rgb, depth, pretrain)
        input = torch.cat((intensity_feature, depth_mutual_feature), 1)
        out = self.cnn(input)
        return out
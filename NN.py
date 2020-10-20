# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 01:22:10 2020

@author: harry
"""
import torch.nn as nn

class Network(nn.Module):

    def __init__(self):
        
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5) 
        #first in channel chosen to be 1 as images in grayscale thus channel is 1
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
        #Convolution
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        #12
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        #10 in output channel chosen as there are 10 classes of objects within the fa
        #shion data set
    
    def forward(self,t):
       
    
        
        return t
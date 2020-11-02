 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:32:42 2020

@author: harry
"""
import torch
import torch.nn as nn
import torch.nn.functional as F





class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, t):
        return self.double_conv(t)
                

class Down(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
            )
    
    def forward(self, t):
        return self.maxpool_conv(t)
    


class UNN(nn.Module):
    
    def __init__(self, n_channels):
        super(UNN, self).__init__() 
        
        
        self.inc = DoubleConv(n_channels,64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128,256)
        self.down3 = Down(256, 512)
        
        
    
        self.down4 = Down(512, 1024)
   
    
   
    def forward(self, t):
        t1= self.inc(t)
        t2 = self.down1(t1)
        t3 = self.down2(t2)
        t4 = self.down3(t3)
        t5 = self.down4(t4)
        t=t5
        
        return t
    
    
test = torch.ones(1,3,572,572)

out = UNN(test)

        
    
    

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:05:47 2020

@author: harry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from PIL import Image

torch.set_grad_enabled(True)
torch.set_printoptions(linewidth=120) 



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
    
class Up(nn.Module):
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
       
        """Performs a bilinear upscaling of the imported tensor
        This process increases the tensor's width and height by 2 """
        
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
        
    
    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        """"Saftey code, used as a precaution to ensure the height & width 
            of x1 is equal to x2's """
        
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)
    
    
class Out(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x):
        return self.conv(x)
         
              
class UNN(nn.Module):
    
    def __init__(self, n_channels,n_classes, bilinear = True):
        super(UNN, self).__init__() 
        
        self.n_channels = n_channels
        self.n_classes = n_classes 
        self.bilinear = bilinear
        
        
        
        self.inc = DoubleConv(n_channels,64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128,256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // 2)
        
        #Complete down method, each step perfoms a double convolution
        
        self.up1 = Up(1024, 512 // 2, bilinear)
        self.up2 = Up(512, 256 // 2, bilinear)
        self.up3 = Up(256, 128 // 2, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = Out(64, n_classes)
   
    
   
    def forward(self, t):
        t1= self.inc(t)
        t2 = self.down1(t1)
        t3 = self.down2(t2)
        t4 = self.down3(t3)
        t5 = self.down4(t4)
        print(t5.size())
        
        t = self.up1(t5, t4)
        print("up1",t.size())
        t = self.up2(t, t3)
        print("up2",t.size())
        t = self.up3(t, t2)
        print("up3",t.size())
        t = self.up4(t, t1)
        print("up4",t.size())
        
        final = self.outc(t)
        print(final.size())
        return final
    
def loadtiffs(file_name):
    img = Image.open(file_name)
    
    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
    
    for i in range(img.n_frames):
        img.seek(i)
        imgArray[:,:,i] = np.asarray(img)
    img.close()
    return(imgArray, img.n_frames)

def get_num_correct(preds,labels):
    
    return preds.argmax(dim=1).eq(labels).sum().item()

epi_data, frames = loadtiffs(
'/Users/harry/Documents/WORK/University/Physics/Third Year/PHY341-project/Training Images/epi.tif')

phase_data, frames = loadtiffs(
'/Users/harry/Documents/WORK/University/Physics/Third Year/PHY341-project/Training Images/phase.tif')



for i in range(frames):
    
  
    norm = (phase_data[:,:,i] - np.min(phase_data[:,:,i])) / (np.max(phase_data[:,:,i])-np.min(phase_data[:,:,i]))
    norm_t = torch.from_numpy((norm[np.newaxis,np.newaxis]))
    

    thresh = epi_data[:,:,i] > 5000
    thresh = thresh.astype(int)
    thresh_t = torch.from_numpy((thresh[np.newaxis,np.newaxis]))
    
    if i == 0:
        norm_temp = norm_t
        thresh_temp = thresh_t
        
    else:
        phase_t = torch.cat((norm_temp,norm_t),0)
        norm_temp = phase_t
        
        epi_t = torch.cat((thresh_temp,thresh_t),0)
        thresh_temp = epi_t
        
    
epi_t = thresh_temp
phase_t = norm_temp

#Normalises phase data, thresholds epi data and transforms np array into 
#a pytorch tensor




i = 0

U_net = UNN(1,1)
optimizer = optim.Adam(U_net.parameters(), lr=0.01)

for epoch in range(10):
    total_loss = 0
    total_correct = 0
    j= 0
    
    
    preds = U_net(phase_t)
    
    loss = F.cross_entropy(preds, epi_t) #Calculate Loss
    
    optimizer.zero_grad()
    
    loss.backward() #Calculates gradients difference between suc and loss
    optimizer.step() #Update weights
    
    total_loss += loss.item()
    total_correct += get_num_correct(preds,epi_t)
    j = j + 1
    print(j)
    

    print("epoch:",i,"total_correct:", total_correct, "loss:", total_loss)
    i = i +1



























test = torch.ones([1,1,572,572])

out = U_net(test)
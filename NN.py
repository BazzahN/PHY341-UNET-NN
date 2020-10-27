# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 01:22:10 2020

@author: harry
"""
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_grad_enabled(True)
torch.set_printoptions(linewidth=120)


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
       # Input Layer
       t=t
       
       # COnv Layer
       t = self.conv1(t)
       t=F.relu(t)
       t=F.max_pool2d(t, kernel_size=2, stride=2)
       
       t = self.conv2(t)
       t=F.relu(t)
       t=F.max_pool2d(t, kernel_size=2, stride=2)
       
       #Linear Layer
       t = t.reshape(-1,12*4*4) #Flattens tensor
       t = self.fc1(t)
       t = F.relu(t)
       
       t = self.fc2(t)
       t = F.relu(t)
       
       
       #Output Layer
       t = self.out(t)
       #t = F.softmax(t,dim=1)
       
       return t
    

def get_num_correct(preds,labels):
    
    return preds.argmax(dim=1).eq(labels).sum().item()



network=Network() #Assigning network class

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
    ) # Data Extraction

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)#Data Loading
optimizer = optim.Adam(network.parameters(), lr=0.01)


batch = next(iter(train_loader)) #Assigning data to variable

i = 0

for epoch in range(5):
    total_loss = 0
    total_correct = 0
    
    for batch in train_loader:
        images,labels = batch #Storing data in tuple
    
        preds = network(images)
    
        loss = F.cross_entropy(preds,labels) #Calculate Loss
        
        optimizer.zero_grad()
        
        loss.backward() #Calculates gradients difference between suc and loss
        optimizer.step() #Update weights
        
        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)
    
    

    print("epock:",i,"total_correct:", total_correct, "loss:", total_loss)
    i = i +1






















    
   
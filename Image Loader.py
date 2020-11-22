# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:34:45 2020

@author: harry
"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def loadtiffs(file_name):
    img = Image.open(file_name)
    
    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
    
    for i in range(img.n_frames):
        img.seek(i)
        imgArray[:,:,i] = np.asarray(img)
    img.close()
    return(imgArray, img.n_frames,img.size[1], img.size[0])

epi_data, frames, h, w = loadtiffs(
'/Users/harry/Documents/WORK/University/Physics/Third Year/PHY341-project/Training Images/epi.tif')



phase_data, frames, h, w = loadtiffs(
'/Users/harry/Documents/WORK/University/Physics/Third Year/PHY341-project/Training Images/phase.tif')



for i in range(frames):
    
    temp1 = phase_data[:,:,i]
    norm = (temp1 - np.min(temp1)) / (np.max(temp1)-np.min(temp1))
    norm_t = torch.from_numpy((norm[np.newaxis,np.newaxis]))
    
    
    if i == 0:
        norm_temp = norm_t
        
    else:
        third = torch.cat((norm_temp,norm_t),0)
        norm_temp = third

#Normalises phase data and transforms np array into a pytorch tensor



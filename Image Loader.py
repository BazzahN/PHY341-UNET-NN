# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:34:45 2020

@author: harry
"""
import torch

from PIL import Image
import numpy as np


def loadtiffs(file_name):
    img = Image.open(file_name)
    
    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
    
    for i in range(img.n_frames):
        img.seek(i)
        imgArray[:,:,i] = np.asarray(img)
    img.close()
    return(imgArray, img.n_frames)






epi_data, frames = loadtiffs(
'/Users/harry/Documents/WORK/University/Physics/Third Year/PHY341-project/Training Images/epi.tif')

phase_data, frames = loadtiffs(
'/Users/harry/Documents/WORK/University/Physics/Third Year/PHY341-project/Training Images/phase.tif')

norm = []

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





#Normalises phase data, thresholds epi data and transforms np array into a pytorch tensor



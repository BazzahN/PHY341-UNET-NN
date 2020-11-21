# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:34:45 2020

@author: harry
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def loadtiffs(file_name):
    img = Image.open(file_name)
    print('The image is', img.size, 'Pixels.')
    print ('With', img.n_frames, 'frames.')
    
    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
    for i in range(img.n_frames):
        img.seek(i)
        imgArray[:,:,i] = np.asarray(img)
    img.close()
    return(imgArray)

epi_data = loadtiffs('/Users/harry/Documents/WORK/University/Physics/Third Year/PHY341-project/Training Images/epi.tif')

phase_data = loadtiffs('/Users/harry/Documents/WORK/University/Physics/Third Year/PHY341-project/Training Images/phase.tif')

plt.imshow(phase_data[:,:,2])
plt.show()
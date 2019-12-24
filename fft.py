# -*- coding: utf-8 -*-
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift


data = h5py.File('data/data8.mat')['data']
data = np.transpose(data) 
    
# =============================================================================
# #截取部分原始数据，进行可视化并画出频谱图
# =============================================================================

#原始数据
fig, (ax1, ax2) = plt.subplots(figsize=(13, 3),ncols=2)
pos = ax1.imshow(data[500:1500,200:1000], cmap='seismic',vmin=-0.01,vmax = 0.01)
ax1.set_yticklabels([0,500,800,1000,1200,1400,1500])
ax1.set_xticklabels([0,200,'',400,'',600,'',800,'',1000])
cbar = fig.colorbar(pos, ax=ax1,shrink = 0.93,format = '%.3f')

xf= fftshift(fft2(data[500:1500,200:1000]))
xfp = np.log(np.abs(xf)+1)
pos = ax2.imshow(xfp,vmin=0,vmax = 4.5)
ax2.set_yticklabels([])
ax2.set_xticklabels([])
cbar = fig.colorbar(pos, ax=ax2,shrink = 0.93)
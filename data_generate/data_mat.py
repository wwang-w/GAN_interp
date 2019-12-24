# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:56:08 2019

@author: yfs20
"""

import torch
import os 
import sys
import glob
import cv2
import h5py
import random
import math
from torch.utils.data import Dataset

torch.set_default_dtype(torch.float64)

class DownsamplingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        rate: data missing rate, e.g. 0.3
    """
    def __init__(self, xs, rate):
        super(DownsamplingDataset, self).__init__()
        self.xs = xs
        self.rate = rate

    def __getitem__(self, index):
        batch_x = self.xs[index]
        #必须生成为tensor的数据类型,xs[1]为torch型数据
        mask = irregular_mask(batch_x,self.rate)
        batch_y = mask.mul(batch_x)
        return batch_y, batch_x,mask

    def __len__(self):
        return self.xs.size(0)


def irregular_mask(data,rate):
    #非规则缺失mask矩阵
    #data为数据，a为采样率，取值范围为(0,1)；

    n = data.size()[-1]
    #生成Tensor型float单位矩阵
    mask = torch.torch.zeros(data.size(),dtype=torch.float64)
    v = round(n*rate)
    TM = random.sample(range(n),v)
    mask[:,:,TM]=1#按列缺失
    #按行缺失
    # mask[TM,:]=0;    
    return mask


def gen_patches(file_name,patch_size = 64,stride=32):
    # read image
    data = h5py.File(file_name)['data']
    data = np.transpose(data)
    h, w = data.shape
    patches = []

    # extract patches
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            x = data[i:i+patch_size, j:j+patch_size]
            
            #删除全为0的地震数据矩阵
            if sum(sum(x))==0:
                continue
            else:
                patches.append(x)        
                # data aug
                #for k in range(0, aug_times):
                    #x_aug = data_aug(x, mode=np.random.randint(0,8))
                    #patches.append(x_aug)
    return patches                   

def datagenerator(data_dir='G:/WJ/data',patch_size = 128,stride = 32, batch_size = 5,train_data_num = 10000,verbose=True):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.mat')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i],patch_size,stride)
        for patch in patches:    
            data.append(patch)
            if len(data)>=train_data_num:
              data = np.expand_dims(data, axis=3)
              print(str(len(data))+' '+'training data finished')
              return data


        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
#    #uint8，表示变量是无符号整数，范围是0到255.
#    data = np.array(data, dtype='uint8')#生成二维数据   
    data = np.expand_dims(data, axis=3)
    #要求生成的patches的数量是batch的整数倍
        #要求生成的patches的数量是batch的整数倍
#    discard_n = len(data)-len(data)//batch_size*batch_size   
#    data = np.delete(data, range(discard_n), axis=0)
    print(str(len(data))+' '+'training data finished')
    return data
'''
def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, cmap='gray',vmin=-0.1,vmax = 0.1)
    plt.colorbar()
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()
'''

def show(img1,img2,img3):
    plt.figure()  #创建一个名为astronaut的窗口,并设置大小 

    plt.subplot(131)     #将窗口分为两行两列四个子图，则可显示四幅图片
    plt.title('sampling data')   #第一幅图片标题
    plt.imshow(img1, cmap='gray',vmin=-0.1,vmax = 0.1)
    
    plt.subplot(1,3,2)     #第二个子图
    plt.title('original data')   #第二幅图片标题
    plt.imshow(img2, cmap='gray',vmin=-0.1,vmax = 0.1)     #绘制第二幅图片,且为灰度图
    plt.yticks([])     #不显示坐标尺寸

    plt.subplot(1,3,3)     #第三个子图
    plt.title('reconstructed data')   #第三幅图片标题
    plt.imshow(img3, cmap='gray',vmin=-0.1,vmax = 0.1)     #绘制第三幅图片,且为灰度图
    plt.yticks([])  
    
#    plt.subplots_adjust(left=0.125, bottom=0, right=0.90, top=None,
#                wspace=None, hspace=None)
    
    plt.tight_layout(pad=0.1, w_pad=0.4, h_pad=1.0)
    plt.show()


def compare_SNR(real_img,recov_img):
    real_mean = np.mean(real_img)
    tmp1 = real_img - real_mean
    real_var = sum(sum(tmp1*tmp1))

    noise = real_img - recov_img
    noise_mean = np.mean(noise)
    tmp2 = noise - noise_mean
    noise_var = sum(sum(tmp2*tmp2))

    if noise_var ==0 or real_var==0:
      s = 999.99
    else:
      s = 10*math.log(real_var/noise_var,10)
    return s
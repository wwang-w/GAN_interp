# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 19:22:42 2019
@author: wj
"""

from __future__ import print_function
import argparse
import random
import torch
import os, time, datetime
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from  data_generate.patch_extraction import extract_patches_2d,reconstruct_from_patches_2d
import os
import json
import h5py

import model.Unet_WGAN as net
from data_generate.data_mat import * 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='CWGAN', type=str, help='the model name')
    parser.add_argument('--config', default='/home/wj/DL_inter/inter_r/CWGAN_2_0529/generator_config.json', type=str, help='path to generator config .json file')
    parser.add_argument('--set_dir', default='/home/wj/DL_inter/test_mat/data', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['test'], help='directory of test dataset')
    parser.add_argument('--patch_size', default=(128,128), help='patch_size')   
    parser.add_argument('--G', type=int, default=10,help = 'the rate of the data to scale')
    parser.add_argument('--slices', default=64, type=int, help='whether overlap')
    parser.add_argument('--model_dir', default='CWGAN_2_0529', help='directory of the model')
    parser.add_argument('--model_name', default='netG_epoch_150.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='CWGAN_2_0529_results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)



def save_result(result, path):
    #find方法用来找出给定字符串在另一个字符串中的位置,如果返回-1则表示找不到子字符串
    path = path if path.find('.') != -1 else path+'.png'
    #os.path.splitext(“文件路径”) ,分离文件名与扩展名
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        #np.clip(a, a_min, a_max, out=None)截取,超出的部分就把它强置为边界部分。
        vutils.save_image(result, path, normalize=True)
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

    plt.subplot(141)     #将窗口分为两行两列四个子图，则可显示四幅图片
    plt.title('sampling data')   #第一幅图片标题
    plt.imshow(img1, cmap='gray',vmin=-0.1,vmax = 0.1)
    
    plt.subplot(1,4,2)     #第二个子图
    plt.title('original data')   #第二幅图片标题
    plt.imshow(img2, cmap='gray',vmin=-0.1,vmax = 0.1)     #绘制第二幅图片,且为灰度图
    plt.yticks([])     #不显示坐标尺寸

    plt.subplot(1,4,3)     #第三个子图
    plt.title('reconstructed data')   #第三幅图片标题
    plt.imshow(img3, cmap='gray',vmin=-0.1,vmax = 0.1)     #绘制第三幅图片,且为灰度图
    plt.yticks([]) 
    '''
    plt.subplot(1,4,4)     #第三个子图
    plt.title('error data')   #第三幅图片标题
    plt.imshow(img4, cmap='gray',vmin=-0.1,vmax = 0.1)     #绘制第三幅图片,且为灰度图
    plt.yticks([])  
    '''    
#    plt.subplots_adjust(left=0.125, bottom=0, right=0.90, top=None,
#                wspace=None, hspace=None)
#    plt.colorbar()
    plt.tight_layout(pad=0.05, w_pad=0.2, h_pad=1.0)
    plt.show()

if __name__=="__main__":


    torch.set_default_dtype(torch.float64)
    opt = parse_args()
    #open(name,model)name为要打开的文件夹的名字，r表示只读
    with open(opt.config, 'r') as gencfg:
        generator_config = json.loads(gencfg.read())


    imageSize = generator_config["imageSize"]
    inputChannel = generator_config["inputChannel"]
    outputChannel = generator_config["outputChannel"]
    ngf = generator_config["ngf"]
    rate = generator_config["rate"]
    model_dir = opt.model_dir
    print(rate)



    netG = net.G(inputChannel, outputChannel, ngf)

    # load weights
    netG.load_state_dict(torch.load(os.path.join(model_dir,opt.model_name)))

    if torch.cuda.is_available():
        netG = netG.cuda()

    if not os.path.exists(opt.result_dir):
        os.mkdir(opt.result_dir)

    for set_cur in opt.set_names:

        if not os.path.exists(os.path.join(opt.result_dir, set_cur)):
            os.mkdir(os.path.join(opt.result_dir, set_cur))


        snrs = []
        # 随机时mask
        mask_name = 'mask_'+str(opt.rate)+'.txt'
        mask = np.loadtxt(mask_name)

        #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        for im in os.listdir(os.path.join(opt.set_dir, set_cur)):
            if  im.endswith(".mat"):
                #原始图像
                data = h5py.File(os.path.join(opt.set_dir, set_cur, im))['data']
                data = np.transpose(data)
                image = data*opt.G

                '''
                规则采样时mask
                n = image.shape[-1]
                mask = np.zeros(image.shape)
                for i in range(n):
                    if (i+1)%rate==1:
                        mask[:,i]=1
                np.savetxt(os.path.join(opt.result_dir,opt.model +'mask.txt'),mask)
                '''

                #整张缺失数据
                y = image * mask
        

                patch_x = extract_patches_2d(image,opt.patch_size,extraction_step=opt.slices, max_patches=None, random_state=None)
                patch_y = extract_patches_2d(y,opt.patch_size,extraction_step=opt.slices, max_patches=None, random_state=None)
                #用于存储重构后的patches
                re_patches = []


                torch.cuda.synchronize()#测试代码时间
                start_time = time.time()  
                

                for pt in range(len(patch_x)):
                    x = patch_x[pt]
                    y_ = patch_y[pt]
                    y_ = torch.from_numpy(y_).view(1, -1, y_.shape[0], y_.shape[1])

                    y_ = y_.cuda()
                    x_ = netG(y_)  # inference
                    x_ = x_.view(x.shape[0], x.shape[1]) #此变换后x_仍为tensor型数据，而非array数据
                    x_ = x_.cpu()
                    #detach()返回一个新的 从当前图中分离的 Variable。 返回的 Variable 永远不会需要梯度
                    x_ = x_.detach().numpy()#恢复后数据，array型数据
                    re_patches.append(x_)
                re_patches = np.array(re_patches)
                re_image  = reconstruct_from_patches_2d(re_patches,extraction_step = opt.slices, image_size = image.shape)

                re_image = (1-mask)*re_image+mask*image
                torch.cuda.synchronize()#测试代码时间
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

                snr_x_ = compare_SNR(image, re_image)

                if opt.save_result:
                    #os.path.splitext(“文件路径”) ,分离文件名与扩展名
                    name, ext = os.path.splitext(im)
#                    show(y,image,re_image,re_image-image) # show the image
#                    show(y,image,re_image)
                    recovered_image = torch.from_numpy(re_image)
                    real_image = torch.from_numpy(image)
                    downsammple_image = torch.from_numpy(y)

                    vutils.save_image(downsammple_image, '%s/%s/%s_rate_%.2f_downsampling.png' % \
                        (opt.result_dir, set_cur, name,rate), normalize=True) 
                    vutils.save_image(recovered_image, '%s/%s/%s_rate_%.2f_SNR_%.2fdB.png' % \
                        (opt.result_dir, set_cur, name,rate,snr_x_), normalize=True) 
                    vutils.save_image(real_image, '%s/%s/%s_realimage.png' % \
                        (opt.result_dir, set_cur, name), normalize=True) 
                    np.savetxt(os.path.join(opt.result_dir,opt.model+name+'.txt'),re_image)

                print(snr_x_)
                snrs.append(snr_x_)
        snr_avg = np.mean(snrs)
        snrs.append(snr_avg)
        if opt.save_result:
            save_result(np.hstack(snrs), path=os.path.join(opt.result_dir, set_cur, 'results.txt'))
        log('Datset: {0:10s} \n  SNR = {1:2.6f}dB'.format(set_cur,snr_avg))

#    for i in range(opt.nimages):
#        vutils.save_image(fake.data[i, ...].reshape((1, nc, imageSize, imageSize)), os.path.join(opt.output_dir, "generated_%02d.png"%i))

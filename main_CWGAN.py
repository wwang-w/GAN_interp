# -*- coding: utf-8 -*-
# encoding: utf-8
from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import time
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import MultiStepLR
import glob
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json
import re
import model.Unet_WGAN as net
from data_generate.data_mat import * 

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='CWGAN',  help='')
    parser.add_argument('--train_data', default='/data/train', help='path to trn dataset')
    parser.add_argument('--val_data', default='/data/val', help='path to val dataset') 
    parser.add_argument('--Dmodel', default='Unet', help='the name of the D')
    parser.add_argument('--Gmodel', default='Unet', help='the name of the G')
    parser.add_argument('--batchSize', type=int, default=40, help='input batch size')
    parser.add_argument('--stride', type=int, default=32, help='the stride when get patch when function:datagenerator')
    parser.add_argument('--valBatchSize', type=int, default=5, help='input batch size')
    parser.add_argument('--originalSize', type=int, default=128, help='the height / width of the original input image')
    parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the cropped input image to network')
    parser.add_argument('--inputChannel', type=int, default=1, help='size of the input channels')
    parser.add_argument('--outputChannel', type=int, default=1, help='size of the output channels')
    parser.add_argument('--rate', type=float, default=2,help = 'the sampling rate of data')
    parser.add_argument('--G', type=int, default=10,help = 'the rate of the data to scale')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=1, help='number of D iters per each G iter')
    parser.add_argument('--Unet', default=True , help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--epoch',  type=int, default=150,help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lambdaGAN', type=float, default=1, help='lambdaGAN, default=1')
    parser.add_argument('--lambdaIMG', type=float, default=100, help='lambdaIMG, default=100')
    parser.add_argument('--wd', type=float, default=0.0004, help='weight decay, default=0.0004')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--rmsprop', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--display', type=int, default=100 , help='interval for displaying train-logs')
    parser.add_argument('--evalIter', type=int, default=600, help='interval for evauating(generating) images from valDataroot') 
    parser.add_argument('--exp', default=None, help='Where to store samples and models')
    
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available()
    torch.set_default_dtype(torch.float64)
    lambdaGAN = opt.lambdaGAN
    lambdaIMG = opt.lambdaIMG  


    exp_name = opt.model+'_'+str(opt.rate)+'_'+time.strftime('%m%d',time.localtime(time.time()))
    if opt.exp is None:
        opt.exp = exp_name
    os.system('mkdir {0}'.format(opt.exp))

    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    #### seed() 方法改变随机数生成器的种子
    random.seed(opt.manualSeed)
    # torch.manual_seed设定生成随机数的种子，并返回一个torch._C.Generator对象
    torch.manual_seed(opt.manualSeed)
    # 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销。
    cudnn.benchmark = True
    

    # get dataloader
    train_data = datagenerator(data_dir=opt.train_data,patch_size =opt.imageSize,stride = opt.stride,batch_size = opt.batchSize,train_data_num = 10000)
    train_data = torch.from_numpy(train_data.transpose((0, 3, 1, 2)))

    #get val
    val_data = datagenerator(data_dir=opt.val_data,patch_size =128,stride = opt.stride,batch_size = opt.valBatchSize)
    val_data = torch.from_numpy(val_data.transpose((0, 3, 1, 2)))
    val_DDataset = DownsamplingDataset(val_data, opt.rate)
    valDataloader = DataLoader(dataset=val_DDataset, num_workers=4, drop_last=True, batch_size=opt.valBatchSize, shuffle=True)


    # get logger
    trainLogger = open('%s/train.log' % opt.exp, 'w')

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, 'netG_epoch_*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                #findall(pattern, string, flags=0)返回string中所有与pattern相匹配的全部字串，返回形式为数组
                result = re.findall(".*netG_epoch_(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch = max(epochs_exist)
        else:
            initial_epoch = 0
        return initial_epoch



    ngf = opt.ngf
    ndf = opt.ndf
    inputChannel = opt.inputChannel
    outputChannel= opt.outputChannel
    imageSize = opt.imageSize
    criterionMSE = nn.MSELoss(reduce = True,size_average = False)


    # choose models
    if opt.Dmodel == 'dcgan_noBN':
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif opt.Dmodel == 'mlp_G':
        netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
    elif opt.Dmodel== 'Unet':
        netG = net.G(inputChannel+inputChannel, outputChannel, ngf) 



    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize,'inputChannel':inputChannel, 'outputChannel':outputChannel, 'ngf':ngf,"rate":opt.rate, "model_file":exp_name}
    with open(os.path.join(opt.exp, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    
    netG.apply(weights_init)

    if opt.Gmodel == 'mlp_D':
        netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
    elif opt.Gmodel=='Unet':
        netD = net.D(imageSize,inputChannel+outputChannel, ndf)
        netD.apply(weights_init)
    else:
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)


    target= torch.DoubleTensor(opt.batchSize, outputChannel, opt.imageSize, opt.imageSize)
    input = torch.DoubleTensor(opt.batchSize, inputChannel, opt.imageSize, opt.imageSize)
    mask = torch.DoubleTensor(opt.batchSize, inputChannel, opt.imageSize, opt.imageSize)  

    val_target= torch.DoubleTensor(opt.valBatchSize, outputChannel, opt.imageSize, opt.imageSize)
    val_input = torch.DoubleTensor(opt.valBatchSize, inputChannel, opt.imageSize, opt.imageSize)
    val_mask = torch.DoubleTensor(opt.valBatchSize, inputChannel, opt.imageSize, opt.imageSize) 

    if cuda:
      val_target, val_input ,val_mask= val_target.cuda(), val_input.cuda(),val_mask.cuda()    

    one = torch.DoubleTensor([1])
    mone = one * -1


    # get randomly sampled validation images and save it
    val_iter = iter(valDataloader)
    data_val = val_iter.next()  
    val_input_original, val_target_original,val_mask_original = data_val
    val_target_cpu, val_input_cpu ,val_mask_cpu = val_target_original*opt.G, val_input_original*opt.G,val_mask_original
    
    if cuda:
      val_target_cpu, val_input_cpu ,val_mask_cpu= val_target_cpu.cuda(), val_input_cpu.cuda(),val_mask_cpu.cuda()
      target, input,mask = target.cuda(), input.cuda(),mask.cuda()

    val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
    val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
    val_mask.resize_as_(val_mask_cpu).copy_(val_mask_cpu)
    vutils.save_image(val_target_original, '%s/real_target.png' % opt.exp, normalize=True)
    vutils.save_image(val_input_original, '%s/real_input.png' % opt.exp, normalize=True)
    vutils.save_image(val_mask_original, '%s/real_mask.png' % opt.exp, normalize=True) 



    if cuda:
        netD.cuda()
        netG.cuda()
        
        one, mone = one.cuda(), mone.cuda()
        criterionMSE = criterionMSE.cuda()

    # setup optimizer
    if opt.rmsprop:
        optimD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
        optimG = optim.RMSprop(netG.parameters(), lr = opt.lrG)
    else:
        optimD = optim.Adam(netD.parameters(), lr = opt.lrD, betas = (opt.beta1, 0.999), weight_decay=opt.wd)
        optimG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=opt.wd)  

    initial_epoch = findLastCheckpoint(save_dir=opt.exp)-1  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        netG.load_state_dict(torch.load(os.path.join(opt.exp, 'netG_epoch_{0}.pth'.format(initial_epoch))))
        netD.load_state_dict(torch.load(os.path.join(opt.exp, 'netD_epoch_{0}.pth'.format(initial_epoch))))

    gen_iterations = 0  #生成器迭代次数---在epoch循环外被初始化为０
    for epoch in range(initial_epoch,opt.epoch):
        # iter() 函数用来生成迭代器

        train_DDataset = DownsamplingDataset(train_data, opt.rate)
        dataloader = DataLoader(dataset=train_DDataset, num_workers=4, drop_last=True, batch_size=opt.batchSize, shuffle=True)

        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # train the discriminator Diters times
            # 也即是说当在第一个epoch中，将判别器迭代100次，将生成器迭代１次。
            # 然后当gen_iteration>=25时，即生成器迭代了２５次以上时，生成器每迭代一次，判别器迭代默认的５次
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = data_iter.next()
                i += 1

                # train with real
                input_cpu_original ,target_cpu_original,mask_cpu_original = data
                input_cpu,target_cpu,mask_cpu = input_cpu_original*opt.G ,target_cpu_original*opt.G,mask_cpu_original
                
                netD.zero_grad()
                batch_size = target_cpu.size(0)

                if cuda:
                    target_cpu, input_cpu,mask_cpu = target_cpu.cuda(), input_cpu.cuda(),mask_cpu.cuda()

                target.data.resize_as_(target_cpu).copy_(target_cpu)
                input.data.resize_as_(input_cpu).copy_(input_cpu)
                mask.data.resize_as_(mask_cpu).copy_(mask_cpu)

                target = Variable(target)
                input = Variable(input)
                mask = Variable(mask)

                errD_real = netD(torch.cat([target, input], 1))
                errD_real.backward(one)

                # train with fake
                x_hat = netG(torch.cat([input, mask], 1))
                fake = Variable(x_hat.data)
                inputv = fake
                errD_fake = netD(torch.cat([inputv, input], 1))
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                optimD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise

            # compute MSE_loss
            err_MSE_ = criterionMSE(x_hat,target)
            err_MSE = lambdaIMG * err_MSE_
            if lambdaIMG != 0: 
                err_MSE.backward(retain_graph=True) 

            errG_ = netD(torch.cat([x_hat, input], 1))
            errG = lambdaGAN*errG_
            errG.backward(one)
            optimG.step()
            gen_iterations += 1
            
            if i % opt.display==0:
                print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                    % (epoch, opt.epoch, i, len(dataloader), gen_iterations,
                    errD.data.mean(), errG.data.mean(), errD_real.data.mean(), errD_fake.data.mean()))
                sys.stdout.flush()
                trainLogger.write('%d\t%f\t%f\t%f\t%f\n' % \
                                (i, errD.data.mean(), errG.data.mean(), errD_real.data.mean(), errD_fake.mean()))
                #flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区
                trainLogger.flush()


            '''
            # 交叉验证
            if gen_iterations % opt.evalIter == 0:
                val_batch_output = torch.DoubleTensor(val_input.size()).fill_(0) 
                SNR_list = []
                for idx in range(val_input.size(0)):
                    val_tar = val_target[idx,:,:,:]
                    val_tar = val_tar.view(val_tar.shape[1], val_tar.shape[2]).cpu()
                    val_tar = val_tar.detach().numpy().astype(np.float32) #the target to compute SNR

                    single_img = val_input[idx,:,:,:].unsqueeze(0) 
                    val_inputv = Variable(single_img, volatile=True)
                    x_hat_val = netG(val_inputv)
                    x_hat_val_data = x_hat_val[0,:,:,:].data
                    val_batch_output[idx,:,:,:].copy_(x_hat_val[0,:,:,:].data) 

                    x_hat_cpu = x_hat_val_data.view(x_hat_val_data.shape[1], x_hat_val_data.shape[2]).cpu()
                    x_hat_cpu = x_hat_cpu.detach().numpy().astype(np.float32)#the output to compute SNR
                    # compute SNR
                    SNR = compare_SNR(val_tar,x_hat_cpu)
                    SNR_list.append(SNR)
                SNR_mean = np.mean(SNR_list)

                vutils.save_image(val_batch_output, '%s/generated_epoch_%08d_iter%08d_SNR%.2fdB.png' % \
                (opt.exp, epoch, gen_iterations,SNR_mean), normalize=True) 
            '''

        # do checkpointing
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.exp, epoch+1))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.exp, epoch+1))
    trainLogger.close()


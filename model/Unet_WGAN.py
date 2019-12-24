import torch
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn




class D(nn.Module):
    def __init__(self, isize, nc, ndf):
        super(D, self).__init__()

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:relu'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            # main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
            #                nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        output = self.main(input)
            
        output = output.mean(0)
        return output.view(1)


'''
class D(nn.Module):
  def __init__(self, nc, nf):
    super(D, self).__init__()

    main = nn.Sequential()
    # 128
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%sconv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 64
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False,leakyrelu=False, dropout=False))

    # 32
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%sleakyrelu' % name, nn.LeakyReLU(0.2, inplace=False))
    main.add_module('%sconv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    main.add_module('%sbn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%sleakyrelu' % name, nn.LeakyReLU(0.2, inplace=False))
    main.add_module('%sconv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%ssigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output

'''


class G(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    # 第一层卷积的channels数
    super(G, self).__init__()
    ## NOTE: encoder
    # input is 128 x 128
    layer1 = nn.Sequential(
      nn.Conv2d(input_nc, nf, 4, 2, 1, bias = True),
      nn.LeakyReLU(0.1)
      )
    # input is 64 x 64
    layer2 = nn.Sequential(
      nn.Conv2d(nf,nf*2, 4, 2, 1, bias = True),
      nn.BatchNorm2d(nf*2),
      nn.LeakyReLU(0.1)
      )
    # input is 32
    layer3 = nn.Sequential(
      nn.Conv2d(nf*2,nf*4, 4, 2, 1, bias = True),
      nn.BatchNorm2d(nf*4),
      nn.LeakyReLU(0.1)
      )
    # input is 16
    layer4 = nn.Sequential(
      nn.Conv2d(nf*4,nf*8, 4, 2, 1, bias = True),
      nn.BatchNorm2d(nf*8),
      nn.LeakyReLU(0.1)
      )
    # input is 8
    layer5 = nn.Sequential(
      nn.Conv2d(nf*8,nf*8, 4, 2, 1, bias = True),
      nn.BatchNorm2d(nf*8),
      nn.LeakyReLU(0.1)
      )
    # input is 4
    layer6 = nn.Sequential(
      nn.Conv2d(nf*8,nf*8, 4, 2, 1, bias = True),
      nn.BatchNorm2d(nf*8),
      nn.LeakyReLU(0.1)
      )
    # input is 2
    layer7 = nn.Sequential(
      nn.Conv2d(nf*8,nf*8, 4, 2, 1, bias = True)
      )


    ## NOTE: decoder
    # input is 1
    dlayer7 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(nf*8,nf*8, 4, 2, 1, bias = True),
      nn.BatchNorm2d(nf*8),
      nn.Dropout2d(0.5, inplace=True)
      )
    # input is 2
    dlayer6 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(nf*8*2,nf*8, 4, 2, 1, bias = True),
      nn.BatchNorm2d(nf*8),
      nn.Dropout2d(0.5, inplace=True)

      )
    # input is 4
    dlayer5 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(nf*8*2,nf*8, 4, 2, 1, bias = True),
      nn.BatchNorm2d(nf*8),
      nn.Dropout2d(0.5, inplace=True)

      )
    # input is 8
    dlayer4 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(nf*8*2,nf*4, 4, 2, 1, bias = True),
      nn.BatchNorm2d(nf*4)
      )
    # input is 16
    dlayer3 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(nf*4*2,nf*2, 4, 2, 1, bias = True),
      nn.BatchNorm2d(nf*2)
      )
    # input is 32
    dlayer2 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(nf*2*2,nf, 4, 2, 1, bias = True),
      nn.BatchNorm2d(nf)
      )
    # input is 64

    dlayer1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(nf*2,output_nc, 4, 2, 1, bias=True))

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5
    self.layer6 = layer6
    self.layer7 = layer7

    self.dlayer7 = dlayer7
    self.dlayer6 = dlayer6
    self.dlayer5 = dlayer5
    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)

    dout7 = self.dlayer7(out7)
    dout7_out6 = torch.cat([dout7, out6], 1)
    dout6 = self.dlayer6(dout7_out6)
    dout6_out5 = torch.cat([dout6, out5], 1)
    dout5 = self.dlayer5(dout6_out5)
    dout5_out4 = torch.cat([dout5, out4], 1)
    dout4 = self.dlayer4(dout5_out4)
    dout4_out3 = torch.cat([dout4, out3], 1)
    dout3 = self.dlayer3(dout4_out3)
    dout3_out2 = torch.cat([dout3, out2], 1)
    dout2 = self.dlayer2(dout3_out2)
    dout2_out1 = torch.cat([dout2, out1], 1)
    dout1 = self.dlayer1(dout2_out1)
    return dout1

    '''
  #初始化权重
  def _initialize_weights(self):
    #self.modules递归地进入网络中的所有模块
    #参见https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/4
    for m in self.modules():
      #isinstance(object, classinfo),如果参数object是classinfo的实例，或者object是classinfo类的子类的一个实例， 返回True
      if isinstance(m, nn.Conv2d):
        #nn.init 参数初始化方法,orthogonal_正交矩阵
        init.orthogonal_(m.weight)
        print('init weight')
        if m.bias is not None:
          #常数数值初始化
          init.constant_(m.bias, 0)
          #当模型为BN层时的数据初始化
      elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
  '''
  
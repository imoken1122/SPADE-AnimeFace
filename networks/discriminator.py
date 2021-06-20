import torch as th
from torch import nn
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
from torchsummary import summary
from .norm import get_norm_layer
import torch.nn.functional as F
class Multiscale_Discriminator(nn.Module):
    def __init__(self,opt):
        super().__init__()
        num_D = opt.num_D
        self.multi_D = []
        for i in range(0, num_D):
            self.multi_D.append(Discriminator(opt))

    def downsample(self,x):
        return F.avg_pool2d(x, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    def forward(self,x):
        if len(self.multi_D)==1:
            result = self.multi_D[0](x)
        else:
            result = []
            for D in self.multi_D:
                out = D(x)
                out = [out]
                result.append(out)
                x = self.downsample(x)
        
        return result


class Discriminator(nn.Module):
    def __init__(self,opt=None):
        super().__init__()
        self.opt = opt
        ks = 4
        in_ch = opt.disc_in_ch
        n_layer_D = opt.n_layer_D 
        ndf =opt.ndf 
        pd = (ks-1)//2

        norm_layer = get_norm_layer("spectral")
        layer = [nn.Conv2d(in_ch,ndf,kernel_size = ks,stride=2,padding=pd),nn.LeakyReLU(0.2,False)]        

        for n in range(1,n_layer_D):
            ndf_prev = ndf
            ndf = min(ndf*2,512)
            stride = 1 if n == n_layer_D-1 else 2
            layer += [get_norm_layer(None,norm_type = "spectral",obj = nn.Conv2d(ndf_prev,ndf,kernel_size=ks, stride=stride, padding=pd)),
                            nn.LeakyReLU(0.2,False)]

        layer += [nn.Conv2d(ndf,1,kernel_size=ks, stride=1, padding=pd)]

        self.block = nn.Sequential(*layer)


    def forward(self,x):
        out = self.block(x)
        return out


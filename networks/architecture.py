
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .norm import SPADE
import torch as th

class SPADEResnetBlock(nn.Module):
    def __init__(self,in_ch,out_ch,opt):
        super().__init__()

        self.learned_shortcut = (in_ch != out_ch)
        mid_ch = min(in_ch, out_ch)
        self.conv_0 = nn.Conv2d(in_ch,mid_ch,kernel_size=3,padding=1)
        self.conv_1 = nn.Conv2d(mid_ch,out_ch,kernel_size=3,padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)


        self.norm_0 = SPADE(norm_ch = in_ch, label_ch = 3,opt=opt)
        self.norm_1 = SPADE(norm_ch = mid_ch, label_ch = 3,opt=opt)

        if self.learned_shortcut:
            self.norm_s = SPADE(in_ch,3, opt)

        self.actv = nn.LeakyReLU(0.2,False)
        self.noise=NoiseInjection() if opt.noise else lambda x: x

    def shortcut(self,x,seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x,seg))
        else: x_s = x
        return x_s

    def forward(self,x,seg):

        x_s = self.shortcut(x,seg)
        x = self.norm_0(x,seg)
        x = self.actv(x)
        dx = self.conv_0(x)
        #dx = self.conv_0(self.actv(self.norm_0(x,seg)))
        x = self.norm_1(dx,seg)
        x = self.actv(x)
        dx = self.conv_1(x)
        dx = self.noise(dx)

        #dx = self.conv_1(self.actv(self.norm_1(dx,seg)))


        out = x_s + dx

        return out
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(th.zeros(1), requires_grad=True)
    def forward(self,feat,noise=None):
        if noise is None:
            batch,_, h, w = feat.shape
            noise = th.randn(batch, 1 , h , w).to(feat.device)
        return feat + self.w * noise



# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(th.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = th.nn.Sequential()
        self.slice1_gray = th.nn.Sequential()
        self.slice2 = th.nn.Sequential()
        self.slice3 = th.nn.Sequential()
        self.slice4 = th.nn.Sequential()
        self.slice5 = th.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        
        vgg_pretrained_features[0]= nn.Conv2d(1,64,3,1,1)
        for x in range(2):
            self.slice1_gray.add_module(str(x), vgg_pretrained_features[x])

        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, ch):
        if ch == 3:
            h_relu1 = self.slice1(X)
        else:
            h_relu1 = self.slice1_gray(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
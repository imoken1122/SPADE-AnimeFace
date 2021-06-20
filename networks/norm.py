from torch import nn
from torch.nn.utils.spectral_norm import spectral_norm
import torch.nn.functional as F
def get_norm_layer(in_ch,norm_type="instance",obj=None):

    if norm_type == "batch":
        norm_layer = nn.BatchNorm2d(in_ch, affine=False)
    elif norm_type == "instance":
        norm_layer = nn.InstanceNorm2d(in_ch, affine=False)
    elif norm_type == "spectral":
        norm_layer = spectral_norm(obj)
    else:
        raise ValueError("normalization layer error")
    return norm_layer


class SPADE(nn.Module):
    def __init__(self,norm_ch,label_ch=3,opt=None):
        super().__init__()
        """
        if opt.spade_norm_type == 'instance':
            self.norm_layer = nn.InstanceNorm2d(norm_ch, affine=False)
        elif opt.spade_norm_type == 'batch':
            self.norm_layer = nn.BatchNorm2d(norm_ch, affine=False)
        """
        self.norm_layer = nn.BatchNorm2d(norm_ch, affine=False) 
        ks = 3
        pad = (ks-1)//2
        out_ch = 128

        self.layer = nn.Sequential(
                    nn.Conv2d(label_ch, out_ch,kernel_size=ks ,padding =pad), 
                    nn.ReLU())   
        self.mlp_gamma=nn.Conv2d(out_ch,norm_ch,kernel_size=ks,padding=pad)
        self.mlp_beta =nn.Conv2d(out_ch,norm_ch,kernel_size=ks,padding=pad)

    def forward(self,x,segmap):

        norm_x = self.norm_layer(x)
        segmap = F.interpolate(segmap,size = x.size()[2:], mode="nearest")

        act = self.layer(segmap)

        gamma = self.mlp_gamma(act)
        beta = self.mlp_beta(act)


        out = norm_x * (1+gamma) + beta
        return out

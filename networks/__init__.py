from .generator import *
from .loss import *

from .discriminator import * 

from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):

        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
       # elif classname.find('BatchNorm2d') != -1:
        ##    init.normal_(m.weight.data, 1.0, gain)
         #   init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    return net

def define_G(opt):
    device = "cuda" if opt.cuda else "cpu"

    net = Generator(opt)

    net = init_weights(net,init_type = opt.init_type)
    net = net.to(device)
    return net

def define_D(opt):
    device = "cuda" if opt.cuda else "cpu"
    net=Discriminator(opt)
    net = init_weights(net,init_type = opt.init_type)
    net = net.to(device)
    return net
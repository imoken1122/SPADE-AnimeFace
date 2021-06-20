from torch import nn
import torch as th
from torch.nn import init
import functools
import networks 
import utils

import torch.nn.functional as F

class pix2pixModel(nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.netG, self.netD = self.initialize_model(opt)
        if opt.isTrain:
            device = "cuda" if opt.cuda else "cpu"
            self.criterionGAN = networks.loss.GANLoss(opt.gan_mode,opt=opt).to(device)
            self.criterionFeat = th.nn.L1Loss().to(device)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss().to(device)
        self.fake_img = None
        self.opt =opt
        #self.avg_param_G = utils.copy_G_params(self.netG)


    def initialize_model(self,opt):
        netG,netD = networks.define_G(opt), networks.define_D(opt) if opt.isTrain else None
        if not opt.isTrain or opt.continue_train or opt.continue_train_latest:
            netG = utils.load_model(netG,"G",opt.which_checkpoint[0],opt.which_checkpoint[1],opt, latest = True if opt.continue_train_latest else False)
            if opt.isTrain:
                netD = utils.load_model(netD,"D",opt.which_checkpoint[0],opt.which_checkpoint[1],opt,latest = True if opt.continue_train_latest else False)

        return netG,netD
        
    def forward(self,data,z=None,mode=None):
        real_img,ref_img = data

        if self.opt.cuda :
            real_img,ref_img = real_img.to("cuda"),ref_img.to("cuda")

        if mode == "generator": 
            g_loss,fake_img = self.calc_G_loss(real_img,ref_img)
            return g_loss,fake_img
        elif mode == "discriminator": 
            d_loss = self.calc_D_loss(real_img,ref_img)
            return d_loss
        elif mode == "inference":
            with th.no_grad():
               fake_img = self.generate_fake(ref_img,z)
            return fake_img
        else:
            raise ValueError("Not define mode")


    def create_optimizers(self, opt):
        opt_G = th.optim.Adam(self.netG.parameters(),lr=opt.lr_G,betas = (opt.beta1,opt.beta2))
        opt_D = th.optim.Adam(self.netD.parameters(),lr=opt.lr_D,betas = (opt.beta1,opt.beta2))
        return opt_G,opt_D

    def calc_G_loss(self,real_img,ref_img):
        
        G_losses = {}

        self.fake_img = self.netG(ref_img)
        pred_fake,pred_real = self.excute_discriminate(self.fake_img,real_img,ref_img)
        G_losses["GAN"] = self.criterionGAN(pred_fake,True,for_disc=False)
        G_losses["L1"] = self.criterionFeat(self.fake_img,real_img) * 100.

        if not self.opt.no_vgg_loss:
            G_losses['G_VGG'] = self.criterionVGG(self.fake_img, real_img,3) \
                * self.opt.lambda_vgg

        return G_losses, self.fake_img


    def calc_D_loss(self,real_img,ref_img):
        D_losses = {}
        with th.no_grad():
            fake_img = self.netG(ref_img)

       
        fake_img = fake_img.detach().requires_grad_()
        pred_fake,pred_real = self.excute_discriminate(fake_img,real_img,ref_img)
        
        D_losses["D_fake"] = self.criterionGAN(pred_fake,False)
        D_losses["D_real"] = self.criterionGAN(pred_real,True)
        return D_losses


    def excute_discriminate(self,fake_img,real_img,ref_img):
        fake_concat = th.cat([fake_img,ref_img],1)
        real_concat = th.cat([real_img,ref_img],1)

        fake_and_real=th.cat([fake_concat,real_concat],0)

        disc_out = self.netD(fake_and_real)


        pred_fake,pred_real = self.div_pred(disc_out)

        return pred_fake,pred_real

    def div_pred(self,pred):
        if type(pred) == list: 
            f,r = [],[]
            for p in pred:
                f.append( disc_out[:len(disc_out)//2] for disc_out in p)
                r.append( disc_out[len(disc_out)//2:] for disc_out in p)
        else:
            f,r = pred[:len(pred)//2],pred[len(pred)//2:]
        return f,r
    

    def generate_fake(self,ref_img,z = None,):
        if z != None:
            return self.netG(ref_img ,z)
        else:
            return self.netG(ref_img)


    def save(self,epoch,iter):
        utils.save_model(self.netG,self.netD,epoch,iter,self.opt)


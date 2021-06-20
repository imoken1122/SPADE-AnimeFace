import torch as th
import torch.nn as nn
import torch.nn.functional as F
from networks.architecture import VGG19

class GANLoss(nn.Module):
    def __init__(self,gan_mode="hinge",target_real_label=1.0, target_fake_label=0.0, tensor=th.FloatTensor,opt=None):
        super(GANLoss, self).__init__()
        self.real_label=target_real_label
        self.fake_label=target_fake_label
        self.Tensor = tensor
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor=None
        self.gan_mode = gan_mode
        self.device = "cuda" if opt.cuda else "cpu"
        

    def get_target_tensor(self,input,target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input).to(self.device)
        else: 
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input).to(self.device)

    def get_zero_tensor(self,input,):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)

        return self.zero_tensor.expand_as(input).to(self.device)

    
    def loss(self, input, target_is_real, for_disc = True):

        if self.gan_mode=="original":
            target_tensor = self.get_target_tensor(input,target_is_real)
            loss = F.binary_cross_entropy_with_logits(input,target_tensor)
            return loss
        
        elif self.gan_mode == "ls": 
            target_tensor = self.get_target_tensor(input,target_is_real)
            return F.mse_loss(input,target_tensor)
        
        elif self.gan_mode == "hinge": 
            if for_disc:
                if target_is_real:

                    minval = th.min(input -1 , self.get_zero_tensor(input))
                    loss = -th.mean(minval)
                else:
                    minval = th.min(-input-1, self.get_zero_tensor(input))
                    loss = -th.mean(minval)
            else:
                loss = -th.mean(input)

            return loss
        else:
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()


    def forward(self,input, target_is_real,for_disc=True):
        if isinstance(input,list):
            loss = 0
            
            for pred_i in input:
                if isinstance(pred_i,list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real,for_disc)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = th.mean(loss_tensor.view(bs,-1),dim =1)
                loss += new_loss

            return loss /len(loss)
        else:
            return self.loss(input, target_is_real,for_disc)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y, ch):
        x_vgg, y_vgg = self.vgg(x,ch), self.vgg(y,ch)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss



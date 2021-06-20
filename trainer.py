from define_model import pix2pixModel
from torch.optim import lr_scheduler

class pix2pixTrainer():

    def __init__(self,opt):
        
        self.pix2pix = pix2pixModel(opt)
        self.opt = opt
        if opt.isTrain:
            self.opt_G,self.opt_D = self.pix2pix.create_optimizers(opt)
            self.old_lr_G = opt.lr_G
            self.old_lr_D = opt.lr_D
            self.opt_G_sch,self.opt_D_sch = self.get_scheduler(self.opt_G,opt),self.get_scheduler(self.opt_D,opt)

    def train_generator_one_step(self,data):
        self.opt_G.zero_grad()
        g_losses, fake_img = self.pix2pix(data, mode = "generator")
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.opt_G.step()

        #for p, avg_p in zip(self.pix2pix.netG.parameters(), self.pix2pix.avg_param_G):
        #    avg_p.mul_(0.999).add_(0.001 * p.data)

        
        self.g_losses = g_losses
        self.generated = fake_img

    def train_discriminator_one_step(self,data):
        self.opt_D.zero_grad()
        d_losses = self.pix2pix(data,mode = "discriminator")
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.opt_D.step()
        self.d_losses = d_losses

    def generate_img(self,data,z = None):
        fake_img = self.pix2pix(data,z,mode="inference")
        return fake_img

    def get_latest_generated(self,):
        return self.generated

    def save(self,epoch,iter):
        self.pix2pix.save(epoch,iter)

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_scheduler(self,optimizer, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + opt.which_checkpoint[0] - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler
    def update_learning_rate(self,):
        self.opt_G_sch.step()
        self.opt_D_sch.step()
        lr_G = self.opt_G.param_groups[0]['lr']
        lr_D = self.opt_D.param_groups[0]['lr']
        print('Generator learning rate = %.7f' % lr_G)
        print('Discriminator learning rate = %.7f' % lr_D)


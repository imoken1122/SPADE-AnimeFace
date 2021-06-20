import torch as th
from torch import nn

from .architecture import SPADEResnetBlock
from torchsummary import summary
import torch.nn.utils.spectral_norm as spectral_norm
class Generator(nn.Module):
    def __init__(self,opt=None):
        super().__init__()
        nf = opt.ngf
        sw,sh = opt.sw,opt.sh
        self.opt=opt
        self.device = "cuda" if opt.cuda else "cpu"
        self.fc = nn.Linear(opt.z_dim, 16*nf*sw*sh) 

        self.head = SPADEResnetBlock(nf*16,nf*16,opt)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf,opt )
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.up = nn.Upsample(scale_factor=2)


        self.dec_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.dec_1 = SPADEResnetBlock(8 * nf, 4 * nf,opt )
        self.dec_2 = SPADEResnetBlock(4 * nf, 2 * nf,opt)
        self.dec_3 = SPADEResnetBlock(2 * nf, 1 * nf,opt)


        self.output = nn.Sequential(
            nn.LeakyReLU(0.2,False),
            nn.Conv2d(nf,3,kernel_size=3,padding=1),
            nn.Tanh()
            )



    def forward(self,seg,z = None):
        if z == None:
            z = th.randn(seg.size(0), self.opt.z_dim,dtype=th.float32).to(self.device)

        z = self.fc(z)
        z = z.view(-1, 16*self.opt.ngf, self.opt.sw, self.opt.sh)


        x = self.head(z,seg)
        x = self.up(x)
        x = self.G_middle_0(x,seg)
        x = self.up(x)
        x = self.G_middle_1(x,seg)

        x = self.up(x)
        x = self.dec_0(x,seg)
        x = self.up(x)
        x = self.dec_1(x,seg)
        x = self.up(x)
        x = self.dec_2(x,seg)
        x = self.up(x)
        x = self.dec_3(x,seg)

        x = self.output(x)

        return x



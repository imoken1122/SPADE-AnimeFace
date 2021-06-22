import numpy as np
import torch as th
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision as tv
from PIL import Image
import cv2
import utils
import glob
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torchvision.transforms import functional as tvf




class GridMask():
    def __init__(self, p=1, d_range=(70, 100), r=0.1):
        self.p = p
        self.d_range = d_range
        self.r = r
        
    def __call__(self, sample):
        """
        sample: torch.Tensor(3, height, width)
        """
        if np.random.uniform() > self.p:
            return sample

        side = sample.shape[1]
        d = np.random.randint(*self.d_range, dtype=np.uint8)
        r = int(self.r * d)
        
        mask = np.ones((side+d, side+d), dtype=np.uint8)
        for i in range(0, side+d, d):
            for j in range(0, side+d, d):
                mask[i: i+(d-r), j: j+(d-r)] = 0
        delta_x, delta_y = np.random.randint(0, d, size=2)
        mask = mask[delta_x: delta_x+side, delta_y: delta_y+side]
        sample *= np.expand_dims(mask, 2)
        sample[sample==0] = 255

        return sample

class Diff_Augmentaion:
    def __init__(self,aug_list,is_aug_ex, seed):
        self.seed = seed
        self.aug_list = aug_list
        self.is_aug_ex = is_aug_ex
    def __call__(self,x):
        for i in range(len(self.aug_list)):
            aug = self.aug_list[i]
            flag = self.is_aug_ex[i]
            if aug =="cutout" and flag:
                x = self.rand_cutout(x)
            elif aug == "color" and flag:
                for f in [self.rand_brightness,self.rand_saturation,self.rand_contrast ]:
                    x = f(x)
            elif aug == "translation" and flag:
                x = self.rand_translation(x)
            elif aug == "hflip" and flag:
                x = x.flip(2)
           
        return x

    def rand_cutout(self,x, ratio=0.5):
        th.manual_seed(self.seed)
        cutout_size = int(x.size(1) * ratio + 0.5), int(x.size(2) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(1) + (1 - cutout_size[0] % 2), size=[ 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(2) + (1 - cutout_size[1] % 2), size=[ 1, 1], device=x.device)
        grid_x, grid_y = torch.meshgrid(

            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(1) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(2) - 1)
        mask = torch.ones( x.size(1), x.size(2), dtype=x.dtype, device=x.device)
        mask[ grid_x, grid_y] = 0
        x = x * mask.unsqueeze(0)
        return x

    def rand_brightness(self,x):
        th.manual_seed(self.seed)
        x = x + (torch.rand(1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
        return x


    def rand_saturation(self,x):
        x_mean = x.mean(dim=0, keepdim=True)
        th.manual_seed(self.seed)
        x = (x - x_mean) * (torch.rand(1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
        return x


    def rand_contrast(self,x):
        th.manual_seed(self.seed)
        x_mean = x.mean(dim=[0, 1, 2], keepdim=True)
        x = (x - x_mean) * (torch.rand( 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
        return x


    def rand_translation(self,x, ratio=0.125):
        th.manual_seed(self.seed)
        shift_x, shift_y = int(x.size(1) * ratio + 0.5), int(x.size(2) * ratio + 0.5)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[ 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[ 1, 1], device=x.device)
        grid_x, grid_y = torch.meshgrid(

            torch.arange(x.size(1), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(1) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(2) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1])
        x = x_pad.permute(1, 2, 0).contiguous()[ grid_x, grid_y].permute( 2, 0, 1)
        return x
class RandomRotation:

    def __init__(self, angles,seed):
        self.angles = angles
        self.seed = seed
    def __call__(self, x):
        th.manual_seed(self.seed)
        return tvf.rotate(x, self.angles)

class Dataset(data.Dataset):
    def __init__(self, dir_path,opt=None):
        super().__init__()

        self.img_path = sorted(glob.glob(f"{dir_path}/full/*"))
        self.ref_img_path = sorted(glob.glob(f"{dir_path}/ref/*"))

        self.len = len(self.img_path)
        self.opt = opt
        self.aug = opt.aug
        #self.tf = self.get_transforms(opt) 
    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        if len(self.aug ) != 0:
            seed = np.random.randint(1e10)
            is_aug_ex = np.random.choice([0,1],4,[0.35,0.65])
            tf_list = [RandomRotation(np.random.randint(-50,50 ),seed)] if np.random.choice([0,1],p = [0.45,0.55]) and "rotation" in self.opt.aug else []
            tf_list += [transforms.ToTensor(),
                        Diff_Augmentaion(self.aug,is_aug_ex,seed = seed),
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  ]
        else:
            tf_list = [transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  ]
        self.tf = transforms.Compose(tf_list)
        path = self.img_path[idx]
        ref_path = self.ref_img_path[idx]

        #img = cv2.resize(cv2.imread(path ),(256,256))
        #ref_img = cv2.resize(cv2.imread(ref_path ),(256,256))
        img = Image.open(path).resize((self.opt.im_size,self.opt.im_size))
        ref_img = Image.open(ref_path).resize((self.opt.im_size,self.opt.im_size))
        #img = cv2.resize(lycon.load(path ),(256,256))
        #ref_img = cv2.resize(lycon.load(ref_path ),(256,256))
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #ref_img = cv2.cvtColor(ref_img,cv2.COLOR_BGR2RGB)

        img = self.tf(img)
        ref_img = self.tf(ref_img)

        return img, ref_img

def create_train_test_dataloader(opt):

    train_dataloader = DataLoader(Dataset(f"{opt.input_path}/train",opt),
                                             batch_size=opt.batch_size, 
                                                shuffle=True,drop_last=True,num_workers=2)
    test_dataloader = DataLoader(Dataset(f"{opt.input_path}/test",opt), batch_size=opt.batch_size, shuffle=True,drop_last=True,num_workers=2)
    return train_dataloader,test_dataloader
def create_dataloader(opt):
    dataloader = DataLoader(Dataset(f"{opt.input_path}",opt), batch_size=opt.batch_size, shuffle=True,drop_last=True,num_workers=2)

    return dataloader
"""
#datasets = Dataset("../../faces/images/train",)
datasets = Dataset("./data/full/train","./data/ref/train")
dataloader = data.DataLoader(datasets,batch_size=5,shuffle = True,num_workers = 2,drop_last = True)
print(len(dataloader))

img,img_rf=next(iter(dataloader))
#img = rand_cutout(img)
print(img[0].min(),img[0].max())
grid1 = tv.utils.make_grid(img)
#grid2 = tv.utils.make_grid(img_sk)
grid3 = tv.utils.make_grid(img_rf)
grid = th.cat([grid1,grid3],dim=1)
print(grid.shape)

grid = np.transpose(( (grid + 1.) * 127.5)/255,[1,2,0])
print(grid.min(),grid.max(),)
plt.axis("off")
plt.imshow(grid)
plt.savefig("output.png")
#tv.utils.save_image(grid,"output.png")
print(grid.shape)

"""
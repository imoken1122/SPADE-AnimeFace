import os
import cv2
from torchvision import transforms
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torchvision as tv
import json


def create_dir(dirlist):
    """
    Create directory

    args: dirs (str or list) create all dirs in 'dirs'
    """
    for dirs in dirlist:
        if isinstance(dirs, (list, tuple)):
            for d in dirs:
                if not os.path.exists(os.path.expanduser(d)):
                    os.makedirs(d)
        elif isinstance(dirs, str):
            if not os.path.exists(os.path.expanduser(dirs)):
                os.makedirs(dirs)


def setup_logging(model_list, model_name='./log'):
    
    # Output path where we store experiment log and weights
    model_dir = [os.path.join(model_name, 'models', mn) for mn in model_list]

    fig_dir = os.path.join(model_name, 'figures')
    
    # Create if it does not exist
    create_dir([model_dir, fig_dir])

def save_model(G,D, epoch,iter, opt):
    model_name = opt.model_name
    th.save(G.state_dict(),f"{model_name}/models/net_G/G_{epoch}_{iter}.pth")
    th.save(D.state_dict(),f"{model_name}/models/net_D/D_{epoch}_{iter}.pth")

def load_model(net,name,epoch,iter,opt,latest = False):
    if latest:
        file = sorted(glob.glob(f"{opt.model_name}/models/net_{name}/*") ,reverse=True)[0] 
    else:
        file = glob.glob(f"{opt.model_name}/models/net_{name}/{name}_{epoch}_{iter}.pth")[0]
    if opt.cuda:
        print(f"GPU loading weight {file}")
        net.load_state_dict(th.load(file))
    else: 
        print("CPU loading weight")
        net.load_state_dict(th.load(file, th.device('cpu')))

    return net


def load_latest_model(net,name,opt):
    file = sorted(glob.glob(f"{opt.model_name}/models/net_{name}/*") ,reverse=True)[0]
    net.load_state_dict(th.load(file))
    return net





def progress_state(epoch=None,iter=None,mode="r",model_name="./log",opt=None):
    dic = {"epoch":"","iter":"","opt":opt}
    if mode == "w":
        with open(f"{model_name}/setup.json",mode) as f:
            dic["epoch"] = str(epoch)
            dic["iter"] = str(iter)
            dic["opt"] = str(opt)
            json.dump(dic,f)
    else:
        f = open(f"{model_name}/setup.json",mode) 
        dic = json.load(f)
        return dic

def plot_generated_image(img_full,img_ref,img_gen,epoch,suffix, model_name):
    grid0 = tv.utils.make_grid(img_full[:])
    grid2 = tv.utils.make_grid(img_ref[:])
    grid3 = tv.utils.make_grid(img_gen[:])
    grid = th.cat([grid0,grid2,grid3],dim=1)
    grid = np.transpose((grid/2 + 0.5),[1,2,0])
    plt.axis("off")
    plt.imshow(grid)
    plt.savefig(f"{model_name}/figures/{suffix}_{epoch}.png")

def plot_paint2face(img_ref,img_gen,suffix):
    grid2 = tv.utils.make_grid(img_ref)
    grid3 = tv.utils.make_grid(img_gen)
    grid = th.cat([grid2,grid3],dim=2)
    grid = np.transpose((grid/2 + 0.5),[1,2,0])
    plt.axis("off")
    plt.imshow(grid)
    plt.savefig(f"./paint/result/paint2gen_{suffix}") 


def make_contour_image(gray):
        neiborhood24 = np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1]],
                                np.uint8)
        dilated = cv2.dilate(gray, neiborhood24, iterations=1)
        diff = cv2.absdiff(dilated, gray)
        contour = 255 - diff

        return contour

def np2tensor(img):

    tf_full = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
                ])
    return tf_full(img).unsqueeze(0)


def getGrayImage(rgbImg):
    gray = 0.114*rgbImg[:,0,:,:] + 0.587*rgbImg[:,1,:,:] + 0.299*rgbImg[:,2,:,:]
    gray = th.unsqueeze(gray,1)
    return gray

from copy import deepcopy

def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
    
import torch as th
from PIL import Image
import numpy as np
import cv2
import glob
import utils
from option import define_option
from trainer import pix2pixTrainer
import torchvision as tv
import matplotlib.pyplot as plt
def get_interpolate_points(z1,z2,n=10):
	ratio = np.linspace(0,1,n)
	vector = [] 
	for r in ratio:
		v = (1. - r)* z1 + r * z2
		vector.append(v)
	return th.stack(vector,0)
def show(img):
    grid = tv.utils.make_grid(img,nrow=4)

    #grid = [tv.utils.make_grid(x) for x in grid]
    #grid = th.cat(grid,2)
    grid = np.transpose((grid/2 + 0.5),[1,2,0])
    plt.axis("off")
    plt.imshow(grid)
    plt.savefig(f"inner.png")

opt = define_option()
trainer = pix2pixTrainer(opt)
if opt.interpolation:
	n = 10
	p = glob.glob(opt.input_refpath)[0]
	img = cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB)
	img = utils.np2tensor(img)

	z = th.randn(2, opt.z_dim,dtype=th.float32)
	interpolated = get_interpolate_points(z[0],z[1],n)
	print(interpolated.shape)
	input = [[None]*n, img.repeat(n,1,1,1)]
	gen_img = trainer.generate_img(input,interpolated).detach().cpu()
	print(gen_img.shape)
	show(gen_img)




elif opt.self:
	path = glob.glob(opt.input_path+"/*")
	for p in path:
		img = cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB)
		img = utils.np2tensor(img)
		gen_img = trainer.generate_img([None,img]).detach().cpu()
		utils.plot_paint2face(img,gen_img,p.split("/")[-1])

else:
	path = sorted(glob.glob(opt.input_path+"/*"))
	bs = 8
	for i in range(len(path)//bs):
		l = path[i*bs:(i+1)*bs]
		img_1 = []
		img_2 = []
		gen = []
		for p in l:
			print(p)
			img1 = np.asarray(Image.open(p).resize((256,256)))
			img = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV )
			img = cv2.medianBlur(img,9)
			img = cv2.pyrMeanShiftFiltering(img,52,52)
			img = cv2.medianBlur(img,15)
			img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
			img_ref = utils.np2tensor(img)
			img_full = utils.np2tensor(img1)
			gen_img = trainer.generate_img([None,img_ref]).detach().cpu()
			img_1.append(img_full[0])
			img_2.append(img_ref[0])
			gen.append(gen_img[0])
		img_full = th.stack(img_1,0)
		img_ref = th.stack(img_2,0)
		gen_img = th.stack(gen,0)
		utils.plot_generated_image(img_full,img_ref, gen_img,0 ,p.split("/")[-1],opt.model_name)



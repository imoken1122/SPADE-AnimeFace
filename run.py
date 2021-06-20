from datasets import create_train_test_dataloader
import pickle
import torch as th
from torch.utils.data import DataLoader
import torchvision as tv
import glob
from torch import nn,optim
from torchsummary import summary
import statistics
import argparse
import os
from tqdm import tqdm
import utils
from trainer import pix2pixTrainer
from log import IterationLogging,Visualizer
from option import define_option

opt = define_option()
print(opt)
device = "cuda" if opt.cuda else "cpu"
print(device)

train_dataloader,test_dataloader = create_train_test_dataloader(opt)
iter_log = IterationLogging(opt,len(train_dataloader))


visualizer = Visualizer()
trainer = pix2pixTrainer(opt)
N = len(train_dataloader)

print("===> Starting")
for epoch in iter_log.train_epochs():
    iter_log.recode_epoch_start(epoch)

    for i,data_i in enumerate(tqdm(train_dataloader, initial = iter_log.epoch_iter),start=iter_log.epoch_iter):
        iter_log.recode_one_iter()
        trainer.train_generator_one_step(data_i)
        trainer.train_discriminator_one_step(data_i)

        losses = trainer.get_latest_losses()
        if iter_log.needs_saving():
            visualizer.print_current_errors(epoch,i * opt.batch_size,losses, iter_log.time_per_iter)
            iter_log.recode_current_iter()
            #trainer.save(epoch,i*opt.batch_size)

            fetch_gen_img = trainer.get_latest_generated().detach().cpu()
            test_data = next(iter(test_dataloader))
            val_gen_img = trainer.generate_img(test_data).detach().cpu()
            utils.plot_generated_image(data_i[0],data_i[1],fetch_gen_img,epoch,"train",opt.model_name)
            utils.plot_generated_image(test_data[0],test_data[1],val_gen_img,epoch,"valid",opt.model_name)
    iter_log.recode_epoch_end ()
    trainer.update_learning_rate()

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_log.total_epochs:
        pass
#        trainer.save(epoch,N)






import utils
import time
import os

class Visualizer():
    def __init__(self):
        pass

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        #with open(self.log_name, "a") as log_file:
        #    log_file.write('%s\n' % message)



class IterationLogging:
    def __init__(self,opt,len_data):
        if not os.path.exists(os.path.expanduser(opt.model_name)):
            os.makedirs(opt.model_name)
        utils.setup_logging(["net_G","net_D"],opt.model_name )
        self.opt = opt
        self.len_data = len_data
        self.start_epochs = 1
        self.total_epochs = opt.niter + opt.niter_decay
        self.epoch_iter = 0
        if opt.isTrain and opt.continue_train or opt.continue_train_latest: 
            try:
                dic = utils.progress_state(model_name=opt.model_name,)
                self.start_epochs = int(dic["epoch"])
                self.epoch_iter = int(dic["iter"])
                print("Resuming from epoch %d at iteration %d" % (self.start_epochs, self.epoch_iter))
            except :
                print("Not exist iteration log file, starting from begginning")
        self.total_steps_so_far = (self.start_epochs - 1) * len_data + self.epoch_iter

    def train_epochs(self, ):
        return range(self.start_epochs, self.total_epochs+1)
    
    def recode_epoch_start(self,epoch):
        self.epoch_start_time = time.time()
        self.last_iter_time = time.time()
        self.current_epoch = epoch

    def recode_one_iter(self):
        current_time = time.time()

        self.time_per_iter = (current_time - self.last_iter_time)/self.opt.batch_size
        self.last_iter_time = current_time
        self.total_steps_so_far += self.opt.batch_size
        self.epoch_iter += self.opt.batch_size



    def recode_epoch_end(self,):
        cur_t = time.time()
        self.time_per_epoch = cur_t - self.epoch_start_time
        self.epoch_iter = 0
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (self.current_epoch, self.total_epochs, self.time_per_epoch))
        if self.current_epoch % self.opt.save_epoch_freq == 0:
            utils.progress_state(model_name=self.opt.model_name,mode="w",epoch=self.current_epoch+1,iter=0,opt=self.opt) 


    def recode_current_iter(self):
        utils.progress_state(model_name=self.opt.model_name,epoch=self.current_epoch,iter=self.epoch_iter,mode="w",opt=self.opt) 


    def needs_saving(self):
        return (self.total_steps_so_far % self.opt.save_iter_freq) < self.opt.batch_size
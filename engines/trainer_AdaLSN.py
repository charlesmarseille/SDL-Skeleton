from PIL import Image
import math, os, time
import logging
import sys
from utils import *
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt

save_dir = 'Ada_LSN/weights/train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(save_dir)
#dataset_name = 'nord_cotiere'

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_dir, 'train_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


class Trainer(object):
    # init function for class
    def __init__(self, network, optimizer, dataloader, args):
        self.args = args
        self.network = network
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.dataset = args.dataset

        if not os.path.exists('weights'):
            os.makedirs('weights')

        self.timeformat = '%Y-%m-%d %H:%M:%S'

    def train(self):
        args_s = ''
        #logging.info(args_s.join([f'{arg} ' for arg in sorthelper.sortNumbers(self.args)]))
        #plt.ion()
        #fig, ax = plt.subplots(1,2)
        self.lossAccList = []
        lossAcc = 0.0
        lossFuse = 0.0
        #self.network.eval()  # if backbone has BN layers, freeze them
        self.network.train()
        dataiter = iter(self.dataloader)
        for _ in range(self.args.resume_iter // self.args.lr_step):
            self.adjustLR()
        self.optimizer.zero_grad()
        for step in range(self.args.resume_iter, self.args.max_step):
            
            for _ in range(self.args.iter_size):
                try:
                    data, target = next(dataiter)
                except StopIteration:
                    dataiter = iter(self.dataloader)
                    data, target = next(dataiter)

                data, target = data.cuda(self.args.gpu_id), target.cuda(self.args.gpu_id)
                data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)

                loss, fuse_loss = self.network(data, target)
                #if np.isnan(float(loss.data[0])):
                #if np.isnan(float(loss.item())):
                #    raise ValueError('loss is nan while training')
                loss /= self.args.iter_size
                loss.backward()
                lossAcc += loss.data[0]
                #lossAcc += loss.item()
                lossFuse += fuse_loss.data[0]
                #lossFuse += fuse_loss.item()
            
            #torch.nn.utils.clip_grad_norm_(self.network.parameters, 5.)        #CM - added to limit gradient explosion
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # adjust hed learning rate
            if (step > 0) and (step % self.args.lr_step) == 0:
                self.adjustLR()
                self.showLR()

            # visualize the loss
            if (step + 1) % self.args.disp_interval == 0:
                timestr = time.strftime(self.timeformat, time.localtime())
                logging.info('{} iter={} totloss={:<8.2f} fuseloss={:<8.2f}'.format(
                    timestr, step + 1, lossAcc / self.args.disp_interval,
                             lossFuse / self.args.disp_interval / self.args.iter_size))
                if step < self.args.max_step - 1:
                    lossAcc = 0.0
                    lossFuse = 0.0       


            if not os.path.exists(f'{save_dir}'):
                os.makedirs(f'{save_dir}')
            if (step + 1) % self.args.save_interval == 0:
                torch.save(self.network.state_dict(),
                           f'{save_dir}/skel_{step + 1}.pth')
        torch.save(self.network.state_dict(),
                   f'{save_dir}/skel_{self.args.max_step}.pth')
        return lossAcc / self.args.disp_interval

    def adjustLR(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.args.lr_gamma

    def showLR(self):
        for param_group in self.optimizer.param_groups:
            logging.info(param_group['lr'])
        logging.info('')

    def testSingle(self):
        gpu_id = 0
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        self.network.mode = 1    # put the model in test mode (dropouts inactive)
        dataiter = iter(self.dataloader)
        self.inp = Variable(next(dataiter)[0].cuda(gpu_id))
        self.test_inp = self.inp[:,-1]
        self.out = self.network(self.inp, None)

        plt.plot((self.step) / self.args.disp_interval, self.lossAccList, label='train lossAcc')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show(block=False)

    def plot_images(self):
        start = 500
        nx=1
        size = 15
        pylab.rcParams['figure.figsize'] = size, size / 2
        shape = 3
        fig, ax = plt.subplots(1, 3)
        image = self.out[0].data[0, 0].cpu().numpy().astype(np.float32)
        ax[0].imshow(self.inp.cpu()[0,0])
        ax[1].imshow(self.test_inp.cpu()[0])
        ax[2].imshow(1 - image, cmap=cm.Greys_r)
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[2].set_xticklabels([])
        ax[2].set_yticklabels([])
        plt.tight_layout()
        plt.savefig(f'epoch_{self.fname}.png')

import argparse
from torch import optim
import torch
from torch.utils.data import DataLoader
from datasets.sklarge_RN_exon import TrainDataset
from engines.trainer_AdaLSN import Trainer, logging
from Ada_LSN.model import Network
from Ada_LSN.utils import *
import os
from Ada_LSN.genotypes import geno_inception as geno
from glob import glob

# Added to limit mem errors from small GPU
#torch.cuda.empty_cache()
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:300'
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


parser = argparse.ArgumentParser(description='TRAIN SDL-skeleton')
parser.add_argument('--dataset', default='ryam', type=str)          #ryam or nord_cotiere
parser.add_argument('--with_mhc', default=None, type=bool)
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--INF', default=1e6, type=int)
parser.add_argument('--C', default=32, type=int)  # 32/64/128
parser.add_argument('--each_steps', default=500, type=int)
parser.add_argument('--numu_layers', default=5, type=int)
parser.add_argument('--u_layers', default=[0, 1, 2, 3], type=list)
parser.add_argument('--lr', default=1.e-6, type=float)
parser.add_argument('--lr_step', default=60000, type=int)
parser.add_argument('--lr_gamma', default=0.05, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--iter_size', default=10, type=int)
parser.add_argument('--max_step', default=80000, type=int)  # train max_step each architecture
parser.add_argument('--disp_interval', default=100, type=int)
parser.add_argument('--resume_iter', default=0, type=int)
parser.add_argument('--save_interval', default=500, type=int)
args = parser.parse_args()


def train_model(g, dataloader, args):
    try:
        logging.info(g[0])
        logging.info(g[1][0])
        logging.info(g[1][1])
        logging.info(g[1][2])
        logging.info(g[1][3])
        logging.info(g[1][4])
        logging.info(
            '%s %s %s %s %s' % (str(g[2][0]), str(g[2][1]), str(g[2][2]), str(g[2][3]), str(g[2][4])))
        logging.info(g[2][5])
        logging.info(g[2][6])
    except Exception as e:
        pass
#    net = Network(args.C, args.numu_layers, args.u_layers, geno,
#                  pretrained_model='./pretrained_model/inceptionV3.pth').cuda(args.gpu_id)
    net = Network(args.C, args.numu_layers, args.u_layers, geno).cuda(args.gpu_id)
    net.mode = 0    # put the model in train mode (dropouts active)

    logging.info('params:%.3fM' % count_parameters_in_MB(net))
    lr = args.lr / args.iter_size
    optimizer = optim.Adam(net.parameter(args.lr), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    trainer = Trainer(net, optimizer, dataloader, args)
    save_dir = glob('Ada_LSN/weights/train*/')[-1]
    with open(save_dir+'train_log.txt', 'a+') as f:
        f.write(' '.join([(arg+','+str(getattr(args, arg))) for arg in vars(args)])+'\n')
    loss = trainer.train()


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(args.gpu_id)
    dataset = TrainDataset(args)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=None)
    

    train_model(geno, dataloader, args)

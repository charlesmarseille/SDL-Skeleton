import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
from datasets.sklarge_RN_exon import TestDataset
import torch
from Ada_LSN.model import Network
from Ada_LSN.genotypes import geno_inception as geno
import os
import time
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.io as scio
import tifffile

def plot_results(out, inp, test_inp, ax, j):
    image = out[0].data[0, 0].cpu().numpy().astype(np.float32)
    ax[j,0].imshow(inp.cpu()[0,0])
    ax[j,1].imshow(test_inp.cpu()[0])
    ax[j,2].imshow(1 - image, cmap=cm.Greys_r)
    ax[j,0].set_xticklabels([])
    ax[j,0].set_yticklabels([])
    ax[j,1].set_xticklabels([])
    ax[j,1].set_yticklabels([])
    ax[j,2].set_xticklabels([])
    ax[j,2].set_yticklabels([])


parser = argparse.ArgumentParser(description='TEST camille')
parser.add_argument('-W', '--weights', default=80000, type=int)
parser.add_argument('-S', '--start', default=100, type=int)
parser.add_argument('--C', default=64, type=int)  # 32/64/128
args = parser.parse_args()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def test_dataset():
    gpu_id = 0
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    net = Network(args.C, 5, [0, 1, 2, 3], geno).cuda(0).eval()
    net.load_state_dict(torch.load(f'./Ada_LSN/weights/inception_sklarge/skel_{args.weights}.pth', map_location=lambda storage, loc: storage))
    net.mode = 1    # put the model in test mode (dropouts inactive)

    dataset = TestDataset()
    dataloader = list(DataLoader(dataset, batch_size=1))
    output_dir = './Ada_LSN/output/inception_camille/results/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    start_time = time.time()
    tep = 1
    start = 500
    nx=5
    size = 15
    pylab.rcParams['figure.figsize'] = size, size / 2
    shape = 3
    fig,ax = plt.subplots(nx, 3)
    for i, (inp, test_inp, fname, H, W) in enumerate(dataloader[start:start+nx]):
        fileName = output_dir + fname[0].split('/')[-1][:-4] + '.png'
        tep += 1
        #inp = Variable(inp[:, :-1].cuda(gpu_id))
        inp = Variable(inp.cuda(gpu_id))
        out = net(inp, None)
        plot_results(out, inp, test_inp, ax, i)
        #out_np = out[0].data[0][0, 0].cpu().numpy()
        #out_resize = cv2.resize(out_np, (W.item(), H.item()), interpolation=cv2.INTER_LINEAR)
        #s = plt.subplot(1, 5, 1)
        #s.imshow(out_resize)
        #scio.savemat(fileName, {'sym': out_resize})
        #print('{} size{} resize{}'.format(fileName, out.size(2) * out.size(3), out_resize.shape[0] * out_resize.shape[1]))
    plt.tight_layout()
    plt.show()
    #plt.savefig(fileName)
    diff_time = time.time() - start_time
    print('Detection took {:.3f}s per image'.format(diff_time / len(dataloader)))


test_dataset()
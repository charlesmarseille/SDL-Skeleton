import cv2
from scipy.io import loadmat
import os
import numpy as np
import random
from matplotlib import pyplot as plt

def random_rotate_flip(img, pred):
    # chose a random transformation
    rot_to_do = random.sample([None, 1, 2, 3], 1)[0]
    flip_to_do = random.sample([None, 0, 1], 1)[0]
    # do transformation on image and mask
    if rot_to_do is not None:
        img = np.rot90(img, k=rot_to_do, axes=(0,1))
        pred = np.rot90(pred, k=rot_to_do, axes=(0,1))
    if flip_to_do is not None:
        img = np.flip(img, axis=flip_to_do)
        pred = np.flip(pred, axis=flip_to_do)
    return img, pred

# Define data direction
data_dir = 'C:/Users/cmarseille/Documents/GitHub/SDL-Skeleton/datasets/sk_large_dataset/sk1491'

# Images and ground truth paths
folder = 'train'
data_path = os.path.join(data_dir, 'images', folder)
gt_path = os.path.join(data_dir, 'groundTruth', folder)

# Loop in folders
fnames = []
for file in os.listdir(data_path):
    # Load image and ground truth
    img = cv2.imread(os.path.join(data_path,file))
    gt = loadmat(os.path.join(gt_path,file[:-4]+'.mat'))['symmetry']
    
    # Process for binary mask (some masks have dtype=float)
    data_mask = gt!=0
    gt[data_mask] = gt[data_mask] / gt[data_mask]
    
    # Rotate and flip
    img, gt = random_rotate_flip(img, gt)
    fname_img = os.path.join(data_path,file[:-4])+'_rot.png'
    fname_gt = os.path.join(gt_path,file[:-4])+'_rot.png'
    cv2.imwrite(fname_img, img)
    cv2.imwrite(fname_gt, gt)
    fnames.append(fname_img+' '+fname_gt)
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.subplot(1,2,2)
    # plt.imshow(gt)
    # plt.show()

# Create list of filenames
with open('./datasets/sk_large_dataset/sk1491/train_pairRN60_255_s_all.lst', 'w+') as f:
    for line in fnames:
        f.write(line+'\n')
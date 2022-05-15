import os
import numpy as np
import pandas as pd
import skimage.io as io
from torch.utils.data import Dataset
import torch
import cv2
import tifffile
import random

def get_fileNames():
    dataset_train = []
    dataset_test = []
    sup_data_paths = [
        'C:/Users/cmarseille/Documents/GitHub/SDL-Skeleton/datasets/dataset_camille_128/true_data/',
        'C:/Users/cmarseille/Documents/GitHub/SDL-Skeleton/datasets/dataset_camille_128/false_data/',
        'C:/Users/cmarseille/Documents/GitHub/SDL-Skeleton/datasets/dataset_camille_128/true_data/',
        'C:/Users/cmarseille/Documents/GitHub/SDL-Skeleton/datasets/dataset_camille_128/false_data/',
        ]
    th_data = [1.0, 0.025, 1.0, 0.025]
    for i, path in enumerate(sup_data_paths):
        for file in os.listdir(path):
            if np.random.random() <= th_data[i]:
                _, idxi, idxj = file.split('_')
                if int(idxi) <= 6000:
                    dataset_train.append(path+file)
                else:
                    dataset_test.append(path+file)
    return dataset_train


class TrainDataset(Dataset):
    def __init__(self, fileNames, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
#        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ', header=None)
        self.fileNames = fileNames

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        inputName = self.fileNames[idx]
        image = tifffile.imread(inputName)       # image format is 256,256,5
        inputImage = image[:,:,:-1]
        inputImage = inputImage[:, :, ::-1]
        inputImage = inputImage.astype(np.float32)
        inputImage = np.true_divide(np.subtract(inputImage, self.means), self.stds)
        inputImage = inputImage.transpose((2, 0, 1))

        targetImage = image[:,:,-1]
        if len(targetImage.shape) == 3:
            targetImage = targetImage[:, :, 0]
        targetImage = targetImage > 0.0
        targetImage = targetImage.astype(np.float32)
        targetImage = np.expand_dims(targetImage, axis=0)

        # choose a random transformation
        rot_to_do = random.sample([None, 1, 2, 3], 1)[0]
        flip_to_do = random.sample([None, 0, 1], 1)[0]

        # do transformation on inputImage and mask
        if rot_to_do is not None:
            inputImage = np.rot90(inputImage, k=rot_to_do, axes=(1,2))
            targetImage = np.rot90(targetImage, k=rot_to_do, axes=(1,2))
        if flip_to_do is not None:
            inputImage = np.flip(inputImage, axis=flip_to_do)
            targetImage = np.flip(targetImage, axis=flip_to_do)

        print(inputImage.shape)

        inputImage = torch.Tensor(inputImage)
        targetImage = torch.Tensor(targetImage)
        return inputImage, targetImage


class TestDataset(Dataset):
    def __init__(self, fileNames, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ', header=None)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        fname = self.frame.iloc[idx, 0]
        inputName = os.path.join(self.rootDir, fname + '.jpg')

        inputImage = io.imread(inputName)[:, :, ::-1]
        K = 60000.0  # 180000.0 for sympascal
        H, W = inputImage.shape[0], inputImage.shape[1]
        sy = np.sqrt(K * H / W) / float(H)
        sx = np.sqrt(K * W / H) / float(W)
        inputImage = cv2.resize(inputImage, None, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
        inputImage = inputImage.astype(np.float32)
        inputImage -= np.array([104.00699, 116.66877, 122.67892])
        inputImage = inputImage.transpose((2, 0, 1))

        inputImage = torch.Tensor(inputImage)
        return inputImage, fname, H, W


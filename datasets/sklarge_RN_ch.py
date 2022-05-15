import os
import numpy as np
import pandas as pd
import skimage.io as io
from torch.utils.data import Dataset
import torch
import cv2
import tifffile
import random




class TrainDataset(Dataset):
    def __init__(self, fileNames, rootDir, transform=None):
        self.transform = transform
        self.rootDir = rootDir
        self.means = [22775.8, 11.3, 7.8, 0.11]
        self.stds = [924.8, 6.8, 6.9, 0.097]
        self.batch_size = 8
        self.fileNames = fileNames

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        inputName = os.path.join(self.rootDir, self.fileNames[idx])
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


        inputImage = torch.Tensor(inputImage.copy()[:-1])
        targetImage = torch.Tensor(targetImage.copy())
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


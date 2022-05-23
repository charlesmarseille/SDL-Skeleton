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
    return dataset_train, dataset_test


class TrainDataset(Dataset):
    def __init__(self, rootDir, transform=None):
        self.transform = transform
        self.rootDir = rootDir
        self.means = [22775.8, 11.3, 7.8, 0.11]
        self.stds = [924.8, 6.8, 6.9, 0.097]
        self.batch_size = 8
        self.fileNames = get_fileNames()[0]

    def __len__(self):
        return int(np.floor(len(self.fileNames) / self.batch_size))

    def __getitem__(self, idx):
        inputNames = self.fileNames[self.batch_size*idx : (idx+1)*self.batch_size]
        X, Y = list(), list()
        for inputName in inputNames:
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


            X.append(inputImage)
            Y.append(targetImage)


        X = np.array(X)
        Y = np.array(Y)

        inputImage = torch.Tensor(X)
        targetImage = torch.Tensor(Y)
        return inputImage, targetImage


# class TestDataset(Dataset):
#     def __init__(self, rootDir, transform=None):
#         self.rootDir = rootDir
#         self.means = [22775.8, 11.3, 7.8, 0.11]
#         self.stds = [924.8, 6.8, 6.9, 0.097]
#         self.transform = transform
#         self.batch_size = 8
#         self.fileNames = get_fileNames()[1]

#     def __len__(self):
#         return len(self.fileNames)

#     def __getitem__(self, idx):
#         X, Hs, Ws = list(), list(), list()
#         inputNames = self.fileNames[self.batch_size*idx : (idx+1)*self.batch_size]
#         for inputName in inputNames[:10]:
#             inputImage = tifffile.imread(inputName)    
#             inputImage = io.imread(inputName)[:,:,:-1]
#             inputImage = inputImage[:, :, ::-1]
#             K = 60000.0  # 180000.0 for sympascal
#             H, W = inputImage.shape[0], inputImage.shape[1]
#             sy = np.sqrt(K * H / W) / float(H)
#             sx = np.sqrt(K * W / H) / float(W)
#             inputImage = cv2.resize(inputImage, None, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
#             inputImage = inputImage.astype(np.float32)
#             inputImage = np.true_divide(np.subtract(inputImage, self.means), self.stds)
#             inputImage = inputImage.transpose((2, 0, 1))
#             X.append(inputImage)
#             Hs.append(H)
#             Ws.append(W)

#         X = np.array(X)
#         #print(X.shape)
#         #print(X[:,:-1].shape)
#         X = torch.Tensor(X)
#         return X, inputNames, Hs, Ws

class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.means = [22775.8, 11.3, 7.8, 0.11]
        self.stds = [924.8, 6.8, 6.9, 0.097]
        self.transform = transform
        self.fileNames = get_fileNames()[1]

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        inputName = self.fileNames[idx]

        image = tifffile.imread(inputName)
        inputImage = image[:,:,:-1]
        testImage = image[:, :, -1]
        #inputImage = io.imread(inputName)[:,:,:-1]
        inputImage = inputImage[:, :, ::-1]
        #K = 60000.0  # 180000.0 for sympascal
        H, W = inputImage.shape[0], inputImage.shape[1]
        #sy = np.sqrt(K * H / W) / float(H)
        #sx = np.sqrt(K * W / H) / float(W)
        #inputImage = cv2.resize(inputImage, None, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
        inputImage = inputImage.astype(np.float32)
        inputImage = np.true_divide(np.subtract(inputImage, self.means), self.stds)
        inputImage = inputImage.transpose((2, 0, 1))
        inputImage = torch.Tensor(inputImage[:,:,:-1])

        return inputImage, testImage, inputName, H, W


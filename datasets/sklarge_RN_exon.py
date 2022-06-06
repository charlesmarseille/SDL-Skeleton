import os
import numpy as np
import pandas as pd
import skimage.io as io
from torch.utils.data import Dataset
import torch
import cv2
import tifffile
import random
import matplotlib.pyplot as plt

def get_fileNames(dataset):
    dataset_train = []
    dataset_test = []
    if dataset == 'ryam':
        sup_data_paths = ['C:/Users/cmarseille/Documents/GitHub/SDL-Skeleton/datasets/dataset_camille_128/true_data/',       #ryam
            'C:/Users/cmarseille/Documents/GitHub/SDL-Skeleton/datasets/dataset_camille_128/false_data/',
            'C:/Users/cmarseille/Documents/GitHub/SDL-Skeleton/datasets/dataset_camille_128/true_data/',
            'C:/Users/cmarseille/Documents/GitHub/SDL-Skeleton/datasets/dataset_camille_128/false_data/']
        th_data = [1.0, 0.025, 1.0, 0.025]

        for i, path in enumerate(sup_data_paths):
            for file in os.listdir(path):
                if np.random.random() <= th_data[i]:
                    a = file.split('_')
                    _, idxi, idxj = file.split('_')
                    if int(idxi) <= 6000:
                        dataset_train.append(path+file)
                    else:
                        dataset_test.append(path+file)

    elif dataset == 'nord_cotiere':
        sup_data_paths = ['D:/cmarseille/reseaux/nordCotiere_128/A/true_data/',
            'D:/cmarseille/reseaux/nordCotiere_128/B/true_data/',
            'D:/cmarseille/reseaux/nordCotiere_128/A/false_data/',
            'D:/cmarseille/reseaux/nordCotiere_128/B/false_data/']
        th_data = [1.0, 0.1, 1.0, 0.25]
    
        for i, path in enumerate(sup_data_paths):
            for file in os.listdir(path):
                if np.random.random() <= th_data[i]:
                    a = file.split('_')
                    _, _, idxi, idxj = file.split('_')
                    if int(idxi) <= 1700:
                        dataset_train.append(path+file)
                    else:
                        dataset_test.append(path+file)
    return dataset_train, dataset_test


class TrainDataset(Dataset):
    def __init__(self, args, transform=None):
        self.args = args
        self.transform = transform
        if self.args.dataset == 'ryam':
            self.means = [22775.8, 7.8, 0.11]              #ryam no MHC
            self.stds = [924.8, 6.9, 0.097]                #ryam no MHC
            if self.args.with_mhc:
                self.means = [22775.8, 11.3, 7.8, 0.11]         #ryam
                self.stds = [924.8, 6.8, 6.9, 0.097]            #ryam
           
        elif self.args.dataset == 'nord_cotiere':
            self.means = [188.32, 294.6, 286.48, 1116.62]   #nord_cotiere
            self.stds = [332.36, 345.12, 355.55, 690.45]    #nord_cotiere
        self.batch_size = 8
        self.fileNames = get_fileNames(self.args.dataset)[0]

    def __len__(self):
        return int(np.floor(len(self.fileNames) / self.batch_size))

    def __getitem__(self, idx):
        inputNames = self.fileNames[self.batch_size*idx : (idx+1)*self.batch_size]
        X, Y = list(), list()
        for inputName in inputNames:
            image = tifffile.imread(inputName)       # image format is 256,256,5
            inputImage = image[:,:,[0,2,3]]     # remove MHC from dataset
            if self.args.with_mhc:
                inputImage = image[:,:,:-1]
            inputImage = np.true_divide(np.subtract(inputImage, self.means), self.stds)
            inputImage = inputImage[:, :, ::-1]
            inputImage = cv2.resize(inputImage, [256, 256])
            inputImage = inputImage.astype(np.float32)
            inputImage = inputImage.transpose((2, 0, 1))

            targetImage = image[:,:,-1]
            targetImage = cv2.resize(targetImage, [256, 256])
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


class TestDataset(Dataset):
    def __init__(self, args, transform=None):
        self.args = args
        if self.args.dataset == 'ryam':
            self.means = [22775.8, 7.8, 0.11]              #ryam no MHC
            self.stds = [924.8, 6.9, 0.097]                #ryam no MHC
            if self.args.with_mhc:
                self.means = [22775.8, 11.3, 7.8, 0.11]         #ryam
                self.stds = [924.8, 6.8, 6.9, 0.097]            #ryam

        elif self.args.dataset == 'nord_cotiere':
            self.means = [188.32, 294.6, 286.48, 1116.62]   #nord_cotiere
            self.stds = [332.36, 345.12, 355.55, 690.45]    #nord_cotiere

        self.transform = transform 
        self.fileNames = get_fileNames(self.args.dataset)[1]

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        inputName = self.fileNames[idx]

        image = tifffile.imread(inputName)
        inputImage = image[:,:,[0,2,3]]     # no MHC
        if self.args.with_mhc:
            inputImage = image[:,:,:-1]        # with MHC
        testImage = image[:, :, -1]
        inputImage = np.true_divide(np.subtract(inputImage, self.means), self.stds)
        inputImage = inputImage[:, :, ::-1]
        inputImage = cv2.resize(inputImage, [256, 256])
        testImage = cv2.resize(testImage, [256, 256])
        H, W = inputImage.shape[0], inputImage.shape[1]
        inputImage = inputImage.astype(np.float32)
        testImage = testImage.astype(np.float32)
        inputImage = inputImage.transpose((2, 0, 1))
        inputImage = torch.Tensor(inputImage)

        return inputImage, testImage, inputName, H, W


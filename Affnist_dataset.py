from __future__ import print_function, division
import pickle
import tarfile
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py

class AffNISTDataset(Dataset):
    def __init__(self, root_folder, mode):
        self.root_folder = root_folder
        self.current_file = ''
        self.current_index = 0
        self.x_train = ''
        self.y_train = ''
        self.x_test = ''
        self.y_test = ''
        self.mode = mode
        if mode == False:
            with h5py.File(f'{self.root_folder}/affnist_test.h5', 'r') as file:
                self.x_test = file['x_test'][:]
                self.y_test = file['y_test'][:]


    def __len__(self):
        if self.mode == True:
            return 169*60,000
        else:
            return 169*10,000

    def __getitem__(self, item):
        if self.mode == True:
            item_index = int(item / 169000 + 1)
            if item_index != self.current_index:
                self.current_index = item_index
                filename = f'affnist_train_{item_index}'
                with open(f'{self.root_folder}/{filename}','rb') as file:
                    (self.x_train, self.y_train) = pickle.load(file)

            c_index = item - 169000*(item_index-1)
            return [torch.Tensor(self.x_train[c_index]), torch.LongTensor(self.y_train[c_index].reshape(1,))]
        else:
            return [torch.Tensor(self.x_test[item]), torch.LongTensor(self.y_test[item].reshape(1,))]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset 
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import torchvision.models as models
import os
import seaborn as sns
import random


#We create our own custom Dataset following the example on this link: 
#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


class NetworkDataset(Dataset):

    def __init__(self, X,y=None):
            """
            Args:
            ----

                X (array):                 Features array
                y (string):                Target array
                
            """
            super(NetworkDataset, self).__init__()
            self.X=X
            self.y=y
            
    def __len__(self):
        return len(self.X)

    @staticmethod
    def transform_meta(l):
        o = np.array([[-1 for _ in range(5)] for i in range(8)])
        # o = [[-1, -1, - 1] for i in range(len(l))]
        for i, x in enumerate(l):
            if x[0] == "SMF":
                o[i, :2] = [10 ** (x[1][0]/10), 10 ** (x[1][1]/10)]
                o[i, -1] = 0
            else:
                o[i, 2:4] = [x[1][0]/ 24, x[1][1] / 10]
                o[i, -1] = 1
        return np.array(o)

    @staticmethod
    def gen_random_translated_meta(meta_matrix):
        n=len(l)
        start=random.choice(range(3,8))
        sub_matrix=meta_matrix[:n,:]
        translated_matrix=np.array([[-1 for _ in range(5)] for i in range(8)])
        translated_matrix[start:start+n,:]=sub_matrix
        return translated_matrix


    def _process_data(self, rebalance=False):
        meta = np.array([transform_meta(x[0]) for x in self.X])
        l = np.array([len(x[0]) for x in self.X])
        X = np.array([np.array(X_i) for X_i in self.X[:, 1]])
        if rebalance: 
            labels, counts = np.unique(l, return_counts=True)
            new_index = np.array([])
            count_max = counts.max()
            for label, count in zip(labels, counts):
                diff = count_max - count
                eq = np.where(l == label)[0]
                new_index = np.hstack((new_index,  eq, 
                                        np.random.choice(eq, diff, replace=True))).astype(int)
            new_index = np.random.permutation(new_index)
            meta = meta[new_index]
            self.l = l[new_index]
            self.X = X[new_index]
            if self.y is not None:
                self.y = self.y[new_index]
        if self.y is not None:
            self.y = torch.tensor(self.y, dtype=torch.float32)
        return  ({
            "X": torch.tensor(self.X, dtype=torch.float32), 
            "meta": torch.tensor(self.meta, dtype=torch.float32), 
            "l": torch.tensor(self.l, dtype=torch.long)
            }, y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        meta = np.array(transform_meta(self.X[idx][0]))
        l = np.array([len(self.X[idx][0])])
        X = np.array([np.array(self.X[idx, 1]))
        

        return {
            "X":X,
            "meta":meta,
            "l":l
                }

    def augment_data(self,ratio):
        n=len(self.X)
        processed_data=self._process_data()
        idx_list=random.choice(range(int(n*ratio)))
        for idx in idx_list :
            item=processed_data[idx]
            processed_data.append({"X":item["X"],
                                "meta": gen_random_translated_meta(item["meta"]),
                                "l":item["l"]})




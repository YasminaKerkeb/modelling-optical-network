import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
        for (i,x) in enumerate(meta_matrix):
            if (x==-1*np.ones(5)).all():
                index=i
                if index==7:
                    start=1
                break
            else: 
                index=8
                start=0

        sub_matrix=meta_matrix[0:index,:]
        translated_matrix=np.array([[-1 for _ in range(5)] for i in range(8)])
        translated_matrix[start:start+index,:]=sub_matrix
    
        return translated_matrix


    def _preprocess_data(self):
        meta = np.array([self.transform_meta(x[0]) for x in self.X])
        l = np.array([len(x[0]) for x in self.X])
        X = np.array([np.array(X_i) for X_i in self.X[:, 1]])
        return  {
            "X":X,
            "meta":meta,
            "l":l
                }
    def _process_data(self, ratio,rebalance=False):
        prep_data=self._preprocess_data()
        aug_data, y=self.augment_data(ratio,prep_data,self.y)
        meta = aug_data["meta"]
        l = aug_data["l"]
        X = aug_data["X"]
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
            l = l[new_index]
            X = X[new_index]
            if y is not None:
                y = y[new_index]
        
        return  ({
            "X": torch.tensor(X, dtype=torch.float32), 
            "meta": torch.tensor(meta, dtype=torch.float32), 
            "l": torch.tensor(l, dtype=torch.long)
            }, torch.tensor(y, dtype=torch.float32))

    def __getitem__(self, X,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        meta = np.array(X[idx][0])
        l = np.array([len(X[idx][0])])
        X = np.array([np.array(X[idx, 1])])
        
        return {
            "X":X,
            "meta":meta,
            "l":l
                }

    def augment_data(self,ratio,data,y):
        n=len(self.X)
        idx_list=random.sample(range(n),int(n*ratio))
        X,meta,l=list(data.values())
        for idx in idx_list :
            X=np.vstack((X,X[idx,:].reshape(1,-1)))
            transf_meta_matrix=self.gen_random_translated_meta(meta[idx,:,:])
            meta=np.concatenate([meta,transf_meta_matrix.reshape(1,8,5)],0)
            l=np.concatenate([l,[l[idx]]])
            y=np.vstack((y,y[idx,:].reshape(1,-1)))
        return ({"X":X, 
                "meta": meta,
                "l":l},
                y)

    @staticmethod
    def to_tensor(X,y):
        dataset = TensorDataset(*X.values(), y)
        return dataset



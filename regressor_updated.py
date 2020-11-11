# imports 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence
import abc
from sklearn.model_selection import train_test_split
import copy
 
 
eps = 1e-8
def em99_loss(y_true, y_pred):
    t = y_true.flatten()
    p = y_pred.flatten()
    ratio = (p[t!=0] + eps) / t[t!=0]
    score = torch.quantile(
            torch.abs(10 * torch.log10(ratio)), 0.99)
    return score
 
# def transform_meta(l):
#     o = np.array([[-1 for _ in range(5)] for i in range(8)])
#     # o = [[-1, -1, - 1] for i in range(len(l))]
#     for i, x in enumerate(l):
#         if x[0] == "SMF":
#             o[i, :2] = [10 ** (x[1][0]/10), 10 ** (x[1][1]/10)]
#             o[i, -1] = 0
#         else:
#             o[i, 2:4] = [10 ** (x[1][0]/ 24) / 100, 10 ** (x[1][1] / 10) / 10]
#             o[i, -1] = 1
#     return np.array(o)

def transform_meta(l):
    o = np.array([[-1 for _ in range(5)] for i in range(8)])
    # o = [[-1, -1, - 1] for i in range(len(l))]
    increm = 0
    for i, x in enumerate(l):
        if x[0] == "SMF":
            if i == 0:
                increm = 1
            o[i + increm, :2] = [10 ** (x[1][0]/10), 10 ** (x[1][1]/10)]
            o[i + increm, -1] = 0
        else:
            o[i + increm, 2:4] = [10 ** (x[1][0]/10)/ 100, 10 ** (x[1][1] / 10) / 10]
            o[i + increm, -1] = 1
    return np.array(o)

 
class BaseRegressor(abc.ABC):
    # inheriting classes should have properties:
    # - optimizer 
    # - net 
    # - loss_fn
    # the net should have the signature net(X, meta_data, cascade_lengths, target)
    def __init__(self):
        self.net = None
        self.optimizer = None
        self.loss_fn = None
        self.nepochs = 100
        self.rebalance_data = False
        self.batch_size = 128
        self.holdout = 0.1 # if 0, we don't hold out the data, other wise,
                            #  we hold out the fraction to make early stopping
        self.best_model = None
 
    def _process_data(self, X, y=None, rebalance=False):
        meta = np.array([transform_meta(x[0]) for x in X])
        l = np.array([len(x[0]) for x in X])
        X = np.array([np.array(X_i) for X_i in X[:, 1]])
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
        if y is not None:
            y = torch.tensor(y, dtype=torch.float32)
        return  ({
            "X": torch.tensor(X, dtype=torch.float32), 
            "meta": torch.tensor(meta, dtype=torch.float32), 
            "l": torch.tensor(l, dtype=torch.long)
        }, y)
    
 
    def train_epoch(self, x, met, l, y):
        self.net.train()
        self.optimizer.zero_grad()   
        o = self.net(x, met, l)
        on_chan = x != 0
        o = torch.mul(on_chan, o)
        y = torch.mul(on_chan, y)
        error = self.loss_fn(y, o)
        error.backward()
        self.optimizer.step()
    
    def fit(self, X, y, verbose=0):
        if self.holdout != 0:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=self.holdout)
            test_data = (Xte, yte)
        else: 
            Xtr, ytr = X, y
            test_data = None
        if test_data is not None:
            Xte_processed, yte = self._process_data(Xte, yte, rebalance=False)
            best_test_score = np.inf
        X_processed, y = self._process_data(Xtr, ytr, rebalance=self.rebalance_data)
        dataset = TensorDataset(*X_processed.values(), y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        k = 0
        for _ in range(self.nepochs):
            print('epoch :', _)
            for i, d in enumerate(dataloader):
                self.train_epoch(*d)
                if k % 10 == 9 and test_data is not None:
                    with torch.no_grad():
                        preds = self.net(*Xte_processed.values())
                        score = em99_loss(yte, preds)
                        if score < best_test_score:
                            best_test_score = score 
                            self.best_model = copy.deepcopy(self.net)
                k += 1
 
    def predict(self, X):
        self.net.eval()
        X_processed, _ = self._process_data(X, y=None)
        with torch.no_grad():
            if self.best_model is not None:
                o =  self.best_model(*X_processed.values()).detach().numpy() 
            else:
                o = self.net(*X_processed.values()).detach().numpy()
        return np.maximum(o, 0)
 
        
def myloss(y_true, y_pred):
    cond = y_true != 0
    rapport = (y_pred[cond] + 1e-8) / y_true[cond]
    log_rapport = 10 * torch.abs(torch.log(rapport))
    poids = torch.sigmoid(10/1.5 * log_rapport - 5)
    return (log_rapport * poids).mean()
 
 
class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=2, dilation=1,  padding=1), 
            nn.Tanh(), 
            nn.Conv1d(1, 1, kernel_size=2, dilation=1,  padding=0), 
            nn.Tanh(), 
            nn.Conv1d(1, 1, kernel_size=2, dilation=2,  padding=1),
            nn.Tanh(), 
            nn.Conv1d(1, 1, kernel_size=2, dilation=4,  padding=2),
            nn.Tanh(), 
            nn.Conv1d(1, 1, kernel_size=2, dilation=8,  padding=4),
            nn.Tanh(), 
            nn.Conv1d(1, 1, kernel_size=2, dilation=16,  padding=8),
            nn.Tanh(), 
            nn.Conv1d(1, 1, kernel_size=2, dilation=32,  padding=16),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x).squeeze(1)
    
class MetaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(5, 32), 
             nn.Tanh(), 
             nn.Linear(32, 7))
        self.net2 = nn.Sequential(nn.Linear(8, 16), 
             nn.Tanh(), 
             nn.Linear(16, 1), 
             nn.Flatten(start_dim=1))
    
    def forward(self, m):
        m = self.net1(m)
        m.transpose_(1, 2)
        return self.net2(m)
    
    
class TotalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = DilatedNet()
        self.meta = MetaNet()
        self.head = nn.Sequential(nn.Linear(32 + 7, 128), 
                                  nn.Tanh(),
                                  nn.Dropout(0.1), 
                                  nn.Linear(128, 128), 
                                  nn.Tanh(),
                                  nn.Dropout(0.1), 
                                  nn.Linear(128, 128), 
                                  nn.Dropout(0.1), 
                                  nn.Linear(128, 32), 
                                 )
        #self.init_weights()
        
    def forward(self, x, met, l):
        o = self.res(x) + x
        met = self.meta(met)
        o = self.head(torch.cat((o, met), dim=1))
        return 1 + o
    
    def init_weights(self):
        def init_apply(m):
            try:
                torch.nn.init.xavier_uniform_(m.weight)
            except: 
                pass
        self.apply(init_apply)
 
 
 
class Regressor(BaseRegressor):
    def __init__(self):
        super().__init__()
        self.net = TotalNet()
        self.batch_size = 128
        self.nepochs = 100
        self.holdout = 0.
        self.rebalance_data = True
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.loss_fn = nn.MSELoss()

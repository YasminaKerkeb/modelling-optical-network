# imports 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence

    
def transform_meta(l):
    o = np.array([[-1 for _ in range(5)] for i in range(8)])
    # o = [[-1, -1, - 1] for i in range(len(l))]
    for i, x in enumerate(l):
        if x[0] == "SMF":
            o[i, :2] = [x[1][0]/24, x[1][1]/20]
            o[i, -1] = 0
        else:
            o[i, 2:4] = [x[1][0]/24, x[1][1]/20]
            o[i, -1] = 1
    return np.array(o)

    
class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn =  nn.LSTM(input_size=5, hidden_size=50, num_layers=2, batch_first=True)
        self.head1 = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 50))
        self.head = nn.Sequential(nn.Linear(50, 40), nn.ReLU(), nn.Linear(40, 32))
        
    def forward(self, x, m, l):
        m = pack_padded_sequence(m, l, batch_first=True, enforce_sorted=False)
        m = self.rnn(m)[-1][1].squeeze(0)
        m = torch.cat((m[0], m[1]), dim=1)
        m = self.head1(m)
        return F.relu(1 + self.head(m)) * x



class Regressor:
    def __init__(self):
        self.net = LSTMNet()
        self.batch_size = 32
        self.nepochs = 30
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def _process_X(self, X):
        meta = np.array([transform_meta(x[0]) for x in X])
        l = np.array([len(x[0]) for x in X])
        X = np.array([np.array(X_i) for X_i in X[:, 1]])
        return  {
            "X": torch.tensor(X, dtype=torch.float32), 
            "meta": torch.tensor(meta, dtype=torch.float32), 
            "l": torch.tensor(l, dtype=torch.long)
        }

    def _train_epoch(self, x, met, l, y):
        self.optimizer.zero_grad()   
        o = self.net(x, met, l)
        error = self.loss_fn(y, o)
        error.backward()
        self.optimizer.step()
        return error.item()
    
    def fit(self, X, y, verbose=0):
        X_processed = self._process_X(X)
        y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(*X_processed.values(), y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.nepochs):
            total_error = 0
            for i, (x, m, l, y) in enumerate(dataloader):
                sub_error = self._train_epoch(x, m, l, y)
                total_error += sub_error
                if verbose: 
                    if i % 500 == 499:
                        print(np.sqrt(total_error / 500))
                        total_error = 0

    def predict(self, X):
        X_processed = self._process_X(X)
        with torch.no_grad():
            return self.net(*X_processed.values()).detach().numpy()    

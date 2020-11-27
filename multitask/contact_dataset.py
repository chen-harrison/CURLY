import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class contactDataset(Dataset):
    def __init__(self, data_path, stack_size):
        data = sio.loadmat(data_path)
        
        X = data['inputs']
        Y = data['contact_labels']
        # make sure we have a sample number divisible by the stack_size
        X = X[:(X.shape[0] // stack_size * stack_size),:]
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = Y[:(Y.shape[0] // stack_size * stack_size),:]

        self.X = X.reshape(-1, X.shape[1] * stack_size)
        self.Y = Y.reshape(-1, Y.shape[1] * stack_size)
        self.stack_size = stack_size
        self.size = self.X.shape[0]
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.X[idx].reshape(self.stack_size, -1)).unsqueeze(0).float()
        # use labels of last data point in stacked inputs - subject to change
        labels = torch.from_numpy(self.Y[idx].reshape(self.stack_size, -1)[-1]).long()

        return inputs, labels

if __name__ == '__main__':
    # testing
    data_path = '/home/harrison/Documents/CURLY/multitask/data/08292020_trial1.mat'
    stack_size = 5
    dataset = contactDataset(data_path, stack_size)
    print(len(dataset))
    x_0, y_0 = dataset[0]
    print(x_0.shape)
    print(y_0.shape)
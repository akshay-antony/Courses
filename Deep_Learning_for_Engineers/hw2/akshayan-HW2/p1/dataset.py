import numpy as np
import torch
from torch.utils.data import Dataset
import os


class FlowDataset(Dataset):
    def __init__(self, num_points=16, data_dir='flow/data', data_session='all_params_50625', mode='train'):
        data_fn = os.path.join(data_dir, 'dataset_' + data_session + '.npy')
        bc_fn = os.path.join(data_dir, 'bc_'+ data_session + '.npy')
        target = np.load(data_fn)
        inputs = np.load(bc_fn)
        
        # Pre-Process Data
        # replace first time step with all the boundary conditions
        # for now, hardcoded to fill in since there are 4 bc's and 17 points
        for i in range(len(inputs)):
            D = inputs[i][0]
            dpdx = inputs[i][1]
            mu = inputs[i][2]
            nu = inputs[i][3]
            for j in range(num_points+1):
                if 0 <= j < 5:
                    target[i][0][j] = D
                elif 5 <= j < 9:
                    target[i][0][j] = dpdx
                elif 9 <= j < 13:
                    target[i][0][j] = mu
                else:
                    target[i][0][j] = nu

        num_data = len(inputs)
        np.random.seed(0)
        # split the dataset inton training and test 
        test_idx = np.random.choice(num_data, num_data//5, replace=False).tolist()
        train_idx = list(set(range(num_data)) - set(test_idx))

        self.mode = mode
        if mode == 'train':
            self.data = target[train_idx,:,:].astype(np.float32)
        elif mode == 'test':
            self.data = target[test_idx,:,:].astype(np.float32)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.data[idx,:-1,:], self.data[idx,1:,:]
        elif self.mode == 'test':
            return self.data[idx,0,:], self.data[idx,1:,:]
    
    def __len__(self):
        return len(self.data)


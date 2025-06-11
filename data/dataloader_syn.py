import os, random, numpy as np, copy

from torch.utils.data import Dataset
import torch

import os

def seq_collate(data):

    (past_traj, future_traj) = zip(*data)
    past_traj = torch.stack(past_traj,dim=0)
    future_traj = torch.stack(future_traj,dim=0)
    data = {
        'past_traj': past_traj,
        'future_traj': future_traj,
        'seq': 'synthetic',
    }

    return data

class SYNDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, obs_len=5, pred_len=10, dataset_part = "train_rigid"
    ):
        super(SYNDataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        if dataset_part == "train_rigid":
            data_root = 'datasets/syn/syn_rigid_data.npy'
            self.trajs = np.load(data_root)
            self.trajs = self.trajs[:32500]

        elif dataset_part == "train_smooth":
            data_root = 'datasets/syn/syn_smooth_data.npy'
            self.trajs = np.load(data_root)
            self.trajs = self.trajs[:32500]

        elif dataset_part == "test_rigid" or dataset_part == "val_rigid":
            data_root = 'datasets/syn/syn_rigid_data.npy'
            self.trajs = np.load(data_root)
            self.trajs = self.trajs[32500:45000]
            print(self.trajs.shape)

        elif dataset_part == "test_smooth" or dataset_part == "val_smooth":
            data_root = 'datasets/syn/syn_smooth_data.npy'
            self.trajs = np.load(data_root)
            self.trajs = self.trajs[32500:45000]

        self.batch_len = len(self.trajs)
        print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs-self.trajs[:,self.obs_len-1:self.obs_len]).type(torch.float)

        # print(self.traj_abs.shape)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        past_traj = self.traj_abs[index, :, :self.obs_len, :]
        future_traj = self.traj_abs[index, :, self.obs_len:, :]
        out = [past_traj, future_traj]
        return out

    def for_GT(self):
        return self.trajs.transpose(0,2,1,3)


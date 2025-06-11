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
        'seq': 'nba',
    }

    return data

class NBADataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, obs_len=5, pred_len=10, dataset_part = "train"
    ):
        super(NBADataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        if dataset_part == "train_1":
            data_root = 'datasets/nba/target/train_trial_1.npy'
            self.trajs = np.load(data_root)
        elif dataset_part == "train_2":
            data_root = 'datasets/nba/target/train_trial_2.npy'
            self.trajs = np.load(data_root)
        elif dataset_part == "train1_train2":
            data_root_train = 'datasets/nba/target/train_trial_1.npy'
            data_root_train2 = 'datasets/nba/target/train_trial_2.npy'
            train_trajs = np.load(data_root_train)
            train2_trajs = np.load(data_root_train2)
            self.trajs = np.concatenate((train_trajs, train2_trajs), axis=0)
        elif dataset_part == "train_3":
            data_root = 'datasets/nba/target/train_trial_3.npy'
            self.trajs = np.load(data_root)
        elif dataset_part == "test" or dataset_part == "val":
            data_root = 'datasets/nba/target/test_trial.npy'
            self.trajs = np.load(data_root)
        elif dataset_part == "con":
            data_root =  'datasets/nba/target/test_2_games_412.npy'
            self.trajs = np.load(data_root)
        self.trajs /= (94/28) # Turn to meters

        # print("all_trajs", self.trajs.shape, self.trajs[:10, :, 0, :])

        if dataset_part == "test":
            self.trajs = self.trajs[:12500]
        elif dataset_part == "val":
            self.trajs = self.trajs[12500:17500]



        self.batch_len = len(self.trajs)
        print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs-self.trajs[:,self.obs_len-1:self.obs_len]).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0,2,1,3)
        self.traj_norm = self.traj_norm.permute(0,2,1,3)
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
        return self.trajs


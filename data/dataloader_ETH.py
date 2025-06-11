import logging
import os
import math
from collections import defaultdict
import pandas as pd
import numpy as np
from numpy.core.defchararray import endswith
from torch.utils.data import Sampler
import torch
from torch.utils.data import Dataset
import random
logger = logging.getLogger(__name__)


def seq_collate(data):

    (past_traj, future_traj) = zip(*data)
    past_traj = torch.stack(past_traj,dim=0)
    future_traj = torch.stack(future_traj,dim=0)
    data = {
        'past_traj': past_traj,
        'future_traj': future_traj,
        'seq': 'eth',
    }

    return data



def read_file(_path, delim='\t'):
    return pd.read_csv(_path, delim_whitespace=True, header=None).values





class TrajectoryDatasetETH(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=10,
        min_ped=1, delim='\t', save_path="seq_eth", test_scene = "seq_eth", mode = 'train'
    ):
        super(TrajectoryDatasetETH, self).__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.full_train_path = "datasets/eth/" + save_path + '_train.pt'
        self.full_test_path = "datasets/eth/" + save_path + '_test.pt'
        self.min_ped = min_ped
        self.test_scene = test_scene

        if os.path.exists(self.full_train_path):
            print(f"Loading cached dataset from {self.full_train_path} and {self.full_test_path}")
            train_data = torch.load(self.full_train_path,  weights_only=False)
            self.obs_traj_train = train_data["obs_traj"]
            self.pred_traj_train = train_data["pred_traj"]
            self.seq_start_end_train = train_data["seq_start_end"]
            self.grouped_seq_indices_train = train_data["grouped_seq_indices"]
            self.num_seq_train = len(self.seq_start_end_train)

            test_data = torch.load(self.full_test_path, weights_only=False)
            self.obs_traj_test = test_data["obs_traj"]
            self.pred_traj_test = test_data["pred_traj"]
            self.seq_start_end_test = test_data["seq_start_end"]
            self.grouped_seq_indices_test = test_data["grouped_seq_indices"]
            self.num_seq_test = len(self.seq_start_end_test)

            return

        print("Preprocessing dataset from raw files...")
        self._preprocess(data_dir,self.test_scene)

        torch.save({
            "obs_traj": self.obs_traj_test,
            "pred_traj": self.pred_traj_test,
            "seq_start_end": self.seq_start_end_test,
            "grouped_seq_indices": self.grouped_seq_indices_test
        }, self.full_test_path)

        torch.save({
            "obs_traj": self.obs_traj_train,
            "pred_traj": self.pred_traj_train,
            "seq_start_end": self.seq_start_end_train,
            "grouped_seq_indices": self.grouped_seq_indices_train
        }, self.full_train_path)
        print(f"Dataset cached to {self.full_test_path} and {self.full_train_path}")

    def set_mode(self, mode):
        assert mode in ['train', 'test']
        self.mode = mode

    def _preprocess(self, data_dir, test_scene):
        num_peds_in_seq_train = []
        seq_list_train = []
        num_peds_in_seq_test = []
        seq_list_test = []

        scenes = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
        print("scenes", scenes)
        for scene in scenes:
            full_scene_path = os.path.join(self.data_dir, scene)
            all_files = [os.path.join(full_scene_path, f) for f in os.listdir(full_scene_path) if f.endswith("_clean.txt")]

            for path in all_files:
                data = read_file(path, self.delim)
                frames = np.unique(data[:, 0]).tolist()

                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :]) #joint frames - for one frame - who is in it
                seq_counter = 0
                num_sequences = int(
                    math.ceil((len(frames) - self.seq_len + 1) / self.skip))
                print(f" in {path} num_sequences", num_sequences)
                for idx in range(0, len(frames) - self.seq_len + 1, self.skip):
                    curr_seq_data = np.concatenate(
                        frame_data[idx:idx + self.seq_len], axis=0) #takes 20 frames at a time
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) #who is in this frames
                    curr_seq = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))
                    curr_loss_mask = np.zeros((len(peds_in_curr_seq),self.seq_len))

                    num_peds_considered = 0

                    for _, ped_id in enumerate(peds_in_curr_seq):

                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        # print("curr_ped_seq", curr_ped_seq[0, 0], curr_ped_seq[-1, 0], frames.index(curr_ped_seq[0, 0]) - idx, frames.index(curr_ped_seq[-1, 0]) - idx + 1)
                        # print("idx", idx)
                        # print("first few raw XY:", curr_ped_seq[:5, 2], curr_ped_seq[:5, 3],curr_ped_seq.shape )

                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        ped_frames = curr_ped_seq[:, 0]
                        expected_frames = frames[idx:idx + self.seq_len]

                        if not np.all(np.isin(expected_frames, ped_frames)):
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) #-> 2, seq_length
                        # print("curr_ped_seq tran", curr_ped_seq[:, :5], curr_ped_seq.shape)
                        _idx = num_peds_considered
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq #N, 2, 20
                        curr_loss_mask[_idx, pad_front:pad_end] = 1
                        num_peds_considered += 1

                    if num_peds_considered > self.min_ped:
                        seq_counter += 1
                        agents_in_seq = curr_seq[:num_peds_considered]
                        num_chunks = int(np.ceil(agents_in_seq.shape[0] / 8))

                        for i in range(num_chunks):
                            start_idx = i * 8
                            end_idx = min((i + 1) * 8, agents_in_seq.shape[0])
                            chunk = agents_in_seq[start_idx:end_idx]

                            if scene == test_scene:
                                num_peds_in_seq_test.append(chunk.shape[0])
                                seq_list_test.append(chunk)
                            else:
                                num_peds_in_seq_train.append(chunk.shape[0])
                                seq_list_train.append(chunk)
                                # if end_idx - start_idx == 4:
                                    # print("chunk", chunk, chunk.shape)
                print("remained", seq_counter)
                print("train len", len(seq_list_train))
                print("test len", len(seq_list_test))

        self.num_seq_test = len(seq_list_test) #final number of sequences
        seq_list_test = np.concatenate(seq_list_test, axis=0)

        self.num_seq_train = len(seq_list_train) #final number of sequences
        seq_list_train = np.concatenate(seq_list_train, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj_train = torch.from_numpy(
            seq_list_train[:, :, :self.obs_len]).type(torch.float).permute(0, 2, 1) #N*seq_list, 8, 4
        self.pred_traj_train = torch.from_numpy(
            seq_list_train[:, :, self.obs_len:]).type(torch.float).permute(0, 2, 1) #N*seq_list, 12, 4

        self.obs_traj_test = torch.from_numpy(
            seq_list_test[:, :, :self.obs_len]).type(torch.float).permute(0, 2, 1) #N*seq_list, 8, 4
        self.pred_traj_test = torch.from_numpy(
            seq_list_test[:, :, self.obs_len:]).type(torch.float).permute(0, 2, 1) #N*seq_list, 12, 4

        print("self.obs_traj_test", self.obs_traj_test[0, :, 0], self.obs_traj_test[0, :, 1])
        cum_start_idx_Test = [0] + np.cumsum(num_peds_in_seq_test).tolist()
        self.seq_start_end_test = [
            (start, end)
            for start, end in zip(cum_start_idx_Test, cum_start_idx_Test[1:])
        ]
        cum_start_idx_train = [0] + np.cumsum(num_peds_in_seq_train).tolist()
        self.seq_start_end_train = [
            (start, end)
            for start, end in zip(cum_start_idx_train, cum_start_idx_train[1:])
        ]


        self.grouped_seq_indices_test = defaultdict(list)
        for i, (start, end) in enumerate(self.seq_start_end_test):
            agent_count = end - start
            self.grouped_seq_indices_test[agent_count].append(i)

        self.grouped_seq_indices_train = defaultdict(list)
        for i, (start, end) in enumerate(self.seq_start_end_train):
            agent_count = end - start
            self.grouped_seq_indices_train[agent_count].append(i)

        print("grouped_seq_indices_test", self.grouped_seq_indices_test[4])
    def __len__(self):
        if self.mode == 'train':
            return self.num_seq_train
        else:
            return self.num_seq_test

    def __getitem__(self, index):
        if self.mode == 'train':
            start, end = self.seq_start_end_train[index]
            return self.obs_traj_train[start:end, :, :2], self.pred_traj_train[start:end, :, :2]
        else:
            # print("index", index)
            # print("self.seq_start_end_test", self.seq_start_end_test, len(self.seq_start_end_train))
            start, end = self.seq_start_end_test[index]
            return self.obs_traj_test[start:end, :, :2], self.pred_traj_test[start:end, :, :2]



class GroupedBatchSampler(Sampler):
    def __init__(self, grouped_indices, batch_size, shuffle=True, drop_last=False):
        """
        Args:
            grouped_indices: dict of {agent_count: dataset indices}
            batch_size: number of samples per batch
            shuffle: whether to shuffle within groups and between groups
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.batches = []
        for agent_count, indices in grouped_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                if drop_last and len(batch) < batch_size:
                    continue
                self.batches.append(batch)

        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)



################ creation test ##############
# from torch.utils.data import DataLoader
# for scene in ["seq_eth", "seq_hotel"]:
# # for scene in ["seq_eth", "seq_hotel", "uni", "zara_01", "zara_02"]:
#     dataset_train = TrajectoryDatasetETH(data_dir="../datasets/eth/raw/", obs_len=8, pred_len=12, skip=1, min_ped=1, delim='space', test_scene=scene,save_path=scene, mode = "train")
#     dataset_test = TrajectoryDatasetETH(data_dir="../datasets/eth/raw/", obs_len=8, pred_len=12, skip=1, min_ped=1, delim='space', test_scene=scene ,save_path=scene, mode = "test")
#
#     train_loader = DataLoader(dataset_train, batch_sampler=GroupedBatchSampler(dataset_train.grouped_seq_indices_train, batch_size=32,shuffle=True,
#         drop_last=False), collate_fn = seq_collate)
#
#     test_loader = DataLoader(dataset_test, batch_sampler=GroupedBatchSampler(dataset_test.grouped_seq_indices_test, batch_size=32,shuffle=True,
#         drop_last=False), collate_fn = seq_collate)
#
# for key, item in dataset_train.grouped_seq_indices_train.items():
#     print(key, len(item))
# # max_x = 0
# max_y = 0
# min_x = float('inf')
# min_y = float('inf')

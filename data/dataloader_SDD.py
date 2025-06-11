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
from sklearn.model_selection import train_test_split


def seq_collate(data):

    (past_traj, future_traj) = zip(*data)
    past_traj = torch.stack(past_traj,dim=0)
    future_traj = torch.stack(future_traj,dim=0)
    data = {
        'past_traj': past_traj,
        'future_traj': future_traj,
        'seq': 'sdd',
    }

    return data


def read_file(_path, delim='\t'):
    return pd.read_csv(_path, delim_whitespace=True, header=None).values





class TrajectoryDatasetSDD(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=10,
        min_ped=1, delim='\t', save_path="datasets/sdd/SDD.pt", mode = 'train'
    ):

        super(TrajectoryDatasetSDD, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.save_path = save_path
        self.min_ped = min_ped
        self.mode = mode
        self.train_path = save_path.replace(".pt", "_train.pt")
        self.val_path = save_path.replace(".pt", "_val.pt")
        self.test_path = save_path.replace(".pt", "_test.pt")

        print( self.train_path)
        if os.path.exists(self.train_path) and os.path.exists(self.val_path) and os.path.exists(self.test_path):
            print(f"Loading cached datasets from {save_path}")
            train_data = torch.load(self.train_path, weights_only=False)
            val_data = torch.load(self.val_path, weights_only=False)
            test_data = torch.load(self.test_path, weights_only=False)

            self.obs_traj = train_data["obs_traj"]
            self.pred_traj = train_data["pred_traj"]
            self.seq_start_end_train = train_data["seq_start_end"]
            self.grouped_seq_indices_train = train_data["grouped_seq_indices"]

            self.obs_traj = val_data["obs_traj"]
            self.pred_traj = val_data["pred_traj"]
            self.seq_start_end_val = val_data["seq_start_end"]
            self.grouped_seq_indices_val = val_data["grouped_seq_indices"]

            self.obs_traj = test_data["obs_traj"]
            self.pred_traj = test_data["pred_traj"]
            self.seq_start_end_test = test_data["seq_start_end"]
            self.grouped_seq_indices_test = test_data["grouped_seq_indices"]

            self.num_seq_train = len(self.seq_start_end_train)
            self.num_seq_val = len(self.seq_start_end_val)
            self.num_seq_test = len(self.seq_start_end_test)

            print(self.num_seq_train, self.num_seq_val, self.num_seq_test)

            # agents_train = set()
            # for start, end in self.seq_start_end_train:
            #     agents_train.update(range(start, end))
            #
            # agents_val = set()
            # for start, end in self.seq_start_end_val:
            #     agents_val.update(range(start, end))
            #
            # agents_test = set()
            # for start, end in self.seq_start_end_test:
            #     agents_test.update(range(start, end))
            #
            # print("Overlap train/val:", len(agents_train.intersection(agents_val)))
            # print("Overlap train/test:", len(agents_train.intersection(agents_test)))
            # print("Overlap val/test:", len(agents_val.intersection(agents_test)))

            return

        print("Preprocessing dataset from raw files...")
        self._preprocess(data_dir)

        torch.save({
            "obs_traj": self.obs_traj,
            "pred_traj": self.pred_traj,
            "seq_start_end": self.seq_start_end_train,
            "grouped_seq_indices": self.grouped_seq_indices_train
        }, self.train_path)

        torch.save({
            "obs_traj": self.obs_traj,
            "pred_traj": self.pred_traj,
            "seq_start_end": self.seq_start_end_val,
            "grouped_seq_indices": self.grouped_seq_indices_val
        }, self.val_path)

        torch.save({
            "obs_traj": self.obs_traj,
            "pred_traj": self.pred_traj,
            "seq_start_end": self.seq_start_end_test,
            "grouped_seq_indices": self.grouped_seq_indices_test
        }, self.test_path)
        print(f"Dataset cached to {self.test_path} etc")



    def set_mode(self, mode):
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val' or 'test'"
        self.mode = mode

    def _preprocess(self, data_dir):
        num_peds_in_seq = []
        seq_list = []

        scenes = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
        print("scenes", scenes)
        for scene in scenes:
            full_scene_path = os.path.join(self.data_dir, scene)
            all_files = [os.path.join(full_scene_path, f) for f in os.listdir(full_scene_path) if f.endswith("_clean.txt")]

            for path in all_files:
                data = read_file(path, self.delim)
                frames = np.unique(data[:, 0]).tolist()
                frames = frames[::12]
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :]) #joint frames - for one frame - who is in it
                seq_counter = 0
                num_sequences = int(
                    math.ceil((len(frames) - self.seq_len + 1) / self.skip))
                print(f" in {path} num_sequences", num_sequences)
                for idx in range(0, len(frames) - self.seq_len + 1, self.skip):
                    curr_seq_data = np.concatenate(
                        frame_data[idx:idx + self.seq_len], axis=0) #takes 20 frmes at a time
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) #who is in this frames
                    curr_seq = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))
                    curr_loss_mask = np.zeros((len(peds_in_curr_seq),self.seq_len))

                    num_peds_considered = 0

                    for _, ped_id in enumerate(peds_in_curr_seq):

                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        # print("curr_ped_seq", curr_ped_seq[0, 0], curr_ped_seq[-1, 0], frames.index(curr_ped_seq[0, 0]) - idx, frames.index(curr_ped_seq[-1, 0]) - idx + 1)
                        # print("idx", idx)

                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        ped_frames = curr_ped_seq[:, 0]
                        expected_frames = frames[idx:idx + self.seq_len]

                        if not np.all(np.isin(expected_frames, ped_frames)):
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) #-> 2, seq_length
                        # print("curr_ped_seq tran", curr_ped_seq)
                        _idx = num_peds_considered
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq #N, 2, 20
                        curr_loss_mask[_idx, pad_front:pad_end] = 1
                        num_peds_considered += 1

                    if num_peds_considered > self.min_ped:
                        seq_counter += 1
                        # Normalize ]
                        curr_seq[:num_peds_considered, 0, :] = ((curr_seq[:num_peds_considered, 0, :] - 9.0) / (1951.0 - 9.0)) * 100
                        curr_seq[:num_peds_considered, 1, :] = ((curr_seq[:num_peds_considered, 1, :] - 7.0) / (1973.0 - 7.0)) * 100
                        agents_in_seq = curr_seq[:num_peds_considered]  # (num_agents, 4, seq_len)

                        agent_indices = list(range(agents_in_seq.shape[0]))
                        used_agents = set()

                        while len(used_agents) < agents_in_seq.shape[0]:
                            group = []
                            available_agents = [idx for idx in agent_indices if idx not in used_agents]
                            if not available_agents:
                                break

                            # a new group from first available agent
                            anchor = available_agents[0]
                            group.append(anchor)
                            used_agents.add(anchor)

                            anchor_traj = agents_in_seq[anchor, :2, :]  # (2, seq_len), x-y only

                            for other in available_agents[1:]:
                                if len(group) >= 8:
                                    break
                                other_traj = agents_in_seq[other, :2, :]  # (2, seq_len)

                                # mean distance over the whole sequence
                                # dists = np.linalg.norm(anchor_traj.T - other_traj.T, axis=1)  # (seq_len,)
                                diffs = anchor_traj.T[:, None, :] - other_traj.T[None, :, :]
                                dists = np.linalg.norm(diffs, axis=-1)

                                if np.any(dists <= 20):
                                    group.append(other)
                                    used_agents.add(other)

                            group_agents = agents_in_seq[group]  # (group_size, 4, seq_len)
                            # print("group_agents.shape[0]", group_agents.shape[0])
                            num_peds_in_seq.append(group_agents.shape[0])
                            seq_list.append(group_agents)
                print("remained", seq_counter)


        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        seq_list = np.concatenate(seq_list, axis=0)

        full_grouped = defaultdict(list)
        for i, (start, end) in enumerate(self.seq_start_end):
            agent_count = end - start
            full_grouped[agent_count].append(i)

        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        self.grouped_seq_indices_train = defaultdict(list)
        self.grouped_seq_indices_val = defaultdict(list)
        self.grouped_seq_indices_test = defaultdict(list)

        train_counter = 0
        val_counter = 0
        test_counter = 0

        for agent_count, indices in full_grouped.items():
            if len(indices) < 3:
                for idx in indices:
                    self.grouped_seq_indices_train[agent_count].append(train_counter)
                    self.train_indices.append(idx)
                    train_counter += 1
                continue

            train, temp = train_test_split(indices, test_size=0.35, random_state=42)

            for idx in train:
                self.grouped_seq_indices_train[agent_count].append(train_counter)
                self.train_indices.append(idx)
                train_counter += 1

            if len(temp) > 2:
                val, test = train_test_split(temp, test_size=0.7, random_state=42)

                for idx in val:
                    self.grouped_seq_indices_val[agent_count].append(val_counter)
                    self.val_indices.append(idx)
                    val_counter += 1

                for idx in test:
                    self.grouped_seq_indices_test[agent_count].append(test_counter)
                    self.test_indices.append(idx)
                    test_counter += 1
            else:
                for idx in temp:
                    self.grouped_seq_indices_train[agent_count].append(train_counter)
                    self.train_indices.append(idx)
                    train_counter += 1

        # Build datasets
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).float().permute(
            0, 2, 1)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).float().permute(
            0, 2, 1)
        self.seq_start_end_train = [self.seq_start_end[i] for i in self.train_indices]
        self.seq_start_end_val = [self.seq_start_end[i] for i in self.val_indices]
        self.seq_start_end_test = [self.seq_start_end[i] for i in self.test_indices]

        self.num_seq_train = len(self.seq_start_end_train)
        self.num_seq_val = len(self.seq_start_end_val)
        self.num_seq_test = len(self.seq_start_end_test)


    def __len__(self):
        if self.mode == 'train':
            return self.num_seq_train
        elif self.mode == 'val':
            return self.num_seq_val
        else:
            return self.num_seq_test

    def __getitem__(self, index):
        if self.mode == 'train':
            start, end = self.seq_start_end_train[index]
        elif self.mode == 'val':
            start, end = self.seq_start_end_val[index]
        else:
            start, end = self.seq_start_end_test[index]

        obs = self.obs_traj[start:end, :, :2]
        pred = self.pred_traj[start:end, :, :2]
        # obs_norm, pred_norm = self.normalize_trajectories(obs, pred)

        return obs, pred


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




class AnalyzeSingleSDDFile(Dataset):
    def __init__(self, file_path, obs_len=8, pred_len=292, skip=1, delim='tab'):
        super(AnalyzeSingleSDDFile, self).__init__()
        self.file_path = file_path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.skip = skip
        self.delim = delim

        self.output_pt_path = self.file_path.replace("_clean.txt", "_special_selected.pt")
        self.filtered_clean_path = self.file_path.replace("_clean.txt", "_clean_filtered.txt")

        if os.path.exists(self.output_pt_path) and os.path.exists(self.filtered_clean_path):
            print(f"Found existing processed files: {self.output_pt_path}, loading them!")
            special = torch.load(self.output_pt_path)
            self.obs_traj = special["obs_traj"]
            self.pred_traj = special["pred_traj"]
            self.seq_start_end = special["seq_start_end"]
        else:
            print(f"Analyzing and processing file: {file_path}")
            self._preprocess()

    def _preprocess(self):
        data = read_file(self.file_path, self.delim)
        frames_original = np.unique(data[:, 0]).tolist()
        frames = frames_original[::12]
        # print("frames 12", frames[:50])


        frame_data = [data[data[:, 0] == frame, :] for frame in frames]
        # print("frame_data", frame_data[0].shape)

        seq_list = []
        num_peds_in_seq = []
        self.seq_ped_ids = []
        self.seq_frames= []

        for idx in range(0, len(frames) - self.seq_len + 1, self.skip):
            # print("idx", idx)
            curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0) #take 300 frames
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) #see whi is in it
            curr_seq = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))


            num_peds_considered = 0
            ped_ids_in_this_seq = []
            for ped_id in peds_in_curr_seq:
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                expected_frames = frames[idx:idx + self.seq_len]
                ped_frames = curr_ped_seq[:, 0]

                if not np.all(np.isin(expected_frames, ped_frames)):
                    continue

                # print("ped_id", ped_id)
                curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])  # (2, seq_len)
                curr_seq[num_peds_considered, :, pad_front:pad_end] = curr_ped_seq


                num_peds_considered += 1
                ped_ids_in_this_seq.append(ped_id)


            if num_peds_considered > 1:
                num_peds_in_seq.append(num_peds_considered)
                seq_list.append(curr_seq[:num_peds_considered])
                if num_peds_considered in [4, 6, 8]:
                    print("num_peds_considered", num_peds_considered)
                    print("last idx frames - 50", frames[idx:idx + self.seq_len][:50])
                self.seq_ped_ids.append(ped_ids_in_this_seq)
                self.seq_frames.append(frames[idx:idx + self.seq_len])

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        # print("cum_start_idx", cum_start_idx)
        full_seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        print("full_seq_start_end", full_seq_start_end)
        seq_list = np.concatenate(seq_list, axis=0)
        print("seq_list", seq_list.shape) #N, 4, 300

        full_grouped = defaultdict(list)
        for i, (start, end) in enumerate(full_seq_start_end):
            agent_count = end - start
            full_grouped[agent_count].append(i)

        for agent_count, indices in full_grouped.items():
            print(f"full_grouped {agent_count} agents: {len(indices)} sequences")


        selected_sequences = []
        selected_seq_start_end = []
        selected_agent_ids = set()
        current_idx = 0
        target_groups = [4, 6, 8]
        sequences_needed = 2
        selected_frames = set()

        for group_size in target_groups:
            print("group_size", group_size)
            indices = full_grouped.get(group_size, [])
            if len(indices) < sequences_needed:
                print(f"Not enough sequences for group size {group_size}")
                continue
            for idx in indices[:sequences_needed]:
                start, end = full_seq_start_end[idx]
                print("start end", start, end)
                selected_sequences.append(seq_list[start:end])
                selected_seq_start_end.append((current_idx, current_idx + (end - start)))
                current_idx += (end - start)
                #start - end, from all sequences i collected - the start and ending point of a group
                selected_agent_ids.update(self.seq_ped_ids[idx])
                selected_frames.update(self.seq_frames[idx])
                print("selected_agent_ids", selected_agent_ids)

        if not selected_sequences:
            print("No sequences selected.")
            return

        selected_sequences_np = np.concatenate(selected_sequences, axis=0)
        print(f"Selected shape: {selected_sequences_np.shape}")

        obs_traj = torch.from_numpy(selected_sequences_np[:, :, :self.obs_len]).float().permute(0, 2, 1)
        pred_traj = torch.from_numpy(selected_sequences_np[:, :, self.obs_len:]).float().permute(0, 2, 1)
        self.obs_traj = obs_traj
        self.pred_traj = pred_traj
        self.seq_start_end = selected_seq_start_end

        torch.save({
            "obs_traj": self.obs_traj,
            "pred_traj": self.pred_traj,
            "seq_start_end": self.seq_start_end
        }, self.output_pt_path)
        print(f"Saved .pt torch file to {self.output_pt_path}")
        print("selected_agent_ids", selected_agent_ids)


        frames_to_remove_full = set()

        for agent_id in selected_agent_ids:
            agent_rows = data[data[:, 1] == agent_id]
            agent_frames = agent_rows[:, 0]
            sampled_agent_frames = [f for f in agent_frames if f in selected_frames]

            if len(sampled_agent_frames) == 0:
                continue

            min_frame = min(sampled_agent_frames)
            max_frame = max(sampled_agent_frames)

            frames_in_range = [f for f in frames_original if min_frame <= f <= max_frame]

            print(f"Agent {int(agent_id)} - Removing frames {min_frame} to {max_frame} ({len(frames_in_range)} frames)")

            for frame in frames_in_range:
                frames_to_remove_full.add((frame, agent_id))

        mask_keep = np.array([
            (frame, agent_id) not in frames_to_remove_full
            for frame, agent_id in zip(data[:, 0], data[:, 1])
        ])
        data_filtered = data[mask_keep]

        pd.DataFrame(data_filtered).to_csv(self.filtered_clean_path, index=False, header=False, sep=' ')
        print(f"Saved filtered clean file to {self.filtered_clean_path}")
        print(f"Original rows: {data.shape[0]} -> After filtering: {data_filtered.shape[0]} -> Removed {data.shape[0] - data_filtered.shape[0]} rows")

    def __len__(self):
        return len(self.seq_start_end)

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        return self.obs_traj[start:end], self.pred_traj[start:end]




#### ------------ testing for creating data--------------
# from torch.utils.data import DataLoader
# #
# dataset_train = TrajectoryDatasetSDD(data_dir="../datasets/SDD/raw", obs_len=8, pred_len=12, skip=1, min_ped=1, delim='space', save_path="../datasets/SDD/SDD.pt", mode="train")
# dataset_test = TrajectoryDatasetSDD(data_dir="../datasets/SDD/raw", obs_len=8, pred_len=12, skip=1, min_ped=1, delim='space', save_path="../datasets/SDD/SDD.pt", mode="test")
# dataset_val = TrajectoryDatasetSDD(data_dir="../datasets/SDD/raw", obs_len=8, pred_len=12, skip=1, min_ped=1, delim='space', save_path="../datasets/SDD/SDD.pt", mode="val")
#
# train_loader = DataLoader(dataset_train,
#                           batch_sampler=GroupedBatchSampler(dataset_train.grouped_seq_indices_train, batch_size=32,
#                                                             shuffle=True,
#                                                             drop_last=False), collate_fn = seq_collate)
#
# test_loader = DataLoader(dataset_test,
#                          batch_sampler=GroupedBatchSampler(dataset_test.grouped_seq_indices_test, batch_size=32,
#                                                            shuffle=True,
#                                                            drop_last=False), collate_fn = seq_collate)
#
# val_loader = DataLoader(dataset_val,
#                          batch_sampler=GroupedBatchSampler(dataset_val.grouped_seq_indices_val, batch_size=32,
#                                                            shuffle=True,
#                                                            drop_last=False), collate_fn = seq_collate)
#
# # print(len(dataset_train))
# for key, item in dataset_train.grouped_seq_indices_train.items():
#     print(key, len(item))
# for key, item in dataset_test.grouped_seq_indices_test.items():
#     print(key, len(item))
# for key, item in dataset_val.grouped_seq_indices_val.items():
#     print(key, len(item))

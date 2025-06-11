import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np





class FishTrajectoryDataset3(Dataset):
    def __init__(self, data_30, data_25, data_35, transpose=False):

        self.data_30 = pd.read_csv(data_30)
        self.data_25 = pd.read_csv(data_25)
        self.data_35 = pd.read_csv(data_35)

        self.transpose = transpose
        #grouping data by target ID (xN, yN pairs) and reshaping into batches
        self.fish_ids = [col.split('X')[1] for col in self.data_30.columns if col.startswith('X')]

        self.data_30 = {fish_id: self.data_30[[f'X{fish_id}', f'Y{fish_id}']].values for fish_id in self.fish_ids}

        self.data_25 = {fish_id: self.data_25[[f'X{fish_id}', f'Y{fish_id}']].values for fish_id in self.fish_ids}
        self.data_35 = {fish_id: self.data_35[[f'X{fish_id}', f'Y{fish_id}']].values for fish_id in self.fish_ids}

        self.num_frames_30 = len(self.data_30["0"])
        self.num_frames_25 = len(self.data_25["0"])

        print("self.num_frames_25", self.num_frames_25)
        self.num_frames_35 = len(self.data_35["0"])

        print("self.num_frames_35", self.num_frames_35)


    def __len__(self):
        #number of fishes
        return len(self.fish_ids)


    def get_trajectories(self):
        traj_num_30 = self.num_frames_30 // 60  -2 #350138/60 = 5835 traj total
        print("traj_num_30", traj_num_30)
        traj_num_25 = self.num_frames_25 // 50 -2  #106000/75 = 1413 traj total
        traj_num_35 = self.num_frames_35 // 70 -2 #415409/105  = 3956 traj total ----> total concat wil be 9259 *0.65 --->6018

        all_all_fish_loc = [] #shape (N,15, 20, 2) ->number of trajectories (3900), frames per 1 trj, target, x.y coords
        for i in range(traj_num_30):
            all_fish_loc = [] #shape 15, 20, 2)
            for j in range(15):
                time_stamp = i*60 + j*12 #each data point here is 0.4sec and we have 15 of those = 6 seconds (use 2 to predict 4)
                curr_frame = [self.data_30[fish_id][time_stamp] for fish_id in self.fish_ids]
                all_fish_loc.append(curr_frame)
            all_all_fish_loc.append(all_fish_loc)# preparing 3900 different trajectories

        for i in range(traj_num_25):
            all_fish_loc = [] #shape 15, 20, 2)
            for j in range(15):
                time_stamp = i*50 + j*10
                curr_frame = [self.data_25[fish_id][time_stamp] for fish_id in self.fish_ids]
                all_fish_loc.append(curr_frame)
            all_all_fish_loc.append(all_fish_loc)


        for i in range(traj_num_35):
            all_fish_loc = [] #shape 15, 20, 2)
            for j in range(15):
                time_stamp = i*70 + j*14
                curr_frame = [self.data_35[fish_id][time_stamp] for fish_id in self.fish_ids]
                all_fish_loc.append(curr_frame)
            all_all_fish_loc.append(all_fish_loc)


        all_all_fish_loc = np.array(all_all_fish_loc, dtype=np.float32)
        return all_all_fish_loc

    def get_no_overlap_trajectories(self):
        traj_num_30 = self.num_frames_30 // 180 #350138/60 = 5835 traj total
        print("traj_num_30", traj_num_30)
        traj_num_25 = self.num_frames_25 //150  #106000/75 = 1413 traj total
        traj_num_35 = self.num_frames_35 // 210 #415409/105  = 3956 traj total ----> total concat wil be 9259 *0.65 --->6018

        all_all_fish_loc = [] #shape (N,15, 20, 2) ->number of trajectories (3900), frames per 1 trj, target, x.y coords
        for i in range(traj_num_30):
            start_time = i * 180
            all_fish_loc = [] #shape 15, 20, 2)
            for j in range(15):
                time_stamp = start_time + j*12 #each data point here is 0.4sec and we have 15 of those = 6 seconds (use 2 to predict 4)
                curr_frame = [self.data_30[fish_id][time_stamp] for fish_id in self.fish_ids]
                all_fish_loc.append(curr_frame)
            all_all_fish_loc.append(all_fish_loc)# preparing 3900 different trajectories

        for i in range(traj_num_25):
            all_fish_loc = [] #shape 15, 20, 2)
            start_time = i * 150
            for j in range(15):
                time_stamp =start_time + j*10
                curr_frame = [self.data_25[fish_id][time_stamp] for fish_id in self.fish_ids]
                all_fish_loc.append(curr_frame)
            all_all_fish_loc.append(all_fish_loc)


        for i in range(traj_num_35):
            all_fish_loc = [] #shape 15, 20, 2)
            start_time = i * 210
            for j in range(15):
                time_stamp = start_time + j*14
                curr_frame = [self.data_35[fish_id][time_stamp] for fish_id in self.fish_ids]
                all_fish_loc.append(curr_frame)
            all_all_fish_loc.append(all_fish_loc)


        all_all_fish_loc = np.array(all_all_fish_loc, dtype=np.float32)

        return all_all_fish_loc


data_target = 'target'
file_path_30 = 'source/fish_coords_48_normalized_100.csv'
file_path_25 = 'source/fish_coords_36_normalized_100.csv'
file_path_35 = 'source/fish_coords_20_normalized_100.csv'

dataset = FishTrajectoryDataset3(file_path_30 ,file_path_25,file_path_35, True)
all_trajs = dataset.get_trajectories()
print(len(all_trajs))
index = list(range(len(all_trajs)))
import random
from random import shuffle
random.seed(0)
shuffle(index)
train_set = all_trajs[index[:9023]]
test_set = all_trajs[index[9023:]]
print('train num:',train_set.shape[0])
print('test num:',test_set.shape[0])


np.save(data_target+'/train_overlap.npy',train_set)
np.save(data_target+'/test_overlap.npy',test_set)

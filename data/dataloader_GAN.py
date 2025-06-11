import os, random, numpy as np, copy

from torch.utils.data import Dataset
import torch
import math



class TrajectoryDatasetNonFlexN(Dataset):
    def __init__(self,past_traj, future_traj, group_net, edge_weights_past_list, edge_features_past_list,
                                         direction_past_list, velocity_past_list, visability_mat_past_list,indexes_list,predictions_real = None, indexes_list_selected=None):

        self.past_traj = past_traj
        self.future_traj = future_traj
        self.edge_weights_past_list = edge_weights_past_list
        self.edge_features_past_list = edge_features_past_list
        self.direction_past_list = direction_past_list
        self.visability_mat_past_list = visability_mat_past_list
        self.indexes_list = indexes_list
        self.group_net = group_net
        self.velocity_past_list = velocity_past_list

        self.predictions_real = predictions_real
        self.indexes_list_selected = indexes_list_selected

    def __len__(self):
        return self.past_traj.shape[0]

    def __getitem__(self, idx):
        if self.predictions_real != None:
            return (self.past_traj[idx], self.future_traj[idx], self.group_net[idx], self.edge_weights_past_list[idx],
                    self.edge_features_past_list[idx], self.direction_past_list[idx], self.velocity_past_list[idx],
                    self.visability_mat_past_list[idx], self.indexes_list[idx], self.predictions_real[idx], self.indexes_list_selected[idx])
        else:
            return (self.past_traj[idx],self.future_traj[idx],self.group_net[idx],self.edge_weights_past_list[idx],
             self.edge_features_past_list[idx],self.direction_past_list[idx], self.velocity_past_list[idx],
              self.visability_mat_past_list[idx],self.indexes_list[idx])




class TrajectoryDatasetFlexN(Dataset):
    def __init__(self,past_traj, future_traj, group_net, edge_weights_past_list, edge_features_past_list,
                 direction_past_list, velocity_past_list, visability_mat_past_list,indexes_list,grouped_seq_indices,seq_start_end, predictions_real = None, indexes_list_selected=None):
        self.past_traj = past_traj
        self.future_traj = future_traj
        self.edge_weights_past_list = edge_weights_past_list
        self.edge_features_past_list = edge_features_past_list
        self.direction_past_list = direction_past_list
        self.visability_mat_past_list = visability_mat_past_list
        self.indexes_list = indexes_list
        self.grouped_seq_indices = grouped_seq_indices
        self.seq_start_end = seq_start_end
        self.group_net = group_net
        self.velocity_past_list = velocity_past_list
        self.predictions_real = predictions_real
        self.indexes_list_selected = indexes_list_selected

        self.num_seq = len(self.seq_start_end)

    def __len__(self):
        return self.num_seq


    def __getitem__(self, idx):
        start, end = self.seq_start_end[idx]
        vis = self.visability_mat_past_list[idx]
        if not torch.is_tensor(vis):
            vis = torch.tensor(vis, dtype=torch.float32).to(self.group_net.device)

        if self.predictions_real != None:
            return (self.past_traj[start:end], self.future_traj[start:end], self.group_net[start:end], self.edge_weights_past_list[start:end],
                 self.edge_features_past_list[start:end],self.direction_past_list[start:end], self.velocity_past_list[start:end],
                  vis,self.indexes_list[start:end], self.predictions_real[start:end], self.indexes_list_selected[start:end])
        else:
            return (self.past_traj[start:end], self.future_traj[start:end], self.group_net[start:end], self.edge_weights_past_list[start:end],
                 self.edge_features_past_list[start:end],self.direction_past_list[start:end], self.velocity_past_list[start:end],
                  vis,self.indexes_list[start:end])

def seq_collate_sampler_GAN(data):

    (past_traj, future_traj, group_net, edge_weights_past, edge_features_past, direction_past,
     velocity_past, visability_mat_past, indexes_list, prediction_real, indexes_list_selected) = zip(*data)
    past_traj = torch.stack(past_traj,dim=0)
    future_traj = torch.stack(future_traj,dim=0)
    group_net = torch.stack(group_net,dim=0)
    edge_weights_past = torch.stack(edge_weights_past,dim=0)
    edge_features_past = torch.stack(edge_features_past,dim=0)
    direction_past = torch.stack(direction_past,dim=0)
    velocity_past = torch.stack(velocity_past,dim=0)
    visability_mat_past =  torch.stack(visability_mat_past,dim=0)
    indexes_list = torch.stack(indexes_list,dim=0)
    prediction_real = torch.stack(prediction_real,dim=0)
    indexes_list_selected = torch.stack(indexes_list_selected,dim=0)
    data = {
        'past_traj': past_traj,
        'old_future': future_traj,
        'group_net': group_net,
        'edge_weights_past': edge_weights_past,
        'edge_features_past': edge_features_past,
        'direction_past': direction_past,
        'velocity_past': velocity_past,
        'visability_mat_past': visability_mat_past,
        'indexes_list': indexes_list,
        'future_traj': prediction_real,
        'indexes_list_selected':indexes_list_selected
    }

    return data

def seq_collate_sampler(data):

    (past_traj, future_traj, group_net, edge_weights_past, edge_features_past, direction_past,
     velocity_past, visability_mat_past, indexes_list) = zip(*data)
    past_traj = torch.stack(past_traj,dim=0)
    future_traj = torch.stack(future_traj,dim=0)
    group_net = torch.stack(group_net,dim=0)
    edge_weights_past = torch.stack(edge_weights_past,dim=0)
    edge_features_past = torch.stack(edge_features_past,dim=0)
    direction_past = torch.stack(direction_past,dim=0)
    velocity_past = torch.stack(velocity_past,dim=0)
    visability_mat_past =  torch.stack(visability_mat_past,dim=0)
    indexes_list = torch.stack(indexes_list,dim=0)
    data = {
        'past_traj': past_traj,
        'future_traj': future_traj,
        'group_net': group_net,
        'edge_weights_past': edge_weights_past,
        'edge_features_past': edge_features_past,
        'direction_past': direction_past,
        'velocity_past': velocity_past,
        'visability_mat_past': visability_mat_past,
        'indexes_list': indexes_list
    }

    return data

class TrajectoryDatasetDisc(Dataset):
    def __init__(self, past_traj, future_traj, selected_traj, H_list, fake):
        """
        past_traj: Tensor (Total_samples, N, 5, 2)
        future_traj: Tensor (Total_samples, N, 10, 2, 20) - future trajectories per agent
        selected_traj: Tensor (Total_samples, N, 10, 2) - selected trajectory per agent
        H_list: Tensor (Total_samples, edges, N) - H matrix from model
        """
        self.past_traj = past_traj
        self.future_traj = future_traj
        self.selected_traj = selected_traj
        self.H_list = H_list
        self.fake = fake

    def __len__(self):
        return self.past_traj.shape[0]

    def __getitem__(self, idx):
        return {
            'past_traj': self.past_traj[idx],
            'group_net': self.future_traj[idx],
            'selected_traj': self.selected_traj[idx],
            'H_list': self.H_list[idx],
            'fake': self.fake[idx]
        }



class TrajectoryDatasetClassifierSamplerNoneFlexN(Dataset):
    def __init__(self, past_traj, future_traj, group_net,
                 edge_weights_past_list, edge_features_past_list,direction_past_list, velocity_past_list, visability_mat_past_list,indexes_list,
                 future_mean_list, future_first_list, predictions_real, predictions_controlled, fake_list,all_agents_idx):
        self.future_traj = future_traj
        self.future_mean_list = future_mean_list
        self.future_first_list = future_first_list
        self.fake_list = fake_list
        self.predictions_real = predictions_real
        self.predictions_controlled = predictions_controlled
        self.past_traj = past_traj
        self.group_net = group_net
        self.controlled_idx = all_agents_idx
        self.edge_weights_past_list = edge_weights_past_list
        self.edge_features_past_list = edge_features_past_list
        self.direction_past_list = direction_past_list
        self.visability_mat_past_list = visability_mat_past_list
        self.velocity_past_list = velocity_past_list
        self.indexes_list = indexes_list

    def __len__(self):
        return self.past_traj.shape[0]

    def __getitem__(self, idx):
        return (self.past_traj[idx], self.future_traj[idx], self.group_net[idx],
                self.edge_weights_past_list[idx], self.edge_features_past_list[idx],
                self.direction_past_list[idx], self.velocity_past_list[idx], self.visability_mat_past_list[idx],
                self.indexes_list[idx],
                self.future_mean_list[idx],
                self.future_first_list[idx], self.fake_list[idx], self.predictions_real[idx],
                self.predictions_controlled[idx] ,self.controlled_idx[idx])


class TrajectoryDatasetClassifierSamplerFlex(Dataset):
    def __init__(self, past_traj, future_traj, group_net,
                 edge_weights_past_list, edge_features_past_list,direction_past_list, velocity_past_list, visability_mat_past_list,indexes_list,
                 future_mean_list,future_first_list, predictions_real, predictions_controlled, fake_list,all_agents_idx, grouped_seq_indices, seq_start_end):
        self.future_traj = future_traj
        self.future_mean_list = future_mean_list
        self.future_first_list = future_first_list
        self.fake_list = fake_list
        self.predictions_real = predictions_real
        self.predictions_controlled = predictions_controlled
        self.past_traj = past_traj
        self.group_net = group_net
        self.controlled_idx = all_agents_idx
        self.edge_weights_past_list = edge_weights_past_list
        self.edge_features_past_list = edge_features_past_list
        self.direction_past_list = direction_past_list
        self.visability_mat_past_list = visability_mat_past_list
        self.velocity_past_list = velocity_past_list
        self.indexes_list = indexes_list
        self.seq_start_end = seq_start_end
        self.grouped_seq_indices = grouped_seq_indices

        self.num_seq = len(self.seq_start_end)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, idx):
        start, end = self.seq_start_end[idx]
        vis = self.visability_mat_past_list[idx]
        if not torch.is_tensor(vis):
            vis = torch.tensor(vis, dtype=torch.float32).to(self.group_net.device)
            print("vis", vis.shape)

        return (self.past_traj[start:end], self.future_traj[start:end], self.group_net[start:end],
                self.edge_weights_past_list[start:end], self.edge_features_past_list[start:end],
                self.direction_past_list[start:end], self.velocity_past_list[start:end],
                vis, self.indexes_list[start:end],
                self.future_mean_list[start:end],
                self.future_first_list[start:end], self.fake_list[start:end],
                self.predictions_real[start:end], self.predictions_controlled[start:end], self.controlled_idx[start:end])

def seq_collate_classifier_sampler(data):

    (past_traj, future_traj, group_net,
     edge_weights_past_list, edge_features_past_list, direction_past_list, velocity_past_list, visability_mat_past_list,
     indexes_list,
     future_mean_list, future_first_list, fake_list,
     predictions_real, predictions_controlled,controlled_idx) = zip(*data)

    past_traj = torch.stack(past_traj,dim=0)
    future_traj = torch.stack(future_traj,dim=0)
    group_net = torch.stack(group_net,dim=0)
    future_mean_list = torch.stack(future_mean_list,dim=0)
    future_first_list = torch.stack(future_first_list,dim=0)
    fake_list = torch.stack(fake_list,dim=0)
    predictions_real = torch.stack(predictions_real,dim=0)
    predictions_controlled = torch.stack(predictions_controlled,dim=0)
    controlled_idx = torch.stack(controlled_idx,dim=0)
    edge_weights_past_list =torch.stack(edge_weights_past_list,dim=0)
    edge_features_past_list = torch.stack(edge_features_past_list,dim=0)
    direction_past_list = torch.stack(direction_past_list,dim=0)
    velocity_past_list= torch.stack(velocity_past_list,dim=0)
    visability_mat_past_list = torch.stack(visability_mat_past_list,dim=0)
    indexes_list = torch.stack(indexes_list,dim=0)

    data = {
        'past_traj': past_traj,
        'future_traj': future_traj,
        'group_net': group_net,
        'future_mean_list': future_mean_list,
        'future_first_list': future_first_list,
        'fake_list': fake_list,
        'predictions_real': predictions_real,
        'predictions_controlled': predictions_controlled,
        'controlled_idx': controlled_idx,
        'edge_weights_past_list' :edge_weights_past_list,
        'edge_features_past_list' :edge_features_past_list,
        'direction_past_list' :direction_past_list,
        'velocity_past_list' :velocity_past_list,
        'visability_mat_past_list' :visability_mat_past_list,
        'indexes_list' :indexes_list,
    }

    return data

class TrajectoryDatasetClassifierGroupnet(Dataset):
    def __init__(self, group_net, past_traj, future_traj, future_mean_list,
                 future_first_list, predictions_real, predictions_controlled, fake_list,all_agents_idx):
        self.future_traj = future_traj
        self.future_mean_list = future_mean_list
        self.future_first_list = future_first_list
        self.fake_list = fake_list
        self.predictions_real = predictions_real
        self.predictions_controlled = predictions_controlled
        self.past_traj = past_traj
        self.group_net = group_net
        self.controlled_idx = all_agents_idx

    def __len__(self):
        return self.past_traj.shape[0]

    def __getitem__(self, idx):
        return (self.past_traj[idx], self.future_traj[idx], self.group_net[idx], self.future_mean_list[idx],
                self.future_first_list[idx], self.fake_list[idx], self.predictions_real[idx],
                self.predictions_controlled[idx] ,self.controlled_idx[idx])

class TrajectoryDatasetClassifierGroupnetFlex(Dataset):
    def __init__(self, group_net, past_traj, future_traj, future_mean_list,
                 future_first_list, predictions_real, predictions_controlled,fake_list,grouped_seq_indices,seq_start_end,controlled_idx):
        self.future_traj = future_traj
        self.future_mean_list = future_mean_list
        self.future_first_list = future_first_list
        self.fake_list = fake_list
        self.past_traj = past_traj
        self.group_net = group_net
        self.seq_start_end = seq_start_end
        self.grouped_seq_indices = grouped_seq_indices
        self.controlled_idx = controlled_idx
        self.predictions_real = predictions_real
        self.predictions_controlled = predictions_controlled

        self.num_seq = len(self.seq_start_end)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, idx):
        start, end = self.seq_start_end[idx]
        return (self.past_traj[start:end], self.future_traj[start:end], self.group_net[start:end],
                self.future_mean_list[start:end],
                self.future_first_list[start:end], self.fake_list[start:end],
                self.predictions_real[start:end], self.predictions_controlled[start:end], self.controlled_idx[start:end])

def seq_collate_classifier(data):

    (past_traj, future_traj, group_net, future_mean_list, future_first_list, fake_list,
     predictions_real, predictions_controlled,controlled_idx) = zip(*data)
    past_traj = torch.stack(past_traj,dim=0)
    future_traj = torch.stack(future_traj,dim=0)
    group_net = torch.stack(group_net,dim=0)
    future_mean_list = torch.stack(future_mean_list,dim=0)
    future_first_list = torch.stack(future_first_list,dim=0)
    fake_list = torch.stack(fake_list,dim=0)
    predictions_real = torch.stack(predictions_real,dim=0)
    predictions_controlled = torch.stack(predictions_controlled,dim=0)
    controlled_idx = torch.stack(controlled_idx,dim=0)

    data = {
        'past_traj': past_traj,
        'future_traj': future_traj,
        'group_net': group_net,
        'future_mean_list': future_mean_list,
        'future_first_list': future_first_list,
        'fake_list': fake_list,
        'predictions_real': predictions_real,
        'predictions_controlled': predictions_controlled,
        'controlled_idx': controlled_idx
    }

    return data

class TrajectoryDatasetMission(Dataset):
    def __init__(self, past_traj,future_traj, old_future_traj,):
        """
        past_traj: Tensor (Total_samples, N, 5, 2)
        """
        self.past_traj = past_traj
        self.old_future_traj = old_future_traj
        self.future_traj = future_traj



    def __len__(self):
        return self.past_traj.shape[0]

    def __getitem__(self, idx):
        return  self.past_traj[idx],self.future_traj[idx],self.old_future_traj[idx],


class TrajectoryDatasetGANTest(Dataset):
    def __init__(self, past_traj, seq_start_end, grouped_seq_indices,):
        self.past_traj = past_traj
        self.seq_start_end = seq_start_end
        self.grouped_seq_indices = grouped_seq_indices
        self.num_seq = len(self.seq_start_end)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, idx):
        start, end = self.seq_start_end[idx]
        return self.past_traj[start:end]

def seq_collate_GANTest(data):
    past_traj = torch.stack(data,dim=0)
    data = {
        'past_traj': past_traj,
        'seq': 'sdd',
    }

    return data



class TrajectoryDatasetMissionUnorder(Dataset):
    def __init__(self, past_traj, future_traj, old_future_traj, seq_start_end, grouped_seq_indices, mission_data=None):
        self.past_traj = past_traj
        self.future_traj = future_traj
        self.old_future_traj = old_future_traj
        self.seq_start_end = seq_start_end
        self.grouped_seq_indices = grouped_seq_indices
        self.num_seq = len(self.seq_start_end)
        self.mission_data = mission_data

    def __len__(self):
        return self.num_seq

    def __getitem__(self, idx):
        if self.mission_data:
            start, end = self.seq_start_end[idx]
            return self.past_traj[start:end], self.future_traj[start:end], self.old_future_traj[start:end], self.mission_data["targets"][start:end], self.mission_data["controlled_agents"][start:end]
        else:
            start, end = self.seq_start_end[idx]
            return self.past_traj[start:end], self.future_traj[start:end], self.old_future_traj[start:end]

def seq_collate_mission(data):

    (past_traj, future_traj, old_future) = zip(*data)
    past_traj = torch.stack(past_traj,dim=0)
    future_traj = torch.stack(future_traj,dim=0)
    old_future = torch.stack(old_future, dim=0)
    data = {
        'past_traj': past_traj,
        'future_traj': future_traj,
        'old_future_traj': old_future,
        'seq': 'sdd',
    }

    return data


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

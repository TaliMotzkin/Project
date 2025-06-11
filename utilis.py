
import re
from matplotlib.ticker import FormatStrFormatter

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import random
from torch.utils.data import Dataset
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from collections import defaultdict
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr
from scipy.stats import ks_2samp
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

import math


def calc_TCC(prediction, future_traj, size):
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)

    if isinstance(future_traj, np.ndarray):
        future_traj = torch.from_numpy(future_traj)

    if size == 20:
        temp = (prediction - future_traj).norm(p=2, dim=-1) #20, BN, T -> L2 distance
        pred_best = prediction[temp[:, :, -1].argmin(dim=0), range(prediction.size(1)), :, :] #(num_ped, seq_len, 2)
    else:
        pred_best = prediction
    pred_gt_stack = torch.stack([pred_best, future_traj], dim=0) #(2, num_ped, seq_len, 2)
    pred_gt_stack = pred_gt_stack.permute(3, 1, 0, 2) # (2D, num_ped, 2, seq_len)
    covariance = pred_gt_stack - pred_gt_stack.mean(dim=-1, keepdim=True)
    factor = 1 / (covariance.shape[-1] - 1)
    covariance = factor * covariance @ covariance.transpose(-1, -2)
    variance = covariance.diagonal(offset=0, dim1=-2, dim2=-1) ## (2, num_ped, 2) for stds
    stddev = variance.sqrt()
    corrcoef = covariance / stddev.unsqueeze(-1) / stddev.unsqueeze(-2) #full Pearson correlation formula (2, num_ped, 2, 2)
    corrcoef = corrcoef.clamp(-1, 1)
    corrcoef[torch.isnan(corrcoef)] = 0
    TCCs = corrcoef[:, :, 0, 1].mean(dim=0)#(num_ped,)
    return TCCs

def get_agent_colors(args, agent_num):
    if args.dataset == 'nba':
        color_map = {}
        group1 = range(0, 5)
        group2 = range(5, 10)
        group3 = [10]
        cmap = plt.get_cmap('tab10')
        for i in group1:
            color_map[i] = cmap(0)  # blue
        for i in group2:
            color_map[i] = cmap(1)  # orange
        for i in group3:
            color_map[i] = cmap(2)  # green

    elif args.dataset == 'target':
        cmap = plt.get_cmap('tab10')
        color_map = {i: cmap(i % 10) for i in range(agent_num)}

    else:
        cmap = plt.get_cmap('tab10')
        color_map = {i: cmap(i % 10) for i in range(agent_num)}
    return color_map

def vis_predictions_no_missions(constant, future_traj, args, stats_path):

    if hasattr(future_traj, 'cpu'):
        future_traj = future_traj.cpu().numpy()

    N, T, _ = future_traj.shape

    fig, ax = plt.subplots(figsize=(8, 6))

    background_path = constant.court
    x_min, x_max = constant.X_MIN, constant.X_MAX
    y_min, y_max = constant.Y_MIN, constant.Y_MAX

    if args.dataset in ['nba', 'target', 'syn']:
        x_max = max(future_traj[:, :, 0].max() + 1, x_max)
        y_max = max(future_traj[:, :, 1].max() + 1,y_max)
        x_min = min(future_traj[:, :, 0].min() - 1,x_min)
        y_min = min(future_traj[:, :, 1].min() - 1,y_min)

    extent = [x_min, x_max, y_min, y_max]
    if os.path.exists(background_path):
        if args.dataset == 'eth' and args.scene in ['uni', 'zara_01', 'zara_02']:
            background_img = plt.imread(background_path)
            ax.imshow(background_img, extent=extent, zorder=0, alpha=0.5)
        elif args.dataset == 'sdd' and args.sdd_scene == 2:
            background_img = plt.imread(background_path)
            ax.imshow(background_img, extent=extent, zorder=0, alpha=0.5)
        elif args.dataset == 'nba':
            background_img = plt.imread(background_path)
            ax.imshow(background_img, extent=extent, zorder=0, alpha=0.5)
    else:
        print(f"Warning: Background image not found at {background_path}")




    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Agent Trajectories Simulation")

    color_map = get_agent_colors(args, N)
    facecolors = []

    for i in range(N):
        color = color_map[i]
        facecolors.append(color)

    scatter = ax.scatter([], [], s=80, zorder=3)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_facecolor(facecolors)
        return scatter,

    def update(frame):
        # updating agent positions
        positions = future_traj[:, frame, :]
        scatter.set_offsets(positions)
        scatter.set_facecolor(facecolors)

        ax.set_title(f"Simulation at time: {frame * 0.4:.1f}s / {T * 0.4:.1f}s")
        return scatter,

    ani = animation.FuncAnimation(
        fig, update, frames=range(T), init_func=init,
        blit=True, interval=10
    )
    writer = PillowWriter(fps=10 )

    if stats_path:
        if os.path.isdir(f"{stats_path}/no_mission/"):
            print("")
        else:
            os.makedirs(f"{stats_path}/no_mission/", exist_ok=True)
        ani.save(f"{stats_path}/no_mission/simulation_no_mission.gif", writer=writer)
    else:
        ani.save(f"plots/simulation_no_mission_{args.timestamp}_{args.dataset}.gif", writer=writer)
    plt.close(fig)

def vis_predictions_missions(constant, future_traj, mission_log, target_status, args, targets, agents_idx_plot, stats_path=None):

    # print("future_traj", len(future_traj), future_traj[0])
    N, T, _ = future_traj.shape
    completed_targets = set()
    targets = targets.cpu().numpy()


    future_traj_transformed = np.zeros_like(future_traj)
    H_inv = constant.H_inv

    mission_log_by_frame = defaultdict(list)
    #figure and axis limits
    fig, ax = plt.subplots(figsize=(8, 6))

    for t_idx, agent, midx, target in mission_log:
        mission_log_by_frame[t_idx].append((agent, midx, target))


    background_path = constant.court

    x_min, x_max = constant.X_MIN, constant.X_MAX
    y_min, y_max = constant.Y_MIN, constant.Y_MAX

    if args.dataset in ['nba', 'target', 'syn']:
        x_max = max(future_traj[:, :, 0].max() + 1, x_max)
        y_max = max(future_traj[:, :, 1].max() + 1,y_max)
        x_min = min(future_traj[:, :, 0].min() - 1,x_min)
        y_min = min(future_traj[:, :, 1].min() - 1,y_min)



    extent = [x_min, x_max, y_min, y_max]
    if os.path.exists(background_path):
        if args.dataset == 'eth' and args.scene in ['uni', 'zara_01', 'zara_02']:
            background_img = plt.imread(background_path)
            ax.imshow(background_img, extent=extent, zorder=0, alpha=0.5)
        elif args.dataset == 'sdd' and args.sdd_scene == 2:
            background_img = plt.imread(background_path)
            ax.imshow(background_img, extent=extent, zorder=0, alpha=0.5)
        elif args.dataset == 'nba':
            background_img = plt.imread(background_path)
            ax.imshow(background_img, extent=extent, zorder=0, alpha=0.5)


    else:
        print(f"Warning: Background image not found at {background_path}")

    future_traj_transformed = future_traj

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Agent Trajectories Simulation")

    agent_current_targets = {}  # {agent_id: (x, y)}
    target_scatters = {}  # agent_id >> plot handle
    # print("agents_idx_plot", agents_idx_plot)
    for i, agent in enumerate(agents_idx_plot):
        # print("targets first agents, " ,targets[:,0])
        target = targets[agent][0]  # first mission
        agent_current_targets[agent] = target
        handle = ax.scatter([target[0]], [target[1]], color='black', marker='x', s=100, zorder=2)
        target_scatters[agent] = handle

    color_map = get_agent_colors(args, N)
    controlled_agents = agents_idx_plot if isinstance(agents_idx_plot, (list, tuple)) else [agents_idx_plot]

    facecolors = []
    edgecolors = []
    linewidths = []

    for i in range(N):
        color = color_map[i]
        facecolors.append(color)

        if i in controlled_agents:
            edgecolors.append('black')  # black border
            linewidths.append(2.5)
        else:
            edgecolors.append(color)
            linewidths.append(0.5)


    scatter = ax.scatter([], [], s=80, zorder=3)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_facecolor(facecolors)
        scatter.set_edgecolor(edgecolors)
        scatter.set_linewidths(linewidths)
        return scatter,

    def update(frame):
        current_positions = future_traj_transformed[:, frame, :]
        scatter.set_offsets(current_positions)
        scatter.set_facecolor(facecolors)
        scatter.set_edgecolor(edgecolors)
        scatter.set_linewidths(linewidths)

        ax.set_title(f"Simulation at time: {frame * 0.4:.1f}s / {T * 0.4:.1f}s", zorder=3)

        if frame in mission_log_by_frame:
            for agent, midx, target in mission_log_by_frame[frame]:
                if (agent, midx) not in completed_targets:
                    completed_targets.add((agent, midx))

                    if agent in target_scatters:
                        target_scatters[agent].remove()

                    # Add next mission if it exists
                    next_midx = midx + 1
                    if next_midx < targets.shape[1]:  # has next mission
                        new_target = targets[agent][next_midx]
                        handle = ax.scatter([new_target[0]], [new_target[1]], color='black', marker='x', s=100, zorder=2)
                        agent_current_targets[agent] = new_target
                        target_scatters[agent] = handle
                    else:
                        agent_current_targets.pop(agent, None)
                        target_scatters.pop(agent, None)

                    print(f"Agent {agent} completed mission {midx} at time {frame * 0.4:.1f}s")

        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=range(T), init_func=init, blit=True, interval=10)
    writer = PillowWriter(fps=10)

    if stats_path:
        ani.save(f"{stats_path}/simulation.gif", writer=writer)
    else:
        ani.save(f"plots/simulation_{args.timestamp}_{args.dataset}.gif", writer=writer)
    plt.close(fig)



def prepare_targets_mission_net(constant, data, dataset, device, epoch, five_missions = False):
    B, N, _, _ = data['past_traj'].shape  # shape = (B, N, T, 2)


    x_min, x_max = constant.X_MIN+1, constant.X_MAX-1
    y_min, y_max = constant.Y_MIN+1, constant.Y_MAX-1
    dx = x_max - x_min
    dy = y_max - y_min

    max_radius = constant.buffer_train * 2
    min_radius = constant.buffer_train/2
    error_tolerance = max_radius - (max_radius - min_radius) * (epoch / 25)

    if five_missions:
        # setting chosen agents.
        num_controlled = random.randint(1, N)
        perm = torch.randperm(N, device=device)
        agents_idx = perm[:num_controlled]

        # M = 400  # number of missions
        # targets = torch.rand(B, N, M, 2, device=device)
        # targets[..., 0] = targets[..., 0] * dx + x_min
        # targets[..., 1] = targets[..., 1] * dy + y_min
        # agents_tragets = targets[:, agents_idx, :, :]  # B, C, M, 2

        targets = torch.rand(B, N, 2, device=device)
        targets[..., 0] = targets[..., 0] * dx + x_min
        targets[..., 1] = targets[..., 1] * dy + y_min
        agents_tragets = targets[:, agents_idx,  :]  # B, C, M, 2

        return agents_tragets, agents_idx, error_tolerance, None


    else:

        # setting chosen agents.
        num_controlled = random.randint(0, N)
        perm = torch.randperm(N, device=device)
        agents_idx = perm[:num_controlled]


        targets = torch.rand(B, N, 2, device=device)
        targets[..., 0] = targets[..., 0] * dx + x_min
        targets[..., 1] = targets[..., 1] * dy + y_min


        # 50% of time nearGT
        use_gt_target = torch.rand(N, device=device) < 0.5

        rand_t_indices = torch.randint(low=0, high=data['future_traj'].shape[2], size=(N,), device=device)
        rand_t_indices = rand_t_indices.unsqueeze(0).expand(B, -1)  # (B, N)
        gt_targets = torch.gather(
            data['future_traj'].to(device), dim=2,
            index=rand_t_indices.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, 2)
        ).squeeze(2)  # B, N, 2
        noise = torch.randn_like(gt_targets, device=device) * 2.0
        noisy_gt_targets = gt_targets + noise
        use_gt_mask = use_gt_target.view(1, N, 1).expand(B, N, 1).float()  # (B, N, 1)
        targets = use_gt_mask * noisy_gt_targets + (1 - use_gt_mask) * targets
        agents_tragets = targets[:, agents_idx, :]  # B, C, 2


        choices = torch.linspace(0.01, 0.3, steps=10, device=device)
        choices_far = torch.linspace(0.3, 1.5, steps=10, device=device)
        choices = torch.cat([choices, choices_far])
        i = torch.randint(len(choices), (), dtype=torch.long)
        val = choices[i]
        rand = val.repeat(B, N)

        not_idxs = list(set(range(N)) - set(agents_idx.tolist()))
        rand[:, not_idxs] = 0

    return agents_tragets, agents_idx, error_tolerance, rand


def baseline_mission_reach(traj_sample, one_mission, agents_idx, error_tolerance, args):

    B, N, T, _ = traj_sample.shape
    C = agents_idx.shape[0]

    last_5 = traj_sample[:, :, -5:, :]  # (B, N, 5, 2)
    avg_vel = (last_5[:, :, 1:, :] - last_5[:, :, :-1, :]).mean(dim=2, keepdim=True)
    last_pos = traj_sample[:, :, -1:, :]
    extrapolated = [last_pos + i * avg_vel for i in range(1, args.future_length + 1)]
    extrapolated = torch.cat(extrapolated, dim=2)  # (B, N, T_fut, 2)

    start_pos = traj_sample[:, agents_idx, -1, :]       # (B, C, 2)
    mission_vec = one_mission - start_pos             # (B, C, 2)
    mission_dist = mission_vec.norm(dim=-1)           # (B, C)

    #max allowed distance: 2 * total past  length
    deltas = traj_sample[:, agents_idx, 1:, :] - traj_sample[:, agents_idx, :-1, :]  # (B, C, T-1, 2)
    past_len = deltas.norm(dim=-1).sum(dim=-1).clamp(min=1e-6)                   # (B, C)
    max_dist = 2.5 * past_len                                                    # (B, C)

    scale = (max_dist / mission_dist).clamp(max=1.0).unsqueeze(-1)              # (B, C, 1)
    clipped_vec = mission_vec * scale                                            # (B, C, 2)
    final_target = start_pos + clipped_vec                                       # (B, C, 2)

    steps = torch.linspace(0, 1, args.future_length, device=traj_sample.device)  # (T,)
    steps = steps.view(1, 1, -1, 1)                                              # (1, 1, T, 1)
    controlled_traj = start_pos.unsqueeze(2) + steps * (final_target - start_pos).unsqueeze(2)  # (B, C, T, 2)

    batch_idx = torch.arange(B, device=traj_sample.device).unsqueeze(1)   # (B, 1)
    agent_idx = agents_idx.view(1, -1).expand(B, -1)                    # (B, C)
    extrapolated[batch_idx, agent_idx] = controlled_traj

    return extrapolated.view(1, B * N, args.future_length, 2).to(args.device)




def baseline_mission_wave(traj_sample, one_mission, agents_idx, error_tolerance, args):
    B, N, T, _ = traj_sample.shape
    C = agents_idx.shape[0]

    deltas = traj_sample[:, agents_idx, 1:, :] - traj_sample[:, agents_idx, :-1, :]
    past_len = deltas.norm(dim=-1).sum(dim=-1).clamp(min=1e-6)  # (B, C)
    max_dist = 2 * past_len

    start_pos = traj_sample[:, agents_idx, -1, :]         # (B, C, 2)
    mission_vec = one_mission - start_pos                 # (B, C, 2)
    mission_dist = mission_vec.norm(dim=-1)               # (B, C)

    scale = (max_dist / mission_dist).clamp(max=1.0).unsqueeze(-1)  # (B, C, 1)
    clipped_vec = mission_vec * scale                                # (B, C, 2)
    final_target = start_pos + clipped_vec                           # (B, C, 2)

    t_grid = torch.linspace(0, 1, args.future_length, device=traj_sample.device).view(1, 1, -1, 1)  # (1, 1, T, 1)
    base_line = start_pos.unsqueeze(2) + t_grid * (final_target - start_pos).unsqueeze(2)  # (B, C, T, 2)

    direction = clipped_vec / (clipped_vec.norm(dim=-1, keepdim=True) + 1e-6)  # (B, C, 2)
    normal = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)  # (B, C, 2)

    amplitude = 0.2 * clipped_vec.norm(dim=-1, keepdim=True).unsqueeze(-1)  # (B, C, 1, 1)
    sine_wave = torch.sin(t_grid * 2 * math.pi) * amplitude  # (1, 1, T, 1)
    offset = sine_wave * normal.unsqueeze(2)  # (B, C, T, 2)

    sinus_traj = base_line + offset  # (B, C, T, 2)

    extrapolated = torch.zeros(B, N, args.future_length, 2, device=args.device)
    batch_idx = torch.arange(B, device=args.device).unsqueeze(1)
    agent_idx = agents_idx.view(1, -1).expand(B, -1)
    extrapolated[batch_idx, agent_idx] = sinus_traj

    return extrapolated.view(1, B * N, args.future_length, 2)




class MissionTestDatasetGAN(Dataset):
    def __init__(self, traj_dataset, mission_data):

        self.traj_dataset = traj_dataset
        self.targets = mission_data["targets"]  # shape: (S, N, M, 2)
        self.controlled_agents = mission_data["controlled_agents"]  # shape: (S, N_max)

    def __len__(self):
        if isinstance(self.targets, list):
            return len(self.targets)
        else: return self.targets.shape[0]

    def __getitem__(self, sim_idx):
        traj_sample = self.traj_dataset[sim_idx]  # shape: (N, T, 2) -> past traj
        missions = self.targets[sim_idx]             # shape: (N, M, 2)
        controlled = self.controlled_agents[sim_idx] # shape: (N_max,) -> need to mask from -1
        return traj_sample, missions, controlled

class MissionTestDataset(Dataset):
    def __init__(self, traj_dataset, mission_data):

        self.traj_dataset = traj_dataset
        self.targets = mission_data["targets"]  # shape: (S, N, M, 2)
        self.controlled_agents = mission_data["controlled_agents"]  # shape: (S, N_max)

    def __len__(self):
        if isinstance(self.targets, list):
            return len(self.targets)
        else: return self.targets.shape[0]

    def __getitem__(self, sim_idx):
        traj_sample = self.traj_dataset[sim_idx][0]  # shape: (N, T, 2) -> past traj
        future = self.traj_dataset[sim_idx][1]  # shape: (N, T, 2) -> future traj
        missions = self.targets[sim_idx]             # shape: (N, M, 2)
        controlled = self.controlled_agents[sim_idx] # shape: (N_max,) -> need to mask from -1
        return traj_sample, missions, controlled, future



class MissionTestDatasetSampler(Dataset):
    def __init__(self, traj_dataset, mission_data):

        self.traj_dataset = traj_dataset
        self.targets = mission_data["targets"]  # shape: (S, N, M, 2)
        self.controlled_agents = mission_data["controlled_agents"]  # shape: (S, N_max)

    def __len__(self):
        if isinstance(self.targets, list):
            return len(self.targets)
        else: return self.targets.shape[0]

    def __getitem__(self, sim_idx):
        traj_sample = self.traj_dataset[sim_idx][0]  # shape: (N, T, 2) -> past traj
        future = self.traj_dataset[sim_idx][1]  # shape: (N, T, 2) -> future traj
        group_net = self.traj_dataset[sim_idx][2]
        edge_weights_past = self.traj_dataset[sim_idx][3]
        edge_features_past =self.traj_dataset[sim_idx][4]
        direction_past =self.traj_dataset[sim_idx][5]
        velocity_past =self.traj_dataset[sim_idx][6]
        visability_mat_past =self.traj_dataset[sim_idx][7]
        indexes_list = self.traj_dataset[sim_idx][8]
        missions = self.targets[sim_idx]             # shape: (N, M, 2)
        controlled = self.controlled_agents[sim_idx] # shape: (N_max,) -> need to mask from -1
        return traj_sample, missions, controlled, future, group_net, edge_weights_past, edge_features_past, direction_past, velocity_past, visability_mat_past, indexes_list



class MissionTestDatasetGroundTruth(Dataset):
    def __init__(self, traj_dataset, mission_data):

        self.traj_dataset = traj_dataset
        self.targets = mission_data["targets"]  # shape: (S, N, M, 2)
        self.controlled_agents = mission_data["controlled_agents"]  # shape: (S, N_max)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, sim_idx):
        traj_sample = self.traj_dataset[sim_idx][0]  # shape: (N, T, 2) -> past traj
        traj_sample_future = self.traj_dataset[sim_idx][1]  # shape: (N, T, 2) -> past traj
        missions = self.targets[sim_idx]             # shape: (N, M, 2)
        controlled = self.controlled_agents[sim_idx] # shape: (N_max,) -> need to mask from -1
        return traj_sample, traj_sample_future, missions, controlled

def prepare_target(args, device, traj_dataset, constant, type="train"):
    x_min, x_max = constant.X_MIN+1, constant.X_MAX-1
    y_min, y_max = constant.Y_MIN+1, constant.Y_MAX-1
    print("TAGETS bOUNDs", x_min, x_max, y_min, y_max)
    dx = x_max - x_min
    dy = y_max - y_min
    max_covert = constant.n_agents
    path = f'datasets/{args.dataset}/tests/{args.length}_{args.sim_num}_{args.mission_num}_{max_covert}_{args.sdd_scene}_{args.scene}_{args.training_type}_{type}.pt'

    if os.path.exists(path):
        print("Loading pre-generated targets")
        print("from ", path)
        return torch.load(path)
    else:
        print("Creating dataset")

        targets_list = []
        controlled_agents_list = []
        print("traj_dataset", len(traj_dataset))
        for sim_idx in range(len(traj_dataset)):
            # finding number of agents for this simulation
            num_agents = traj_dataset[sim_idx][0].shape[0]

            # create random missions for (num_agents, mission_num, 2)
            targets = torch.rand(num_agents, args.mission_num, 2, device=device)
            targets[..., 0] = targets[..., 0] * dx + x_min
            targets[..., 1] = targets[..., 1] * dy + y_min

            targets_list.append(targets)
            # print("targets", targets[0])
            # Controlled agents
            if args.dataset == 'nba':
                max_idx = num_agents - 1
            else:
                max_idx = num_agents
            num_controlled = random.randint(1, min(max_covert, max_idx))
            perm = torch.randperm(max_idx, device=device)
            controlled_agents = perm[:num_controlled]

            #paDDING controlled_agents to maximum possible length (-1 where unused)
            padded = torch.full((num_agents,), fill_value=-1, dtype=torch.long, device=device)
            padded[:controlled_agents.shape[0]] = controlled_agents
            controlled_agents_list.append(padded)


        data = {
            "targets": targets_list,  # list of tensors: each (N, M, 2)
            "controlled_agents": controlled_agents_list     #list of tensors: each (N,) with padding
        }

        torch.save(data, path)
        print(f"Saved mission dataset to {path}")
        return data

def create_stats_folder(args, constant, model_type, root="stats"):

    if model_type == "GT":
        if args.dataset in ['nba', 'target', 'syn']:
            run_name = f"{model_type}/{root}/{args.dataset}/{args.training_type}_{args.info}"
        else:
            run_name = f"{model_type}/{root}/{args.dataset}/{args.training_type}_{args.scene}_{args.sdd_scene}_{constant.n_agents}_{args.info}"


    elif model_type ==  "Sampler":
        numbers = re.findall(r'\d+', args.saved_models_SAM)
        seed = int(numbers[-1]) if numbers else None
        if args.dataset in ['nba', 'target', 'syn']:
            run_name = f"{model_type}/{root}/{args.dataset}/{str(args.test_mlp)}/{args.training_type}/missions{args.mission_num}_length{args.length}s_sim{args.sim_num}_buf{constant.buffer}_{seed}_{args.seed}_{args.info}"
        else:
            run_name = f"{model_type}/{root}/{args.dataset}/{str(args.test_mlp)}/{args.scene}_{args.sdd_scene}_{constant.n_agents}/missions{args.mission_num}_length{args.length}s_sim{args.sim_num}_buf{constant.buffer}_{seed}_{args.info}"


    elif model_type == "GroupNet":
        numbers = re.findall(r'\d+', args.model_names)
        seed = int(numbers[-1]) if numbers else None
        if args.dataset in ['nba', 'target', 'syn']:
            run_name = f"G1/{root}/{args.dataset}/{args.training_type}/missions{args.mission_num}_length{args.length}s_sim{args.sim_num}_buf{constant.buffer}_{seed}_{args.seed}_{args.info}"
        else:
            run_name = f"G1/{root}/{args.dataset}/{args.scene}_{args.sdd_scene}_{constant.n_agents}/missions{args.mission_num}_length{args.length}s_sim{args.sim_num}_buf{constant.buffer}_{seed}_{args.info}"


    elif model_type == "groupnet-mission":
        numbers = re.findall(r'\d+', args.model_names)
        far = int(numbers[-1]) if numbers else None
        if args.dataset in ['nba', 'target', 'syn']:
            run_name = f"GM/{root}/{args.dataset}/{args.training_type}/missions{args.mission_num}_length{args.length}s_sim{args.sim_num}_buf{constant.buffer}_how_far_{far}_{args.seed}_{args.info}"
        else:
            run_name = f"GM/{root}/{args.dataset}/{args.scene}_{args.sdd_scene}_{constant.n_agents}/missions{args.mission_num}_length{args.length}s_sim{args.sim_num}_buf{constant.buffer}_how_far_{far}_{args.seed}_{args.info}"

    elif model_type == "sampler-mission":
        numbers = re.findall(r'\d+', args.model_names)
        far = int(numbers[-1]) if numbers else None
        if args.dataset in ['nba', 'target', 'syn']:
            run_name = f"SM/{root}/{args.dataset}/{args.training_type}/missions{args.mission_num}_length{args.length}s_sim{args.sim_num}_buf{constant.buffer}_how_far_{far}_{args.seed}_{args.info}"
        else:
            run_name = f"SM/{root}/{args.dataset}/{args.scene}_{args.sdd_scene}_{constant.n_agents}/missions{args.mission_num}_length{args.length}s_sim{args.sim_num}_buf{constant.buffer}_how_far_{far}_{args.seed}_{args.info}"


    elif model_type in ["groupnet-gan", "groupnet-gan-disc"]:
        numbers = re.findall(r'\d+', args.saved_models_GM)
        far = int(numbers[-1]) if numbers else None
        if model_type == "groupnet-gan":
            disc_type = ''
        else:
            disc_type = args.disc_type
        if args.dataset in ['nba', 'target', 'syn']:
            run_name = f"GANG/{root}/{args.dataset}/{args.training_type}/missions{args.mission_num}_length{args.length}s_sim{args.sim_num}_buf{constant.buffer}_how_far_{far}_{args.seed}_{args.info}_{disc_type}"
        else:
            run_name = f"GANG/{root}/{args.dataset}/{args.scene}_{args.sdd_scene}_{constant.n_agents}/missions{args.mission_num}_length{args.length}s_sim{args.sim_num}_buf{constant.buffer}_how_far_{far}_{args.seed}_{args.info}_{disc_type}"


    elif model_type in ["sampler-gan", 'sampler-gan-disc']:

        numbers = re.findall(r'\d+', args.saved_models_SM)
        far = int(numbers[-1]) if numbers else None
        if model_type == "sampler-gan":
            disc_type = ''
        else:
            disc_type = args.disc_type
        if args.dataset in ['nba', 'target', 'syn']:
            run_name = f"GANS/{root}/{args.dataset}/{args.training_type}/missions{args.mission_num}_length{args.length}s_sim{args.sim_num}_buf{constant.buffer}_how_far_{far}_{args.seed}_{args.info}_{disc_type}"
        else:
            run_name = f"GANS/{root}/{args.dataset}/{args.scene}_{args.sdd_scene}_{constant.n_agents}/missions{args.mission_num}_length{args.length}s_sim{args.sim_num}_buf{constant.buffer}_how_far_{far}_{args.seed}_{args.info}_{disc_type}"





    os.makedirs(run_name, exist_ok=True)
    return run_name


class Subset(Dataset):
    def __init__(self,
                 base_dataset,
                 target_num_agents: int,
                 max_samples: int = None,
                 sdd_scene: str = None):

        super().__init__()
        self.base = base_dataset
        self.keep = []

        for seq_idx, (start, end) in enumerate(self.base.seq_start_end_test):
            if end - start != target_num_agents:
                continue
            if sdd_scene is not None:
                right_data = self.base.obs_traj[start:end, :, :]
                right_scene = right_data[0, 0, 3]
                if right_scene != sdd_scene:
                    continue
            self.keep.append(seq_idx)
            if max_samples and len(self.keep) >= max_samples:
                break

        if len(self.keep) == 0:
            raise ValueError(
                f"No test sequences found with {target_num_agents} agents"
                + (f" in scene {sdd_scene}" if sdd_scene else "") + "."
            )
        print(f"Found {len(self.keep)} test sequences.")

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, idx):
        orig_idx = self.keep[idx]
        return self.base.__getitem__(orig_idx)

    def get_all_trajectories_sdd(self, args):

        device = self.base.obs_traj.device
        S = len(self.keep)
        T = args.future_length + args.past_length
        N = args.testing_num_agents
        out = torch.zeros(S, T, N, 2, device=device)

        for si, seq_idx in enumerate(self.keep):
            start, end = self.base.seq_start_end_test[seq_idx]
            agents = slice(start, end)  #all agents in that sequence
            pieces = []
            obs = self.base.obs_traj[agents, :, :2] # (N, T_obs, 2)
            pieces.append(obs)
            pred = self.base.pred_traj[agents, :, :2]  # (N, T_pred, 2)
            pieces.append(pred)
            traj = torch.cat(pieces, dim=1)  # (N, T_obs+T_pred, 2)
            out[si] = traj.permute(1, 0, 2)  # (T, N, 2)

        return out

    def get_all_trajectories_eth(self, args):

        device = self.base.obs_traj_test.device
        S = len(self.keep)
        T = args.future_length + args.past_length
        N = args.testing_num_agents
        out = torch.zeros(S, T, N, 2, device=device)

        for si, seq_idx in enumerate(self.keep):
            start, end = self.base.seq_start_end_test[seq_idx]
            agents = slice(start, end)  #all agents in that sequence
            pieces = []
            obs = self.base.obs_traj_test[agents, :, :2]  # (N, T_obs, 2)
            pieces.append(obs)
            pred = self.base.pred_traj_test[agents, :, :2]  # (N, T_pred, 2)
            pieces.append(pred)
            traj = torch.cat(pieces, dim=1)  # (N, T_obs+T_pred, 2)
            out[si] = traj.permute(1, 0, 2)  # (T, N, 2)

        return out

def analyze_usage_GT(all_trajectories, all_centroids, field_length, num_blocks, timestep_duration, args, folder, constant):
    """
    all_centroids:[B*15, 2]
    all_trajectories: B, 15, N, 2
    field_length: total field length
    num_blocks: number of vertical blocks
    timestep_duration: time per step (0.4)
    """

    # Denormaliz
    if args.dataset == 'sdd':
        print("unormalizing")
        all_centroids[:, 0] = (all_centroids[:, 0] / 100.0) * (1951.0 - 9.0) + 9.0
        all_centroids[:, 1] = (all_centroids[:, 1] / 100.0) * (1973.0 - 7.0) + 7.0

        all_trajectories[:, :,:,0] = (all_trajectories[:,:,:, 0] / 100.0) * (1951.0 - 9.0) + 9.0
        all_trajectories[:, :,:,1] = (all_trajectories[:, :,:,1] / 100.0) * (1973.0 - 7.0) + 7.0

    print("all_trajectories", all_trajectories.shape)
    print("all_centroids", all_centroids.shape)
    print("all_trajectories_gt X", all_trajectories[:, :, :,0].min(), all_trajectories[:, :, :,0].max())
    print("all_trajectories_gt Y", all_trajectories[:, :, :,1].min(), all_trajectories[:,:, :, 1].max())

    B, T_f, N, _ = all_trajectories.shape

    all_centroids = all_centroids.cpu().numpy() if torch.is_tensor(all_centroids) else all_centroids
    block_edges = np.linspace(0, field_length, num_blocks + 1)
    block_centers = (block_edges[:-1] + block_edges[1:]) / 2


    simulation_legnth = all_centroids.shape[0]
    time_sim = (simulation_legnth * timestep_duration)
    x_coords = all_centroids[:, 0] #B*15

    average_per_block = np.zeros(num_blocks)
    block_entries = [[] for _ in range(num_blocks)]
    current_block = None
    entry_time = None

    #calcualtes percentage time of centroid in certain area, and average stay in area per each enterence.
    for t, x in enumerate(x_coords):
        block = np.searchsorted(block_edges, x, side='right') - 1 # finds block index for x
        block = np.clip(block, 0, num_blocks - 1) #s turn any x into an integer block index [0,4,5,0,1,4..]
        average_per_block[block] += timestep_duration/time_sim

        if block != current_block:
            if current_block is not None:
                duration = t - entry_time
                block_entries[current_block].append(duration * timestep_duration)
            current_block = block
            entry_time = t
    if current_block is not None:
        duration = len(x_coords) - entry_time
        block_entries[current_block].append(duration * timestep_duration)

    avg_stay_per_entry = [np.mean(times) if times else 0 for times in block_entries]
    std_stay_per_entry = [np.std(times) if times else 0 for times in block_entries]


    velocity_means_sim = np.zeros((N, num_blocks))
    velocity_stds_sim = np.zeros_like(velocity_means_sim)
    accel_means_sim = np.zeros_like(velocity_means_sim)
    accel_stds_sim = np.zeros_like(velocity_means_sim)
    agent_block_time = np.zeros((N, num_blocks))
    fragments = all_trajectories # B,15, N, 2
    print("fragments", fragments.shape)

    velocity_vals = [[[] for _ in range(num_blocks)] for _ in range(N)]
    accel_vals = [[[] for _ in range(num_blocks)] for _ in range(N)]

    xy_vals, vel_vals, acc_vals = [], [], []

    for i, frag in enumerate(fragments): #15, N, 2
        vel = np.diff(frag, axis=0) / timestep_duration  # [14, N, 2]
        acc = np.diff(vel, axis=0) / timestep_duration  # [13, N, 2]
        T = frag.shape[0]
        # if i == 0 :
        #     print("vel", vel)

        for agent in range(N):
            # collect for KDE
            for t in range(T-2):
                pos = frag[t, agent]               # (x,y)
                v_mag = np.linalg.norm(vel[t, agent])
                a_mag = np.linalg.norm(acc[t, agent])
                xy_vals.append(pos)
                vel_vals.append(v_mag)
                acc_vals.append(a_mag)

            agent_x = frag[:-1, agent, 0]  # [14]
            agent_blocks = np.clip(np.searchsorted(block_edges, agent_x, side='right') - 1, 0, num_blocks - 1)

            for b in range(num_blocks):

                mask_v = (agent_blocks == b)
                v_in_block = np.linalg.norm(vel[mask_v, agent], axis=-1)

                mask_a = (agent_blocks[:-1] == b)
                a_in_block = np.linalg.norm(acc[mask_a, agent], axis=-1) if acc.shape[0] > 0 else []

                if len(v_in_block) > 0:
                    velocity_vals[agent][b].extend(v_in_block)
                if len(a_in_block) > 0:
                    accel_vals[agent][b].extend(a_in_block)


                block_time = np.sum(mask_v) * timestep_duration
                agent_block_time[agent, b] += block_time

    # final aggregation
    for agent in range(N):
        for b in range(num_blocks):
            v_vals = np.array(velocity_vals[agent][b])
            a_vals = np.array(accel_vals[agent][b])

            if len(v_vals) > 0:
                velocity_means_sim[agent, b] = v_vals.mean()
                velocity_stds_sim[agent, b] = v_vals.std()
            if len(a_vals) > 0:
                accel_means_sim[agent, b] = a_vals.mean()
                accel_stds_sim[agent, b] = a_vals.std()


    # normalize occupancy
    total_sim_time = fragments.shape[0] * (args.future_length +args.past_length) * timestep_duration
    agent_block_percent = agent_block_time / total_sim_time

    np.savez_compressed(
        f"{folder}/GT_logged_data.npz",
        centroids=all_centroids,
        trajectories=all_trajectories,
        xy_vals=np.array(xy_vals),
        vel_vals=np.array(vel_vals),
        acc_vals=np.array(acc_vals)
    )
    print("np.array(xy_vals)", np.array(xy_vals).shape)

    if args.dataset == "nba":
        ball_block_percent_sims_mean = agent_block_percent[10, :] #only ball ->num_blocks
        plt.figure(figsize=(10, 5))
        plt.bar(block_centers, ball_block_percent_sims_mean)
        plt.title(f"Centroid Field Percentage of Time Usage Histogram of Ball ({args.dataset}) - {time_sim}s")
        plt.xlabel("Field X Position (meters)")
        plt.ylabel(f"Percentage of Time Spent (s) out of {time_sim}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"{folder}/centroid_avg_usage_histogram_ball_{args.dataset}_{time_sim}s_{args.timestamp}.png")
        plt.close()


        #row represents a block
        df = pd.DataFrame({
            "Block Index": range(num_blocks),
            "Block Center (m)": block_centers,
            "Avg Time in Block (%)": average_per_block,
            "Avg Stay Per Entry (s)": avg_stay_per_entry,
            "Std Stay Per Entry (s)": std_stay_per_entry
        })

        vel_mean_per_block = velocity_means_sim.mean(axis=(0))  # average over all agents per block
        vel_std_per_block = velocity_stds_sim.mean(axis=(0))
        acc_mean_per_block = accel_means_sim.mean(axis=(0))
        acc_std_per_block = accel_stds_sim.mean(axis=(0))

        loc_mean_per_block = agent_block_percent.mean(axis=0)
        loc_std_per_block = agent_block_percent.mean(axis=0)

        df["Velocity Mean (All Sims All agents)"] = vel_mean_per_block
        df["Velocity Std (All Sims All agents)"] = vel_std_per_block
        df["Acceleration Mean (All Sims All agents)"] = acc_mean_per_block
        df["Acceleration Std (All Sims All agents)"] = acc_std_per_block
        df["location Mean (All Sims All agents)"] = loc_mean_per_block
        df["location Std (All Sims All agents)"] = loc_std_per_block

        for agent in range(N):
            for b in range(num_blocks):
                df.loc[b, f"Agent{agent}_Vel_Mean"] = velocity_means_sim[agent, b]
                df.loc[b, f"Agent{agent}_Vel_Std"] = velocity_stds_sim[agent, b]
                df.loc[b, f"Agent{agent}_Acc_Mean"] = accel_means_sim[agent, b]
                df.loc[b, f"Agent{agent}_Acc_Std"] = accel_stds_sim[agent, b]
                df.loc[b, f"Agent{agent}_Occupancy_Pct"] = agent_block_percent[agent, b]

        plt.figure(figsize=(10, 5))
        plt.bar(block_centers, df["Avg Time in Block (%)"], color="red")
        plt.tick_params(axis='both', which='major', labelsize=16)
        max_y = df["Avg Time in Block (%)"].max()
        yticks = np.linspace(0, max_y, 5)
        plt.yticks(yticks, [f"{tick:.2f}" for tick in yticks])
        plt.title(r"(a) Spatial Occupancy Distribution of" "\n" r"Agents' Centroids - Ground Truth", fontsize=22, pad=15)
        plt.xlabel("Spatial Zones (m)",  fontsize=20)
        plt.ylabel(f"Block Occupancy Fraction",  fontsize=20)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"{folder}/centroid_avg_usage_histogram_{args.dataset}_{time_sim}s_{args.timestamp}.png")
        plt.close()




        plt.figure(figsize=(10, 5))
        plt.bar(block_centers, df["Avg Stay Per Entry (s)"], yerr=df["Std Stay Per Entry (s)"], capsize=5)
        plt.title(f"Centroid Field Average Time Usage Histogram ({args.dataset}) - {time_sim}s")
        plt.xlabel("Field X Position (meters)")
        plt.ylabel("Average Time Spent (s) per Block")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{folder}/centroid_avg_usage_histogram_per_Block_{args.dataset}_{time_sim}s_{args.timestamp}.png")
        plt.close()


        # vel\acc bar plots with std error bars

        def bar_val_acc(vel_mean_per_block, vel_std_per_block, acc_mean_per_block, acc_std_per_block, block_centers, folder, args, ball):
            for label, mean, std, metric in [
                ("Velocity", vel_mean_per_block, vel_std_per_block, "Velocity"),
                ("Acceleration", acc_mean_per_block, acc_std_per_block, "Acceleration")
            ]:
                plt.figure(figsize=(10, 5))
                plt.bar(block_centers, mean, yerr=std, capsize=5)
                if ball:
                    plt.title(f"{label} per Block (Mean ± Std across Sims of Ball)")
                else:
                    plt.title(f"{label} per Block (Mean ± Std across Agents & Sims)")
                plt.xlabel("Field X Position (meters)")
                plt.ylabel(metric)
                plt.tight_layout()
                if ball:
                    plt.savefig(
                        f"{folder}/ball_barplot_{label}_per_block_{args.dataset}_{time_sim}s_{args.timestamp}.png")
                else:
                    plt.savefig(
                        f"{folder}/agents_barplot_{label}_per_block_{args.dataset}_{time_sim}s_{args.timestamp}.png")
                plt.close()

        velocity_means_sim_block_agents = velocity_means_sim[:10, :].mean(axis=0)
        velocity_std_sim_block_agents = velocity_stds_sim[:10, :].mean(axis=0)
        acc_means_sim_block_agents = accel_means_sim[:10, :].mean(axis=0)
        acc_std_sim_block_agents = accel_stds_sim[:10, :].mean(axis=0)
        df["Velocity Mean (All Sims only Players)"] = velocity_means_sim_block_agents
        df["Velocity Std (All Sims only Players)"] = velocity_std_sim_block_agents
        df["Acceleration Mean (All Sims only Players)"] = acc_means_sim_block_agents
        df["Acceleration Std (All Sims only Players)"] = acc_std_sim_block_agents
        bar_val_acc(velocity_means_sim_block_agents, velocity_std_sim_block_agents, acc_means_sim_block_agents, acc_std_sim_block_agents, block_centers, folder, args, False)


        velocity_means_sim_block_ball = velocity_means_sim[ 10, :]
        velocity_std_sim_block_ball = velocity_stds_sim[ 10, :]
        acc_means_sim_block_ball = accel_means_sim[ 10, :]
        acc_std_sim_block_ball = accel_stds_sim[10, :]
        df["Velocity Mean (All Sims only Ball)"] = velocity_means_sim_block_ball
        df["Velocity Std (All Sims only Ball)"] = velocity_std_sim_block_ball
        df["Acceleration Mean (All Sims only Ball)"] = acc_means_sim_block_ball
        df["Acceleration Std (All Sims only Ball)"] = acc_std_sim_block_ball
        bar_val_acc(velocity_means_sim_block_ball, velocity_std_sim_block_ball, acc_means_sim_block_ball,
                    acc_std_sim_block_ball, block_centers, folder, args, True)

        plt.figure(figsize=(10, 5))
        plt.bar(block_centers, df["location Mean (All Sims All agents)"],
                yerr=df["location Std (All Sims All agents)"], capsize=5)
        plt.title(f"Average Percentage of Time Spent (%) of Agents per Block ({args.dataset}) – {time_sim}s")
        plt.xlabel("Field X Position (meters)")
        plt.ylabel(f"Percentage of Time Spent (%) out of {total_sim_time}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{folder}/location_mean_per_block_{args.dataset}_{time_sim}s_{args.timestamp}.png")
        plt.close()
        return df
    return None

def compute_kde_2d(data, grid_size=100, bounds=[[0, 25], [0, 14]]):
    if isinstance(data, list):
        data = np.vstack(data)  #  shape (P, 2)
    else:
        data = np.asarray(data)
        # if it's (F, T, 2), flattening time too:
        if data.ndim == 3:
            data = data.reshape(-1, 2)
    x, y = data[:, 0], data[:, 1]
    kde = gaussian_kde([x, y])
    x_grid = np.linspace(bounds[0][0], bounds[0][1], grid_size)
    y_grid = np.linspace(bounds[1][0], bounds[1][1], grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz / zz.sum()

def empirical_histogram2d(points, field_length, field_width, weights = None, flex_n = False):

    nx, ny = 100, 100 #reslution

    if flex_n:
        points = np.concatenate([p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in points], axis=0)

    points = np.asarray(points, dtype=float)
    if points.ndim > 2:
        #shape (S, T, 2) to (S*T, 2)
        points = points.reshape(-1, 2)

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_max = max(x_max, field_length)
    y_max = max(y_max, field_width)
    x_min = min(x_min, field_length)
    y_min = min(y_min, field_width)

    x_edges = np.linspace(x_min, x_max, nx+1)
    y_edges = np.linspace(y_min, y_max, ny+1)

    if weights is None:
        H, _, _ = np.histogram2d(points[:, 0], points[:, 1],
                                 bins=[x_edges, y_edges])

    else:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim > 1:
            weights = weights.ravel()
        H, _, _ = np.histogram2d(points[:, 0], points[:, 1],
                                   bins=[x_edges, y_edges],
                                   weights=weights)

    return H / H.sum()

def compute_metrics_hist(hist_gt, hist_sim, field_width, field_length, folder, args, hist_name):
    p = hist_gt.ravel()
    q = hist_sim.ravel()

    js = jensenshannon(p, q)
    l2 = np.linalg.norm(p - q)
    cosine_sim = np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))


    n_xticks = 6
    n_yticks = 6
    nx, ny = hist_gt.shape

    x_ticks = np.linspace(0, nx, n_xticks)
    y_ticks = np.linspace(0, ny, n_yticks)
    x_labels = np.linspace(0, field_length, n_xticks).round(1)
    y_labels = np.linspace(0, field_width, n_yticks).round(1)


    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(hist_gt.T,ax=ax1,cmap="Reds",cbar=True,xticklabels=False,yticklabels=False)
    ax1.invert_yaxis()
    ax1.set_title("Empirical Heatmap - Ground Truth")
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels)
    ax1.set_xlabel("X (m)")
    ax0.set_ylabel("Y (m)")

    sns.heatmap(hist_sim.T, ax=ax0, cmap="Blues", cbar=True, xticklabels=False,  yticklabels=False)
    ax0.invert_yaxis()
    ax0.set_title("Empirical Heatmap - Simulation")
    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(x_labels)
    ax0.set_yticks(y_ticks)
    ax0.set_yticklabels(y_labels)
    ax0.set_xlabel("X (m)")
    plt.tight_layout()
    if os.path.isdir(f"{folder}/Emirical_Hist/"):
        print("")
    else:
        os.makedirs(f"{folder}/Emirical_Hist/", exist_ok=True)
    plt.savefig(
            f"{folder}/Emirical_Hist/{hist_name}.png")
    plt.close()


    return js, l2, cosine_sim

def calculate_empiracal_hist_comp_plots(dataset, all_centroids, all_trajectories,all_centroids_gt, all_trajectories_gt, field_length, field_width,
                                        xy_values, xy_vals,
                                        velocity, acc, velocity_gt, acc_gt, folder,args,
                                        all_centroids_gen=None, all_trajectories_gen=None,velocity_gen=None, acc_gen=None,xy_vals_gen=None):

    # print("all_centroids_gt, ", np.asarray(all_centroids_gt, dtype=float).shape)
    # print("all_centroids, ", len(all_centroids))
    if dataset in ['sdd', 'eth']:
        flex_n = True
    else:
        flex_n = False
    hist_gt_cen = empirical_histogram2d(all_centroids_gt, field_length, field_width)
    hist_sim_cen = empirical_histogram2d(all_centroids, field_length, field_width,flex_n = flex_n)

    print("all_trajectories_gt", all_trajectories_gt.shape)
    print("all_trajectories_gt X", all_trajectories_gt[:,0].min(), all_trajectories_gt[:,0].max())
    print("all_trajectories_gt Y", all_trajectories_gt[:,1].min(), all_trajectories_gt[:,1].max())

    hist_gt_agents = empirical_histogram2d(all_trajectories_gt, field_length, field_width)
    hist_sim_agents =  empirical_histogram2d(all_trajectories, field_length, field_width)

    hist_gt_vel = empirical_histogram2d(xy_vals.T, field_length, field_width, velocity_gt)
    hist_sim_vel = empirical_histogram2d(xy_values.T, field_length, field_width, velocity)
    hist_gt_acc = empirical_histogram2d(xy_vals.T, field_length, field_width, acc_gt)
    hist_sim_acc =  empirical_histogram2d(xy_values.T, field_length, field_width,acc)

    if all_centroids_gen:
        hist_gen_cen = empirical_histogram2d(all_centroids_gen, field_length, field_width,flex_n = flex_n)
        hist_gen_agents = empirical_histogram2d(all_trajectories_gen, field_length, field_width)
        hist_gen_vel = empirical_histogram2d(xy_vals_gen.T, field_length, field_width, velocity_gen)
        hist_gen_acc = empirical_histogram2d(xy_vals_gen.T, field_length, field_width, acc_gen)

        js_cen_gen, l2_cen_gen, cosine_sim_cen_gen = compute_metrics_hist(hist_gen_cen, hist_sim_cen, field_width, field_length,folder, args, "centriods_gen")
        js_agents_gen, l2_agents_gen, cosine_sim_agents_gen = compute_metrics_hist(hist_gen_agents,field_width, field_length, hist_sim_agents, folder, args, "agents_gen")
        js_vel_gen, l2_vel_gen, cosine_sim_vel_gen = compute_metrics_hist(hist_gen_vel, hist_sim_vel, field_width, field_length,folder, args, "velocity_gen")
        js_acc_gen, l2_acc_gen, cosine_sim_acc_gen = compute_metrics_hist(hist_gen_acc, hist_sim_acc,field_width, field_length, folder, args, "acceleration_gen")

        with open(f"{folder}/Emirical_Hist/metrics_gen.txt", "w") as f:
            f.write(f"js_cen_gen: {js_cen_gen}\n")
            f.write(f"l2_cen_gen: {l2_cen_gen}\n")
            f.write(f"cosine_sim_cen_gen: {cosine_sim_cen_gen}\n")

            f.write(f"js_agents_gen: {js_agents_gen}\n")
            f.write(f"l2_agents_gen: {l2_agents_gen}\n")
            f.write(f"cosine_sim_agents_gen: {cosine_sim_agents_gen}\n")

            f.write(f"js_vel_gen: {js_vel_gen}\n")
            f.write(f"l2_vel_gen: {l2_vel_gen}\n")
            f.write(f"cosine_sim_vel_gen: {cosine_sim_vel_gen}\n")

            f.write(f"js_acc_gen: {js_acc_gen}\n")
            f.write(f"l2_acc_gen: {l2_acc_gen}\n")
            f.write(f"cosine_sim_acc_gen: {cosine_sim_acc_gen}\n")

    js_cen, l2_cen, cosine_sim_cen = compute_metrics_hist(hist_gt_cen, hist_sim_cen, field_width, field_length,folder, args, "centriods_gt")
    js_agents, l2_agents, cosine_sim_agents = compute_metrics_hist(hist_gt_agents, hist_sim_agents, field_width, field_length, folder, args, "agents_gt")
    js_vel, l2_vel, cosine_sim_vel = compute_metrics_hist(hist_gt_vel, hist_sim_vel, field_width, field_length,folder, args, "velocity_gt")
    js_acc, l2_acc, cosine_sim_acc = compute_metrics_hist(hist_gt_acc, hist_sim_acc, field_width, field_length,folder, args, "acceleration_gt")

    with open(f"{folder}/Emirical_Hist/metrics_gt.txt", "w") as f:
        f.write(f"js_cen: {js_cen}\n")
        f.write(f"l2_cen: {l2_cen}\n")
        f.write(f"cosine_sim_cen: {cosine_sim_cen}\n")

        f.write(f"js_agents: {js_agents}\n")
        f.write(f"l2_agents: {l2_agents}\n")
        f.write(f"cosine_sim_agents: {cosine_sim_agents}\n")

        f.write(f"js_vel: {js_vel}\n")
        f.write(f"l2_vel: {l2_vel}\n")
        f.write(f"cosine_sim_vel: {cosine_sim_vel}\n")

        f.write(f"js_acc: {js_acc}\n")
        f.write(f"l2_acc: {l2_acc}\n")
        f.write(f"cosine_sim_acc: {cosine_sim_acc}\n")

    print(f"js_cen: {js_cen}\n",f"js_agents: {js_agents}\n", f"js_vel: {js_vel}\n",f"js_acc: {js_acc}\n")


def compute_polarization(trajectory):
    # trajectory: [T, N, 2]


    trajectory = np.asarray(trajectory)
    # if trajectory.dtype == object:
    #     trajectory = trajectory.astype(np.float32, copy=False)
    # if trajectory.ndim != 3 or trajectory.shape[0] < 2:
    #     print("Invalid trajectory shape:", trajectory.shape)
    #     return np.nan

    vel = np.diff(trajectory, axis=0)  # [T-1, N, 2]

    norm = np.linalg.norm(vel, axis=-1, keepdims=True) + 1e-8
    unit_vel = vel / norm
    polarization = np.linalg.norm(np.sum(unit_vel, axis=1), axis=1) / unit_vel.shape[1]  # [T-1]
    return polarization.mean()

def compute_milling(trajectory):
    # trajectory: [T, N, 2]
    pos = trajectory[:-1]  # [T-1, N, 2]
    vel = np.diff(trajectory, axis=0)  # [T-1, N, 2]
    com = pos.mean(axis=1, keepdims=True)  # [T-1, 1, 2]
    rel_pos = pos - com
    rel_pos /= (np.linalg.norm(rel_pos, axis=-1, keepdims=True) + 1e-8)
    unit_vel = vel / (np.linalg.norm(vel, axis=-1, keepdims=True) + 1e-8)
    angular_momentum = np.cross(rel_pos, unit_vel)  # [T-1, N]
    milling = np.abs(angular_momentum.mean(axis=1))  # [T-1]
    return milling.mean()

def compute_swarming(trajectory):
    # trajectory: [T, N, 2]
    com = trajectory.mean(axis=1, keepdims=True)  # [T, 1, 2]
    dispersion = np.linalg.norm(trajectory - com, axis=-1)  # [T, N]
    return dispersion.mean()  # average distance from COM

def compute_group_metrics(all_trajectories, GT=''):
    polarization_scores = []
    milling_scores = []
    swarming_scores = []
    # if GT == 'GT':
    #     T_target = 305
    #
    #     flat = all_trajectories.reshape(-1, 8, 2)
    #     total_steps = flat.shape[0]
    #     sim_num = total_steps // T_target
    #     usable_steps = sim_num * T_target
    #     flat = flat[:usable_steps]
    #     reshaped = flat.reshape(sim_num, T_target, 8, 2)

    for i, traj in enumerate(all_trajectories):

        polarization_scores.append(compute_polarization(traj))
        milling_scores.append(compute_milling(traj))
        swarming_scores.append(compute_swarming(traj))
    return {
        "polarization_mean": np.mean(polarization_scores),
        "polarization_std": np.std(polarization_scores),
        "milling_mean": np.mean(milling_scores),
        "milling_std": np.std(milling_scores),
        "swarming_mean": np.mean(swarming_scores),
        "swarming_std": np.std(swarming_scores),
    }

def save_group_metrics_to_csv(sim_metrics, gt_metrics, path="group_metrics.csv"):
    records = []
    for key in sim_metrics:
        sim_val = sim_metrics[key]
        gt_val = gt_metrics[key]
        diff = sim_val - gt_val
        rel_diff = abs(diff) / (gt_val + 1e-8)
        records.append({
            "Metric": key,
            "Simulation": sim_val,
            "Ground Truth": gt_val,
            "Abs Difference": diff,
            "Relative Difference (%)": rel_diff * 100
        })

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    print(f"Saved metrics to: {path}")


def analyze_usage( all_trajectories, all_centroids, field_length,field_width , num_blocks, timestep_duration,
                   args, folder, constant, gen_gt_type=None):
    """
    all_centroids: List of arrays [T, 2] for each simulation
    field_length: total field length (X dimension)
    num_blocks: number of vertical blocks
    timestep_duration: time per step ( 0.4)
    """
    EPS = 1e-10
    all_trajectories = [traj.transpose(1, 0, 2) for traj in all_trajectories] #sims, T, N, 2
    if args.dataset == 'sdd':
        for i in range(len(all_trajectories)):
            all_trajectories[i][:, :, 0] = (all_trajectories[i][:, :, 0] / 100.0) * (1951.0 - 9.0) + 9.0
            all_trajectories[i][:, :, 1] = (all_trajectories[i][:, :, 1] / 100.0) * (1973.0 - 7.0) + 7.0

            all_centroids[i][:, 0] = (all_centroids[i][:, 0] / 100.0) * (1951.0 - 9.0) + 9.0
            all_centroids[i][:, 1] = (all_centroids[i][:, 1] / 100.0) * (1973.0 - 7.0) + 7.0


    print("all_trajectories", len(all_trajectories))
    N = all_trajectories[0].shape[1]
    df_gt = pd.read_csv(constant.gt_path)
    data = np.load(constant.gt_path_data, allow_pickle=True)



    df_gt = df_gt.sort_values("Block Index").reset_index(drop=True)
    all_trajectories_gt = data["trajectories"]
    all_trajectories_gt = all_trajectories_gt.astype(np.float64)

    all_centroids_gt = data["centroids"]
    xy_vals = data["xy_vals"]
    vel_vals = data["vel_vals"]
    acc_vals =  data["acc_vals"]
    xy_vals = np.array(xy_vals).T
    vel_vals = np.array(vel_vals)
    acc_vals = np.array(acc_vals)

    if args.dataset == 'target':
        sim_metrics = compute_group_metrics(all_trajectories)
        gt_metrics = compute_group_metrics(all_trajectories_gt, "GT")
        save_group_metrics_to_csv(sim_metrics, gt_metrics, path=f'{folder}/fish_group_metrics.csv')

        for key in sim_metrics:
            sim_val = sim_metrics[key]
            gt_val = gt_metrics[key]
            diff = sim_val - gt_val
            rel_diff = abs(diff) / (gt_val + 1e-8)
            print(f"{key}: Sim={sim_val:.4f}, GT={gt_val:.4f}, Diff={diff:.4f}, RelDiff={rel_diff:.2%}")

    if gen_gt_type:
        if gen_gt_type == 'groupnet':
            gen_df_gt = pd.read_csv(constant.groupnet_gt_path)
            data_gen = np.load(constant.groupnet_gt_path_data, allow_pickle=True)
        else:
            gen_df_gt = pd.read_csv(constant.sampler_gt_path)
            data_gen = np.load(constant.sampler_gt_path_data, allow_pickle=True)

        gen_df_gt = gen_df_gt.sort_values("Block Index").reset_index(drop=True)
        all_trajectories_gt_gen = data_gen["trajectories"]
        all_centroids_gt_gen = data_gen["centroids"]
        all_centroids_gt_gen = np.array(all_centroids_gt_gen)
        all_trajectories_gt_gen = [np.asarray(traj, dtype=np.float64) for traj in all_trajectories_gt_gen]
        all_centroids_gt_gen = [np.asarray(traj, dtype=np.float64) for traj in all_centroids_gt_gen]


        xy_vals_gen = data_gen["xy_vals"]
        vel_vals_gen = data_gen["vel_vals"]
        acc_vals_gen = data_gen["acc_vals"]
        xy_vals_gen = np.array(xy_vals_gen)
        vel_vals_gen = np.array(vel_vals_gen)
        acc_vals_gen = np.array(acc_vals_gen)




        if args.dataset == 'target':
            sim_metrics = compute_group_metrics(all_trajectories)
            gt_metrics = compute_group_metrics(all_trajectories_gt_gen)
            save_group_metrics_to_csv(sim_metrics, gt_metrics, path=f'{folder}/fish_group_metrics_gen.csv')
            for key in sim_metrics:
                sim_val = sim_metrics[key]
                gt_val = gt_metrics[key]
                diff = sim_val - gt_val
                rel_diff = abs(diff) / (gt_val + 1e-8)
                print(f" GEN DATA {key}: Sim={sim_val:.4f}, GT={gt_val:.4f}, Diff={diff:.4f}, RelDiff={rel_diff:.2%}")

    if args.dataset == 'sdd':
        block_edges = np.linspace(9, field_length, num_blocks + 1)
    else:
        block_edges = np.linspace(constant.X_MIN, field_length, num_blocks + 1)
    block_times_all_avg = np.zeros((len(all_centroids), num_blocks))  # [sims, blocks]
    avg_stay_per_block_all = np.zeros((len(all_centroids), num_blocks))  # [sims, blocks]
    block_centers = (block_edges[:-1] + block_edges[1:]) / 2


    for sim_idx, centroids in enumerate(all_centroids): #S, T-changes , 2
        x_coords = centroids[:, 0]
        current_length = centroids.shape[0]
        average_per_block = np.zeros(num_blocks)
        block_entries = [[] for _ in range(num_blocks)]
        current_block = None
        entry_time = None

        #calcualtes percentage time of centroid in certain area, and average stay in area per each enterence.
        for t, x in enumerate(x_coords):
            block = np.searchsorted(block_edges, x, side='right') - 1 # finds block index for x
            block = np.clip(block, 0, num_blocks - 1)
            # average_per_block[block] += timestep_duration/args.length # 0.4/120
            average_per_block[block] += 1/current_length # 0.4/(sim timesteps length*0.4) = seconds

            if block != current_block:
                if current_block is not None:
                    duration = t - entry_time
                    block_entries[current_block].append(duration * timestep_duration)
                current_block = block
                entry_time = t
        if current_block is not None:
            duration = len(centroids) - entry_time
            block_entries[current_block].append(duration * timestep_duration)

        avg_stay_per_entry = [np.mean(times) if times else 0 for times in block_entries]
        avg_stay_per_block_all[sim_idx] = avg_stay_per_entry
        block_times_all_avg[sim_idx] = average_per_block


    avg_time_per_block_sim = block_times_all_avg.mean(axis=0)
    std_time_per_block_sim = block_times_all_avg.std(axis=0)
    avg_stay_per_block_sim = avg_stay_per_block_all.mean(axis=0)
    std_stay_per_block_sim = avg_stay_per_block_all.std(axis=0)



    velocity_means_sim = np.zeros((len(all_trajectories), N, num_blocks))
    velocity_stds_sim = np.zeros_like(velocity_means_sim)
    accel_means_sim = np.zeros_like(velocity_means_sim)
    accel_stds_sim = np.zeros_like(velocity_means_sim)
    agent_block_time_sim0 = np.zeros((N, num_blocks))
    agent_block_time = np.zeros((len(all_trajectories), N, num_blocks))



    xy_values = []  # all agent x,y positions
    vel_values = []  # corresponding velocity magnitudes
    xy_acc_values = []
    acc_values = []

    for j, sim in enumerate(all_trajectories):  # (T, N, 2)
        vel = np.diff(sim, axis=0)/ timestep_duration #T-1, N, 2
        acc = np.diff(vel, axis=0)/ timestep_duration #T-2, N, 2

        current_length = sim.shape[0]

        vel_for_map = np.linalg.norm(np.diff(sim, axis=0)/timestep_duration, axis=-1)  # T-1, N
        acc_for_map = np.linalg.norm(acc, axis=-1)   # shape (T-2, N)

        # if j ==0:
        #     print("sim", sim)
        #     print("vel", vel)
        #     print("acc", acc)

        for agent in range(sim.shape[1]):
            agent_x = sim[:-1, agent, 0] #T-1

            xy_values.extend(sim[2:, agent])  # T-2 , 2
            vel_values.extend(vel_for_map[1:, agent])  # T-2
            xy_acc_values.extend(sim[2:, agent]) # T-2, 2
            acc_values.extend(acc_for_map[:, agent])

            agent_blocks = np.clip(np.searchsorted(block_edges, agent_x, side='right') - 1, 0, num_blocks - 1)

            for b in range(num_blocks):

                mask_v = (agent_blocks == b) #current block
                # if j ==0 :
                    # print("mask_v", mask_v)
                v_in_block = np.linalg.norm(vel[mask_v, agent], axis=-1)
                # if j ==0 :
                    # print("v_in_block", v_in_block, "avg", v_in_block.mean())
                mask_a = (agent_blocks[:-1] == b)
                a_in_block = np.linalg.norm(acc[mask_a, agent], axis=-1) if acc.shape[0] > 0 else []
                # if j ==0 :
                #     print("a_in_block", a_in_block)

                if len(v_in_block) > 0:
                    velocity_means_sim[j, agent, b] = v_in_block.mean()
                    velocity_stds_sim[j, agent, b] = v_in_block.std()
                if len(a_in_block) > 0:
                    accel_means_sim[j, agent, b] = a_in_block.mean()
                    accel_stds_sim[j, agent, b] = a_in_block.std()


                # occupancy
                block_time = np.sum(mask_v) * timestep_duration
                agent_block_time[j, agent, b] += block_time / (current_length * timestep_duration)
                if j == 0:
                    agent_block_time_sim0[agent, b] += block_time

    #to make sure acc do not contain invalid values:
    valid_mask = (
            ~np.isnan(acc_values)
            & ~np.isinf(acc_values)
            & ~np.isnan(xy_acc_values).any(axis=1)
            & ~np.isinf(xy_acc_values).any(axis=1)
    )

    xy_values = np.array(xy_values)
    vel_values = np.array(vel_values)
    xy_acc_values = np.array(xy_acc_values)
    acc_values = np.array(acc_values)

    xy_acc_values = xy_acc_values[valid_mask]
    acc_values = acc_values[valid_mask]
    xy_values = xy_values[valid_mask]
    vel_values = vel_values[valid_mask]

    xy_values = xy_values.T
    xy_acc_values = xy_acc_values.T
    print(f"Filtered out {np.sum(~valid_mask)} invalid entries out of {len(valid_mask)}")

    print("xy_values", xy_values.shape)



    # per-agent metrics across blocks
    velocity_mean = velocity_means_sim.mean(axis=0)  # (agent_num, num_blocks)
    velocity_std = velocity_stds_sim.mean(axis=0)
    accel_mean = accel_means_sim.mean(axis=0)
    accel_std = accel_stds_sim.mean(axis=0)
    agent_block_percent = agent_block_time
    agent_block_percent_sims = agent_block_percent.mean(axis=0)



    if args.dataset == "nba":
        ball_block_percent_sims_mean = agent_block_percent[:, 10, :].mean(0) #only ball ->num_blocks
        ball_block_percent_sims_std = agent_block_percent[:, 10, :].std(0)  # only ball ->num_blocks
        plt.figure(figsize=(10, 5))
        plt.bar(block_centers, ball_block_percent_sims_mean, yerr=ball_block_percent_sims_std, capsize=5)
        plt.title(f"Centroid Field Percentage of Time Usage Histogram of Ball ({args.dataset}) - {args.mission_num} Missions - {args.length}s - Simulations:{args.sim_num} - Buffer:{constant.buffer}")
        plt.xlabel("Field X Position (meters)")
        plt.ylabel(f"Percentage of Time Spent (%) out of {args.length}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"{folder}/centroid_avg_usage_histogram_ball.png")
        plt.close()


        agent_block_percent_sim0 = agent_block_time_sim0
        #row represents a block
        df = pd.DataFrame({
            "Block Index": range(num_blocks),
            "Block Center (m)": block_centers,
            "Avg Time in Block (%)": avg_time_per_block_sim,
            "Std Time in Block (%)": std_time_per_block_sim,
            "Avg Stay Per Entry (s)": avg_stay_per_block_sim,
            "Std Stay Per Entry (s)": std_stay_per_block_sim
        })

        vel_mean_per_block = velocity_means_sim.mean(axis=(0, 1))  # average over all agents in all sims per block
        vel_std_per_block = velocity_stds_sim.mean(axis=(0, 1))
        acc_mean_per_block = accel_means_sim.mean(axis=(0, 1))
        acc_std_per_block = accel_stds_sim.mean(axis=(0, 1))
        avg_x_pos_sim_per_block = agent_block_percent.mean(axis=(0, 1))
        std_x_pos_sim_per_block = agent_block_percent.mean(axis=(0, 1))

        df["Velocity Mean (All Sims All agents)"] = vel_mean_per_block
        df["Velocity Std (All Sims All agents)"] = vel_std_per_block
        df["Acceleration Mean (All Sims All agents)"] = acc_mean_per_block
        df["Acceleration Std (All Sims All agents)"] = acc_std_per_block
        df["location Mean (All Sims All agents)"] = avg_x_pos_sim_per_block
        df["location Std (All Sims All agents)"] = std_x_pos_sim_per_block

        for agent in range(N):
            for b in range(num_blocks):
                df.loc[b, f"Agent{agent}_Vel_Mean"] = velocity_mean[agent, b]
                df.loc[b, f"Agent{agent}_Vel_Std"] = velocity_std[agent, b]
                df.loc[b, f"Agent{agent}_Acc_Mean"] = accel_mean[agent, b]
                df.loc[b, f"Agent{agent}_Acc_Std"] = accel_std[agent, b]
                df.loc[b, f"Agent{agent}_Occupancy_Pct"] = agent_block_percent_sims[agent, b]


        plt.figure(figsize=(10, 5))
        plt.bar(block_centers, df["Avg Time in Block (%)"], capsize=5)
        plt.tick_params(axis='both', which='major', labelsize=16)

        max_y = df["Avg Time in Block (%)"].max()
        yticks = np.linspace(0, max_y, 5)
        plt.yticks(yticks, [f"{tick:.2f}" for tick in yticks])

        # plt.bar(block_centers, df["Avg Time in Block (%)"], yerr=df["Std Time in Block (%)"], capsize=5)
        plt.title(r"(c) Spatial Occupancy Distribution of" "\n" r"Agents' Centroids - $S_{HG}$", fontsize=22, pad=15)
        plt.xlabel("Spatial Zones (m)", fontsize=20)
        plt.ylabel(f"Block Occupancy Fraction", fontsize=20)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"{folder}/centroid_avg_usage_histogram_percentage.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(block_centers, df["Avg Stay Per Entry (s)"], yerr=df["Std Stay Per Entry (s)"], capsize=5)
        plt.title(f"Centroid Field Average Time Usage Histogram ({args.dataset}) - {args.mission_num} Missions - {args.length}s - Simulations:{args.sim_num} - Buffer:{constant.buffer}")
        plt.xlabel("Field X Position (meters)")
        plt.ylabel("Average Time Spent (s) per Block")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{folder}/centroid_avg_usage_histogram_per_Block.png")
        plt.close()

        # for one simulation
        plt.figure(figsize=(10, 5))
        x = block_centers
        for agent in range(N):
            y = agent_block_percent_sim0[agent]
            sns.lineplot(x=x, y=y, label=f"Agent {agent}")

        plt.title("Field Occupancy per Agent (Simulation 0)")
        plt.xlabel("Field X Position (meters)")
        plt.ylabel("Time % (in block)")
        plt.xticks(block_centers)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{folder}/Field_Block_Occupancy_Per_Agent_SIM0_.png")
        plt.close()


        # vel/acc bar plots with std error bars

        def bar_val_acc(vel_mean_per_block, vel_std_per_block, acc_mean_per_block, acc_std_per_block, block_centers, folder, args, ball):
            for label, mean, std, metric in [
                ("Velocity", vel_mean_per_block, vel_std_per_block, "Velocity"),
                ("Acceleration", acc_mean_per_block, acc_std_per_block, "Acceleration")
            ]:
                plt.figure(figsize=(10, 5))
                plt.bar(block_centers, mean, yerr=std, capsize=5)
                if ball:
                    plt.title(f"{label} per Block (Mean ± Std across Sims of Ball)")
                else:
                    plt.title(f"{label} per Block (Mean ± Std across Agents & Sims)")
                plt.xlabel("Field X Position (meters)")
                plt.ylabel(metric)
                plt.tight_layout()
                if ball:
                    plt.savefig(
                        f"{folder}/ball_barplot_{label}_per_block_.png")
                else:
                    plt.savefig(
                        f"{folder}/agents_barplot_{label}_per_block_.png")
                plt.close()

        velocity_means_sim_block_agents = velocity_means_sim[:, :10, :].mean(axis=(0, 1))
        velocity_std_sim_block_agents = velocity_stds_sim[:, :10, :].mean(axis=(0, 1))
        acc_means_sim_block_agents = accel_means_sim[:, :10, :].mean(axis=(0, 1))
        acc_std_sim_block_agents = accel_stds_sim[:, :10, :].mean(axis=(0, 1))
        df["Velocity Mean (All Sims only Players)"] = velocity_means_sim_block_agents
        df["Velocity Std (All Sims only Players)"] = velocity_std_sim_block_agents
        df["Acceleration Mean (All Sims only Players)"] = acc_means_sim_block_agents
        df["Acceleration Std (All Sims only Players)"] = acc_std_sim_block_agents
        bar_val_acc(velocity_means_sim_block_agents, velocity_std_sim_block_agents, acc_means_sim_block_agents, acc_std_sim_block_agents, block_centers, folder, args, False)

        velocity_means_sim_block_ball = velocity_means_sim[:, 10, :].mean(axis=(0))
        velocity_std_sim_block_ball = velocity_stds_sim[:, 10, :].std(axis=0)
        acc_means_sim_block_ball = accel_means_sim[:, 10, :].mean(axis=(0))
        acc_std_sim_block_ball = accel_stds_sim[:, 10, :].std(axis=0)
        df["Velocity Mean (All Sims only Ball)"] = velocity_means_sim_block_ball
        df["Velocity Std (All Sims only Ball)"] = velocity_std_sim_block_ball
        df["Acceleration Mean (All Sims only Ball)"] = acc_means_sim_block_ball
        df["Acceleration Std (All Sims only Ball)"] = acc_std_sim_block_ball
        bar_val_acc(velocity_means_sim_block_ball, velocity_std_sim_block_ball, acc_means_sim_block_ball,
                    acc_std_sim_block_ball, block_centers, folder, args, True)

        plt.figure(figsize=(10, 5))
        plt.bar(block_centers, df["location Mean (All Sims All agents)"],
                yerr=df["location Std (All Sims All agents)"], capsize=5)
        plt.title(f"Average Agent X‐Position per Block ({args.dataset}) – {args.length}s")
        plt.xlabel("Field X Position (meters)")
        plt.ylabel("Average X (meters)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{folder}/location_mean_per_block.png")
        plt.close()


    # KDE over (x,y) weighted by velocity
    kde_vel = gaussian_kde(xy_values, weights=vel_values)
    kde_acc = gaussian_kde(xy_acc_values, weights=acc_values)
    kde_vel_gt = gaussian_kde(xy_vals, weights=vel_vals)
    kde_acc_gt = gaussian_kde(xy_vals, weights=acc_vals)

    x_min, x_max = xy_values[0, :].min(), xy_values[0, :].max()
    y_min, y_max = xy_values[ 1, :].min(), xy_values[1, :].max()
    x_max = max(x_max, field_length)
    y_max = max(y_max, field_width)
    x_min = min(x_min, field_length)
    y_min = min(y_min, field_width)
    # x_max = 105
    # y_max = 105
    # x_min = 0
    # y_min = 0


    if gen_gt_type:
        xx_gen_gt, yy_gen_gt, kde_gen_gt = compute_kde_2d(all_centroids_gt_gen,
                                                          bounds=[[x_min, x_max], [y_min, y_max]])

    print("XYs", x_min, x_max, y_min, y_max)


    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 100 + 1),
        np.linspace(y_min, y_max, 100 + 1)
    )

    n_xticks = 5
    n_yticks = 5
    res = 100
    x_ticks = np.linspace(0, res, n_xticks)
    x_labels = np.linspace(x_min, x_max, n_xticks).round(1)
    y_ticks = np.linspace(0, res, n_yticks)
    y_labels = np.linspace(y_min, y_max, n_yticks).round(1)

    xx_gt, yy_gt, kde_gt = compute_kde_2d(all_centroids_gt, bounds=[[x_min, x_max], [y_min, y_max]])

    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()])
    vel_heatmap = kde_vel(grid_points).reshape(grid_x.shape)
    vel_heatmap_gt = kde_vel_gt(grid_points).reshape(grid_x.shape)
    acc_heatmap = kde_acc(grid_points).reshape(grid_x.shape)
    acc_heatmap_gt = kde_acc_gt(grid_points).reshape(grid_x.shape)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(vel_heatmap, ax=axs[0], cmap="Blues", cbar=True, xticklabels=False,  yticklabels=False)
    axs[0].invert_yaxis()
    axs[0].set_title("Velocity KDE Heatmap - Simulated")
    axs[0].set_xticks(x_ticks)
    axs[0].set_xticklabels(x_labels)
    axs[0].set_yticks(y_ticks)
    axs[0].set_yticklabels(y_labels)
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    sns.heatmap(vel_heatmap_gt, ax=axs[1], cmap="Reds", cbar=True, xticklabels=False,  yticklabels=False)
    axs[1].invert_yaxis()
    axs[1].set_title("Velocity KDE Heatmap - Ground Truth")
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels(x_labels)
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels(y_labels)
    axs[1].set_xlabel("X (m)")
    plt.tight_layout()
    plt.savefig(
        f"{folder}/velocity_with_gt_heatmap.png")
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(acc_heatmap, ax=axs[0], cmap="Blues", cbar=True, xticklabels=False,  yticklabels=False)
    axs[0].invert_yaxis()
    axs[0].set_title("Acceleration KDE Heatmap - Simulated")
    axs[0].set_xticks(x_ticks)
    axs[0].set_xticklabels(x_labels)
    axs[0].set_yticks(y_ticks)
    axs[0].set_yticklabels(y_labels)
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    sns.heatmap(acc_heatmap_gt, ax=axs[1], cmap="Reds", cbar=True, xticklabels=False,  yticklabels=False)
    axs[1].invert_yaxis()
    axs[1].set_title("Acceleration KDE Heatmap - Ground Truth")
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels(x_labels)
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels(y_labels)
    axs[1].set_xlabel("X (m)")
    plt.tight_layout()
    plt.savefig(
        f"{folder}/acc_with_gt_heatmap.png")
    plt.close()

    flattened_sims = [sim.reshape(-1, 2) for sim in all_trajectories]
    all_agent_positions = np.vstack(flattened_sims)
    xx_agents, yy_agents, kde_agents = compute_kde_2d(all_agent_positions ,bounds=[[x_min, x_max], [y_min, y_max]])
    all_agent_positions_gt = all_trajectories_gt.reshape(-1, 2)
    xx_agents_gt, yy_agents_gt, kde_agents_gt = compute_kde_2d(all_agent_positions_gt,bounds=[[x_min, x_max], [y_min, y_max]])

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(kde_agents_gt* 1000, ax=axs[1], cmap="Reds", cbar=True, xticklabels=False,  yticklabels=False)
    axs[1].invert_yaxis()
    axs[1].set_title("(a) 2D KDE of Agents' Position\nDistribution - Ground Truth", fontsize=22)
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels(x_labels, fontsize=14)
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels(y_labels, fontsize=14)
    axs[1].set_xlabel("X", fontsize=16)
    axs[1].set_ylabel("Y", fontsize=16)
    sns.heatmap(kde_agents* 1000, ax=axs[0], cmap="Blues", cbar=True, xticklabels=False,  yticklabels=False)
    axs[0].invert_yaxis()
    axs[0].set_title(r"(c) 2D KDE of Agents' Position" "\n" r"Distribution - $S_{HG}$", fontsize=22)
    axs[0].set_xticks(x_ticks)
    axs[0].set_xticklabels(x_labels, fontsize=14)
    axs[0].set_yticks(y_ticks)
    axs[0].set_yticklabels(y_labels, fontsize=14)
    axs[0].set_xlabel("X", fontsize=16)
    axs[0].set_ylabel("Y", fontsize=16)
    for ax in axs:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.savefig(
        f"{folder}/positions_xy_location_heatmap.png")
    plt.close()



    kde_gt_flat_agents = kde_agents_gt.ravel()+ EPS
    kde_agents_flat = kde_agents.ravel()+ EPS
    kde_gt_flat_agents = kde_gt_flat_agents / kde_gt_flat_agents.sum()
    kde_agents_flat = kde_agents_flat / kde_agents_flat.sum()

    js_div_agents = jensenshannon(kde_gt_flat_agents, kde_agents_flat)
    cos_sim_agents = cosine_similarity([kde_gt_flat_agents], [kde_agents_flat])[0, 0]
    l2_dist_agents = np.linalg.norm(kde_gt_flat_agents - kde_agents_flat)
    print("js_div_agents GT", js_div_agents)


    xx_sim, yy_sim, kde_sim = compute_kde_2d(all_centroids,  bounds=[[x_min, x_max], [y_min, y_max]])

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(kde_gt, ax=axs[1], cmap="Reds", cbar=True, xticklabels=False,  yticklabels=False)
    axs[1].invert_yaxis()
    axs[1].set_title("KDE Heatmap - Ground Truth Centroids")
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels(x_labels)
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels(y_labels)
    axs[1].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    sns.heatmap(kde_sim, ax=axs[0], cmap="Blues", cbar=True, xticklabels=False,  yticklabels=False)
    axs[0].invert_yaxis()
    axs[0].set_title("KDE Heatmap - Simulated Centroids")
    axs[0].set_xticks(x_ticks)
    axs[0].set_xticklabels(x_labels)
    axs[0].set_yticks(y_ticks)
    axs[0].set_yticklabels(y_labels)
    axs[0].set_xlabel("X (m)")
    plt.tight_layout()
    plt.savefig(
        f"{folder}/centroids_xy_location_heatmap.png")
    plt.close()

    kde_gt_flat = kde_gt.ravel() + EPS
    kde_sim_flat = kde_sim.ravel() + EPS
    kde_gt_flat = kde_gt_flat / kde_gt_flat.sum()
    kde_sim_flat = kde_sim_flat / kde_sim_flat.sum()

    js_div = jensenshannon(kde_gt_flat, kde_sim_flat)
    cos_sim = cosine_similarity([kde_gt_flat], [kde_sim_flat])[0, 0]
    l2_dist = np.linalg.norm(kde_gt_flat - kde_sim_flat)
    print("js_div centroids GT", js_div)

    kde_gt_flat_vel = vel_heatmap_gt.ravel()
    kde_sim_flat_vel = vel_heatmap.ravel()


    js_div_vel = jensenshannon(kde_gt_flat_vel, kde_sim_flat_vel)
    cos_sim_vel = cosine_similarity([kde_gt_flat_vel], [kde_sim_flat_vel])[0, 0]
    l2_dist_vel = np.linalg.norm(kde_gt_flat_vel - kde_sim_flat_vel)
    # print("js_div_vel", js_div_vel, "cos_sim_vel", cos_sim_vel, "l2_dist_vel", l2_dist_vel)

    kde_gt_flat_acc = acc_heatmap_gt.ravel()
    kde_sim_flat_acc = acc_heatmap.ravel()
    js_div_acc = jensenshannon(kde_gt_flat_acc, kde_sim_flat_acc)
    cos_sim_acc = cosine_similarity([kde_gt_flat_acc], [kde_sim_flat_acc])[0, 0]
    l2_dist_acc = np.linalg.norm(kde_gt_flat_acc - kde_sim_flat_acc)
    # print("js_div_acc", js_div_acc, "cos_sim_acc", cos_sim_acc, "l2_dist_acc", l2_dist_acc)



    with open(f"{folder}/metrics_with_GT.txt", "w") as f:
        f.write(f"js_div: {js_div}\n")
        f.write(f"cos_sim: {cos_sim}\n")
        f.write(f"l2_dist: {l2_dist}\n")
        f.write(f"js_div_vel: {js_div_vel}\n")
        f.write(f"cos_sim_vel: {cos_sim_vel}\n")
        f.write(f"l2_dist_vel: {l2_dist_vel}\n")
        f.write(f"js_div_acc: {js_div_acc}\n")
        f.write(f"cos_sim_acc: {cos_sim_acc}\n")
        f.write(f"l2_dist_acc: {l2_dist_acc}\n")
        f.write(f"js_div_agents: {js_div_agents}\n")
        f.write(f"cos_sim_agents: {cos_sim_agents}\n")
        f.write(f"l2_dist_agents: {l2_dist_agents}\n")

    if args.dataset == 'nba':
        metrics = [
            "Avg Time in Block (%)",
            "Avg Stay Per Entry (s)",
            "Velocity Mean (All Sims All agents)",
            "Acceleration Mean (All Sims All agents)",
            "location Mean (All Sims All agents)",
            "Velocity Mean (All Sims only Ball)",
            "Acceleration Mean (All Sims only Ball)"
        ]

        for metric in metrics:
            if metric in df_gt.columns and metric in df.columns:
                print("metric: ", metric)
                gt_values = df_gt[metric].values
                sim_values = df[metric].values
                # if metric in ["Avg Time in Block (%)", "Avg Stay Per Entry (s)"]:
                #     print("Metric , " ,metric)
                #     print("GT", gt_values)
                #     print("Sim", sim_values)

                distance = wasserstein_distance(gt_values, sim_values)
                df[f"Wasserstein Distance GT ({metric})"] = distance
                # print(f"Wasserstein Distance for GT {metric}: {distance:.4f}")
                kl_divergence = compute_kl_divergence(sim_values, gt_values)
                df[f"KL Divergence GT ({metric})"] = kl_divergence
                print(f"KL Divergence for GT {metric}: {kl_divergence:.4f}")
                statistic, p_value = ks_2samp(sim_values, gt_values)
                # df[f"KS Statistic ({metric})"] = statistic
                # df[f"KS p-value ({metric})"] = p_value
                # print(f"KS Statistic for {metric}: {statistic:.4f}, p-value: {p_value:.4f}")
            else:
                print(f"Metric '{metric}' not found in ground truth or/and simulated data.")
        if gen_gt_type:
            for metric in metrics:
                if metric in gen_df_gt.columns and metric in df.columns:
                    print("metric: ", metric)
                    gt_values = gen_df_gt[metric].values
                    sim_values = df[metric].values
                    distance = wasserstein_distance(gt_values, sim_values)
                    df[f"Wasserstein Distance {gen_gt_type} ({metric})"] = distance
                    # print(f"Wasserstein Distance for {gen_gt_type} {metric}: {distance:.4f}")
                    kl_divergence = compute_kl_divergence(sim_values, gt_values)
                    df[f"KL Divergence {gen_gt_type} ({metric})"] = kl_divergence
                    print(f"KL Divergence for {gen_gt_type} {metric}: {kl_divergence:.4f}")
                    statistic, p_value = ks_2samp(sim_values, gt_values)
                    # df[f"KS Statistic ({metric})"] = statistic
                    # df[f"KS p-value ({metric})"] = p_value
                    # print(f"KS Statistic for {metric}: {statistic:.4f}, p-value: {p_value:.4f}")
                else:
                    print(f"Metric '{metric}' not found in ground truth or/and simulated data.")


    if gen_gt_type:
        kde_vel_gen = gaussian_kde(xy_vals_gen, weights=vel_vals_gen)
        kde_acc_gen = gaussian_kde(xy_vals_gen, weights=acc_vals_gen)

        vel_heatmap_gen = kde_vel_gen(grid_points).reshape(grid_x.shape)
        acc_heatmap_gen = kde_acc_gen(grid_points).reshape(grid_x.shape)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(vel_heatmap, ax=axs[0], cmap="Blues", cbar=True, xticklabels=False,  yticklabels=False)
        axs[0].invert_yaxis()
        axs[0].set_title("Velocity KDE Heatmap - Simulated")
        axs[0].set_xticks(x_ticks)
        axs[0].set_xticklabels(x_labels)
        axs[0].set_yticks(y_ticks)
        axs[0].set_yticklabels(y_labels)
        axs[0].set_xlabel("X (m)")
        axs[0].set_ylabel("Y (m)")
        sns.heatmap(vel_heatmap_gen, ax=axs[1], cmap="Reds", cbar=True, xticklabels=False,  yticklabels=False)
        axs[1].invert_yaxis()
        axs[1].set_title("Velocity KDE Heatmap - New Ground Truth")
        axs[1].set_xticks(x_ticks)
        axs[1].set_xticklabels(x_labels)
        axs[1].set_yticks(y_ticks)
        axs[1].set_yticklabels(y_labels)
        axs[1].set_xlabel("X (m)")
        plt.tight_layout()
        plt.savefig(
            f"{folder}/velocity_with_gen_heatmap.png")
        plt.close()

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(acc_heatmap, ax=axs[0], cmap="Blues", cbar=True, xticklabels=False,  yticklabels=False)
        axs[0].invert_yaxis()
        axs[0].set_title("Acceleration KDE Heatmap - Simulated")
        axs[0].set_xticks(x_ticks)
        axs[0].set_xticklabels(x_labels)
        axs[0].set_yticks(y_ticks)
        axs[0].set_yticklabels(y_labels)
        axs[0].set_xlabel("X (m)")
        axs[0].set_ylabel("Y (m)")
        sns.heatmap(acc_heatmap_gen, ax=axs[1], cmap="Reds", cbar=True, xticklabels=False,  yticklabels=False)
        axs[1].invert_yaxis()
        axs[1].set_title("Acceleration KDE Heatmap - New Ground Truth")
        axs[1].set_xticks(x_ticks)
        axs[1].set_xticklabels(x_labels)
        axs[1].set_yticks(y_ticks)
        axs[1].set_yticklabels(y_labels)
        axs[1].set_xlabel("X (m)")
        plt.tight_layout()
        plt.savefig(
            f"{folder}/acc_with_gen_heatmap.png")
        plt.close()

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(kde_gen_gt, ax=axs[1], cmap="Reds", cbar=True, xticklabels=False,  yticklabels=False)
        axs[1].invert_yaxis()
        axs[1].set_title("KDE Heatmap - New Ground Truth Centroids")
        axs[1].set_xticks(x_ticks)
        axs[1].set_xticklabels(x_labels)
        axs[1].set_yticks(y_ticks)
        axs[1].set_yticklabels(y_labels)
        axs[1].set_xlabel("X (m)")
        axs[0].set_ylabel("Y (m)")
        sns.heatmap(kde_sim, ax=axs[0], cmap="Blues", cbar=True, xticklabels=False,  yticklabels=False)
        axs[0].invert_yaxis()
        axs[0].set_title("KDE Heatmap - Simulated Centroids")
        axs[0].set_xticks(x_ticks)
        axs[0].set_xticklabels(x_labels)
        axs[0].set_yticks(y_ticks)
        axs[0].set_yticklabels(y_labels)
        axs[0].set_xlabel("X (m)")
        plt.tight_layout()
        plt.savefig(
            f"{folder}/centroids_xy_location_heatmap_gen.png")
        plt.close()

        kde_gen_flat = kde_gen_gt.ravel() + EPS
        kde_gen_flat = kde_gen_flat / kde_gen_flat.sum()

        js_div_gen = jensenshannon(kde_gen_flat, kde_sim_flat)
        cos_sim_gen = cosine_similarity([kde_gen_flat], [kde_sim_flat])[0, 0]
        l2_dist_gen = np.linalg.norm(kde_gen_flat - kde_sim_flat)
        print("js_div centroids GEN", js_div_gen)

        kde_vel_gen_heatmap = kde_vel_gen(grid_points).reshape(grid_x.shape)
        kde_gen_flat_vel = kde_vel_gen_heatmap.ravel()
        js_div_vel_gen = jensenshannon(kde_gen_flat_vel, kde_sim_flat_vel)
        cos_sim_vel_gen = cosine_similarity([kde_gen_flat_vel], [kde_sim_flat_vel])[0, 0]
        l2_dist_vel_gen = np.linalg.norm(kde_gen_flat_vel - kde_sim_flat_vel)
        # print("js_div_vel_gen", js_div_vel_gen, "cos_sim_vel_gen", cos_sim_vel_gen, "l2_dist_vel_gen", l2_dist_vel_gen)

        kde_acc_gen_heatmap = kde_acc_gen(grid_points).reshape(grid_x.shape)
        kde_gen_flat_acc = kde_acc_gen_heatmap.ravel()
        js_div_acc_gen = jensenshannon(kde_gen_flat_acc, kde_sim_flat_acc)
        cos_sim_acc_gen = cosine_similarity([kde_gen_flat_acc], [kde_sim_flat_acc])[0, 0]
        l2_dist_acc_gen = np.linalg.norm(kde_gen_flat_acc - kde_sim_flat_acc)
        # print("js_div_acc_gen", js_div_acc_gen, "cos_sim_acc_gen", cos_sim_acc_gen, "l2_dist_acc_gen", l2_dist_acc_gen)


        all_agent_positions_gt_gen = np.concatenate([sim.reshape(-1, 2) for sim in all_trajectories_gt_gen],axis=0)
        all_agent_positions_gt_gen = np.array(all_agent_positions_gt_gen)

        all_agent_positions_gt_gen = all_agent_positions_gt_gen.astype(np.float64)
        xx_agents_gt_gen, yy_agents_gt_gen, kde_agents_gt_gen = compute_kde_2d(all_agent_positions_gt_gen,bounds=[[x_min, x_max], [y_min, y_max]])

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(kde_agents_gt_gen, ax=axs[1], cmap="Reds", cbar=True, xticklabels=False,  yticklabels=False)
        axs[1].invert_yaxis()
        axs[1].set_title("KDE Heatmap - New Ground Truth Positions")
        axs[1].set_xticks(x_ticks)
        axs[1].set_xticklabels(x_labels)
        axs[1].set_yticks(y_ticks)
        axs[1].set_yticklabels(y_labels)
        axs[1].set_xlabel("X (m)")
        axs[0].set_ylabel("Y (m)")
        sns.heatmap(kde_agents, ax=axs[0], cmap="Blues", cbar=True, xticklabels=False,  yticklabels=False)
        axs[0].invert_yaxis()
        axs[0].set_title("KDE Heatmap - Simulated Positions")
        axs[0].set_xticks(x_ticks)
        axs[0].set_xticklabels(x_labels)
        axs[0].set_yticks(y_ticks)
        axs[0].set_yticklabels(y_labels)
        axs[0].set_xlabel("X (m)")
        plt.tight_layout()
        plt.savefig(
            f"{folder}/positions_gen_xy_location_heatmap.png")
        plt.close()

        kde_gt_flat_agents_gen = kde_agents_gt_gen.ravel() + EPS
        kde_gt_flat_agents_gen = kde_gt_flat_agents_gen / kde_gt_flat_agents_gen.sum()

        js_div_agents_gen = jensenshannon(kde_gt_flat_agents_gen, kde_agents_flat)
        cos_sim_agents_gen = cosine_similarity([kde_gt_flat_agents_gen], [kde_agents_flat])[0, 0]
        l2_dist_agents_gen = np.linalg.norm(kde_gt_flat_agents_gen - kde_agents_flat)
        print("js_div_agents Gen",js_div_agents_gen )

        with open(f"{folder}/metrics_gen.txt", "w") as f:
            f.write(f"js_div: {js_div_gen}\n")
            f.write(f"cos_sim: {cos_sim_gen}\n")
            f.write(f"l2_dist: {l2_dist_gen}\n")
            f.write(f"js_div_vel: {js_div_vel_gen}\n")
            f.write(f"cos_sim_vel: {cos_sim_vel_gen}\n")
            f.write(f"l2_dist_vel: {l2_dist_vel_gen}\n")
            f.write(f"js_div_acc: {js_div_acc_gen}\n")
            f.write(f"cos_sim_acc: {cos_sim_acc_gen}\n")
            f.write(f"l2_dist_acc: {l2_dist_acc_gen}\n")
            f.write(f"js_div_agents_gen: {js_div_agents_gen}\n")
            f.write(f"cos_sim_agents_gen: {cos_sim_agents_gen}\n")
            f.write(f"l2_dist_agents_gen: {l2_dist_agents_gen}\n")

    np.savez_compressed(
        f"{folder}/GT_logged_data.npz",
        centroids=np.array(all_centroids, dtype=object),
        trajectories=np.array(all_trajectories, dtype=object),
        xy_vals=np.array(xy_values),
        vel_vals=np.array(vel_values),
        acc_vals=np.array(acc_values)
    )

    if args.dataset == "nba":
        return df
    return None

def denormalize_traj(traj_norm, prediction = ''):
    traj_denorm = traj_norm.copy() if isinstance(traj_norm, np.ndarray) else traj_norm.clone()
    if prediction =='prediction':# (Total_samples, N, 10, 2, 20)
        traj_denorm[:, :, :, 0, :] = (traj_denorm[:, :, :, 0, :] / 100.0) * (1951.0 - 9.0) + 9.0
        traj_denorm[:, :, :, 1, :] = (traj_denorm[:, :, :, 1, :] / 100.0) * (1973.0 - 7.0) + 7.0
    else:
        traj_denorm[..., 0] = (traj_denorm[..., 0] / 100.0) * (1951.0 - 9.0) + 9.0
        traj_denorm[..., 1] = (traj_denorm[..., 1] / 100.0) * (1973.0 - 7.0) + 7.0
    return traj_denorm

def compute_kl_divergence(p, q):
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p /= p.sum()
    q /= q.sum()
    return np.sum(rel_entr(p, q))

class ConstantNBA:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 6
    X_MIN = 0
    X_MAX = 29.9
    Y_MIN = 0
    Y_MAX = 16.27
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = X_MAX / 2 - DIFF / 1.5 + 0.10
    Y_CENTER = Y_MAX - DIFF / 1.5 - 0.35
    n_agents = 11
    field = 29.9
    field_width = 16.27
    num_blocks = 10
    buffer = 2
    buffer_train = 2
    H_inv =None
    court = "datasets/nba/court.png"

    MESSAGE = 'You can rerun the script and choose any event from 0 to '

    gt_path = "GT/stats/nba/test_/centroids_agents.csv"
    groupnet_gt_path = "G1/stats/nba/test/missions12_length120s_sim100_buf2_1_1_/centroids_agents.csv"
    sampler_gt_path = "Sampler/stats/nba/False/test/missions12_length120s_sim100_buf2_0_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/nba/test_/GT_logged_data.npz'
    groupnet_gt_path_data = "G1/stats/nba/test/missions12_length120s_sim100_buf2_1_1_/GT_logged_data.npz"
    sampler_gt_path_data = "Sampler/stats/nba/False/test/missions12_length120s_sim100_buf2_0_0_/GT_logged_data.npz"

class ConstantSYNR:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    X_MIN = -8.74
    X_MAX = 54.3
    Y_MIN = -8.75
    Y_MAX = 53.8
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = 50
    Y_CENTER = 50
    n_agents = 6
    field = 54.3
    field_width = 53.8
    num_blocks = 10
    buffer = 5
    buffer_train = 5
    court = "datasets/target/tank2.png"
    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/syn/test_rigid_/centroids_agents.csv"
    groupnet_gt_path = "G1/stats/syn/test_rigid/missions5_length120s_sim100_buf5_1_/centroids_agents.csv"
    sampler_gt_path = "Sampler/stats/syn/False/test_rigid/missions5_length120s_sim100_buf5_0_/centroids_agents.csv"
    H_inv =None

    gt_path_data = 'GT/stats/syn/test_rigid_/GT_logged_data.npz'
    groupnet_gt_path_data = "G1/stats/syn/test_rigid/missions5_length120s_sim100_buf5_1_/GT_logged_data.npz"
    sampler_gt_path_data = "Sampler/stats/syn/False/test_rigid/missions5_length120s_sim100_buf5_0_/GT_logged_data.npz"

class ConstantSYNS:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    X_MIN = -6.08
    X_MAX = 57.6
    Y_MIN = -6.00
    Y_MAX =  50.99
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = 50
    Y_CENTER = 50
    field = 57.6
    field_width =  50.99
    num_blocks = 10
    buffer = 4.82
    buffer_train = 4.82
    H_inv =None
    court = "datasets/target/tank2.png"
    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/syn/test_smooth_/centroids_agents.csv"
    groupnet_gt_path ="G1/stats/syn/test_smooth/missions5_length120s_sim100_buf4.82_1_/centroids_agents.csv"
    sampler_gt_path = "Sampler/stats/syn/False/test_smooth/missions5_length120s_sim100_buf4.82_0_/centroids_agents.csv"
    n_agents = 6
    gt_path_data = 'GT/stats/syn/test_smooth_/GT_logged_data.npz'
    groupnet_gt_path_data = "G1/stats/syn/test_smooth/missions5_length120s_sim100_buf4.82_1_/GT_logged_data.npz"
    sampler_gt_path_data = "Sampler/stats/syn/False/test_smooth/missions5_length120s_sim100_buf4.82_0_/GT_logged_data.npz"

class ConstantFish:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    X_MIN = 0.0399
    X_MAX = 99.9
    Y_MIN = 0.056
    Y_MAX = 99.89
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = 30
    Y_CENTER = 30
    n_agents = 8
    field = 99.9
    field_width = 99.89
    num_blocks = 10
    buffer = 8
    buffer_train = 8
    H_inv =None

    court = "datasets/target/tank2.png"
    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/target/train1_train3_/centroids_agents.csv"
    groupnet_gt_path = "G1/stats/target/test/missions5_length120s_sim100_buf8_1_/centroids_agents.csv"
    sampler_gt_path = "Sampler/stats/target/False/test/missions5_length120s_sim100_buf8_0_/centroids_agents.csv"
    gt_path_data = 'GT/stats/target/train1_train3_/GT_logged_data.npz'
    groupnet_gt_path_data = "G1/stats/target/test/missions5_length120s_sim100_buf8_1_/GT_logged_data.npz"
    sampler_gt_path_data = "Sampler/stats/target/False/test/missions5_length120s_sim100_buf8_0_/GT_logged_data.npz"

class ConstantETH:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    X_MIN = -5.49
    X_MAX = 13.89
    Y_MIN = -0.44
    Y_MAX = 8.84
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = 10
    Y_CENTER = 10
    n_agents = 2
    field = 13.89
    field_width = 8.84
    num_blocks = 10
    buffer = 1
    buffer_train = 1
    court = "datasets/target/tank2.png"

    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/eth/test_seq_eth_None_2_/centroids_agents.csv"
    groupnet_gt_path = "G1/stats/eth/seq_eth_None_2/missions5_length120s_sim100_buf1_1_/centroids_agents.csv"
    sampler_gt_path = "Sampler/stats/eth/False/seq_eth_None_2/missions5_length120s_sim100_buf1_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/eth/test_seq_eth_None_2_/GT_logged_data.npz'
    groupnet_gt_path_data = "G1/stats/eth/seq_eth_None_2/missions5_length120s_sim100_buf1_1_/GT_logged_data.npz"
    sampler_gt_path_data = "Sampler/stats/eth/False/seq_eth_None_2/missions5_length120s_sim100_buf1_0_/GT_logged_data.npz"

class ConstantHotel:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    X_MIN = -1.4600
    X_MAX = 4.0400
    Y_MIN = -10.1900
    Y_MAX = 4.3100
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = 10
    Y_CENTER = 10
    n_agents = 4
    field = 4.0400
    field_width = 4.3100
    num_blocks = 10
    buffer = 1
    buffer_train = 1
    court = "datasets/target/tank2.png"


    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stat/eth/test_seq_hotel_None_4_/centroids_agents.csv"
    groupnet_gt_path = "G1/stats/eth/seq_hotel_None_4/missions5_length120s_sim100_buf1_1_/centroids_agents.csv"
    sampler_gt_path = "Sampler/stats/eth/False/seq_hotel_None_4/missions5_length120s_sim100_buf1_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/eth/test_seq_hotel_None_4_/GT_logged_data.npz'
    groupnet_gt_path_data = "G1/stats/eth/seq_hotel_None_4/missions5_length120s_sim100_buf1_1_/GT_logged_data.npz"
    sampler_gt_path_data = "Sampler/stats/eth/False/seq_hotel_None_4/missions5_length120s_sim100_buf1_0_/GT_logged_data.npz"

class ConstantUni:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    X_MIN = -0.4620
    X_MAX = 15.4692
    Y_MIN =-0.3184
    Y_MAX =13.8919
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = 10
    Y_CENTER = 10
    n_agents = 8
    field = 25
    field_width = 25
    num_blocks = 10
    buffer = 1.12
    buffer_train = 1


    court = f"datasets/eth/raw/uni/map.png"
    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/eth/test_uni_8_/centroids_agents.csv"
    groupnet_gt_path = "G1/stats/eth/uni_None_8/missions5_length120s_sim100_buf1.12_1_/centroids_agents.csv"
    sampler_gt_path = "Sampler/stats/eth/False/uni_None_8/missions5_length120s_sim100_buf1.12_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/eth/test_uni_8_/GT_logged_data.npz'
    groupnet_gt_path_data = "G1/stats/eth/uni_None_8/missions5_length120s_sim100_buf1.12_1_/GT_logged_data.npz"
    sampler_gt_path_data = "Sampler/stats/eth/False/uni_None_8/missions5_length120s_sim100_buf1.12_0_/GT_logged_data.npz"

class ConstantZara1:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    X_MIN = -0.1254
    X_MAX =  15.4332
    Y_MIN = 0
    Y_MAX = 10.9373
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = 10
    Y_CENTER = 10
    n_agents = 5
    field = 25
    field_width = 25
    num_blocks = 10
    buffer = 1
    buffer_train = 1


    court = f"datasets/eth/raw/zara_01/map.png"
    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/eth/test_zara_01_None_5_/centroids_agents.csv"
    groupnet_gt_path = "G1/stats/eth/zara_01_None_5/missions5_length120s_sim100_buf1_1_/centroids_agents.csv"
    sampler_gt_path = "Sampler/stats/eth/False/zara_01_None_5/missions5_length120s_sim100_buf1_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/eth/test_zara_01_None_5_/GT_logged_data.npz'
    groupnet_gt_path_data = "G1/stats/eth/zara_01_None_5/missions5_length120s_sim100_buf1_1_/GT_logged_data.npz"
    sampler_gt_path_data ="Sampler/stats/eth/False/zara_01_None_5/missions5_length120s_sim100_buf1_0_/GT_logged_data.npz"

class ConstantZara2:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    X_MIN = -0.3578
    X_MAX = 15.5584
    Y_MIN = -0.0654
    Y_MAX = 13.4797
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = 10
    Y_CENTER = 10
    n_agents = 8
    field = 25
    buffer = 1.12
    field_width = 25
    num_blocks = 10
    buffer_train = 1


    court = f"datasets/eth/raw/zara_02/map.png"
    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/eth/test_zara_02_None_8_/centroids_agents.csv"
    groupnet_gt_path = "G1/stats/eth/zara_02_None_8/missions5_length120s_sim100_buf1.12_1_/centroids_agents.csv"
    sampler_gt_path = "Sampler/stats/eth/False/zara_02_None_8/missions5_length120s_sim100_buf1.12_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/eth/test_zara_02_None_8_/GT_logged_data.npz'
    groupnet_gt_path_data = "G1/stats/eth/zara_02_None_8/missions5_length120s_sim100_buf1.12_1_/GT_logged_data.npz"
    sampler_gt_path_data = "Sampler/stats/eth/False/zara_02_None_8/missions5_length120s_sim100_buf1.12_0_/GT_logged_data.npz"


def return_the_eth_scene(scene):
    if scene == 'seq_eth':
        return ConstantETH
    elif scene == 'seq_hotel':
        return ConstantHotel
    elif scene == 'uni':
        return ConstantUni
    elif scene == 'zara_01':
        return ConstantZara1
    elif scene == 'zara_02':
        return ConstantZara2




class ConstantSDD0:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    # X_MIN = 11.5000
    # X_MAX = 1390
    # Y_MIN = 20
    # Y_MAX = 1068.5000
    # COL_WIDTH = 0.3
    # SCALE = 1.65
    # FONTSIZE = 6
    # X_CENTER = 1000
    # Y_CENTER = 1000
    n_agents = 4
    field_T = 1390
    field_width_T = 1068.5000
    num_blocks = 10
    # buffer = 96.8
    # buffer_train = 115

    court = "datasets/target/tank2.png"
    H_inv =None


    X_MIN = 0.0399
    X_MAX = 99.9
    Y_MIN = 0.056
    Y_MAX = 99.89
    X_CENTER = 30
    Y_CENTER = 30
    field = 99.9
    field_width = 99.89
    buffer = 8
    buffer_train = 8

    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/sdd/test__0_4_/centroids_agents.csv"
    groupnet_gt_path =  'G1/stats/sdd/_0_4/missions5_length120s_sim100_buf96.8_1_/centroids_agents.csv'
    sampler_gt_path = "Sampler/stats/sdd/False/_0_4/missions5_length120s_sim100_buf96.8_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/sdd/test__0_4_/GT_logged_data.npz'
    groupnet_gt_path_data =  'G1/stats/sdd/_0_4/missions5_length120s_sim100_buf96.8_1_/GT_logged_data.npz'
    sampler_gt_path_data = "Sampler/stats/sdd/False/_0_4/missions5_length120s_sim100_buf96.8_0_/GT_logged_data.npz"

class ConstantSDD1:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    # X_MIN = 26
    # X_MAX = 1951
    # Y_MIN = 70
    # Y_MAX = 1050.5000
    # COL_WIDTH = 0.3
    # SCALE = 1.65
    # FONTSIZE = 6
    # X_CENTER = 1000
    # Y_CENTER = 1000
    n_agents = 4
    field_T = 1951
    field_width_T = 1050.5000
    num_blocks = 10
    # buffer = 115
    # buffer_train = 115

    X_MIN = 0.0399
    X_MAX = 99.9
    Y_MIN = 0.056
    Y_MAX = 99.89
    X_CENTER = 30
    Y_CENTER = 30
    field = 99.9
    field_width = 99.89
    buffer = 8
    buffer_train = 8

    court = "datasets/target/tank2.png"
    H_inv =None

    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/sdd/test__1_4_/centroids_agents.csv"
    groupnet_gt_path = 'G1/stats/sdd/_1_4/missions5_length120s_sim100_buf115_1_/centroids_agents.csv'
    sampler_gt_path = "Sampler/stats/sdd/False/_1_4/missions5_length120s_sim100_buf115_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/sdd/test__1_4_/GT_logged_data.npz'
    groupnet_gt_path_data = 'G1/stats/sdd/_1_4/missions5_length120s_sim100_buf115_1_/GT_logged_data.npz'
    sampler_gt_path_data = "Sampler/stats/sdd/False/_1_4/missions5_length120s_sim100_buf115_0_/GT_logged_data.npz"

class ConstantSDD2:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    # DIFF = 0
    # X_MIN = 11.5000
    # X_MAX = 1610.5000
    # Y_MIN = 7.5000
    # Y_MAX = 1915
    # COL_WIDTH = 0.3
    # SCALE = 1.65
    # FONTSIZE = 6
    # X_CENTER = 1000
    # Y_CENTER = 1000
    n_agents = 8
    field_T = 1610.5000
    field_width_T = 1915
    num_blocks = 10
    # buffer = 140
    # buffer_train = 115
    X_MIN = 0.0399
    X_MAX = 99.9
    Y_MIN = 0.056
    Y_MAX = 99.89
    X_CENTER = 30
    Y_CENTER = 30
    field = 99.9
    field_width = 99.89
    buffer = 8
    buffer_train = 8
    court = f"datasets/sdd/raw/deathcircle/reference.jpg"
    H_inv =None

    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/sdd/test__2_8_/centroids_agents.csv"
    groupnet_gt_path = 'G1/stats/sdd/_2_8/missions5_length120s_sim100_buf140_1_/centroids_agents.csv'
    sampler_gt_path = "Sampler/stats/sdd/False/_2_8/missions5_length120s_sim100_buf140_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/sdd/test__2_8_/GT_logged_data.npz'
    groupnet_gt_path_data = 'G1/stats/sdd/_2_8/missions5_length120s_sim100_buf140_1_/GT_logged_data.npz'
    sampler_gt_path_data = "Sampler/stats/sdd/False/_2_8/missions5_length120s_sim100_buf140_0_/GT_logged_data.npz"

class ConstantSDD3:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    # INTERVAL = 10
    # DIFF = 0
    # X_MIN = 13
    # X_MAX = 1417
    # Y_MIN = 16.5
    # Y_MAX = 1973
    # COL_WIDTH = 0.3
    # SCALE = 1.65
    # FONTSIZE = 6
    # X_CENTER = 1000
    # Y_CENTER = 1000
    n_agents = 3
    field_T = 1417
    field_width_T = 1973
    num_blocks = 10
    # buffer = 134.4
    # buffer_train = 115
    X_MIN = 0.0399
    X_MAX = 99.9
    Y_MIN = 0.056
    Y_MAX = 99.89
    X_CENTER = 30
    Y_CENTER = 30
    field = 99.9
    field_width = 99.89
    buffer = 8
    buffer_train = 8

    court = "datasets/target/tank2.png"
    H_inv =None

    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/sdd/test__3_3_/centroids_agents.csv"
    groupnet_gt_path = 'G1/stats/sdd/_3_3/missions5_length120s_sim100_buf134.4_1_/centroids_agents.csv'
    sampler_gt_path = "Sampler/stats/sdd/False/_3_3/missions5_length120s_sim100_buf134.4_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/sdd/test__3_3_/GT_logged_data.npz'
    groupnet_gt_path_data = 'G1/stats/sdd/_3_3/missions5_length120s_sim100_buf134.4_1_/GT_logged_data.npz'
    sampler_gt_path_data = "Sampler/stats/sdd/False/_3_3/missions5_length120s_sim100_buf134.4_0_/GT_logged_data.npz"

class ConstantSDD4:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    # X_MIN = 19
    # X_MAX = 1409
    # Y_MIN = 29
    # Y_MAX = 1886
    # COL_WIDTH = 0.3
    # SCALE = 1.65
    # FONTSIZE = 6
    # X_CENTER = 1000
    # Y_CENTER = 1000
    n_agents = 8
    field_T = 1409
    field_width_T = 1886
    # buffer = 129.2
    num_blocks = 10
    # buffer_train = 115

    X_MIN = 0.0399
    X_MAX = 99.9
    Y_MIN = 0.056
    Y_MAX = 99.89
    X_CENTER = 30
    Y_CENTER = 30
    field = 99.9
    field_width = 99.89
    buffer = 8
    buffer_train = 8

    court = "datasets/target/tank2.png"
    H_inv =None

    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/sdd/test__4_8_/centroids_agents.csv"
    groupnet_gt_path = 'G1/stats/sdd/_4_8/missions5_length120s_sim100_buf129.2_1_/centroids_agents.csv'
    sampler_gt_path = "Sampler/stats/sdd/False/_4_8/missions5_length120s_sim100_buf129.2_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/sdd/test__4_8_/GT_logged_data.npz'
    groupnet_gt_path_data = 'G1/stats/sdd/_4_8/missions5_length120s_sim100_buf129.2_1_/GT_logged_data.npz'
    sampler_gt_path_data = "Sampler/stats/sdd/False/_4_8/missions5_length120s_sim100_buf129.2_0_/GT_logged_data.npz"

class ConstantSDD5:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    # X_MIN = 15.5
    # X_MAX = 1398
    # Y_MIN = 26
    # Y_MAX = 1959.5
    # COL_WIDTH = 0.3
    # SCALE = 1.65
    # FONTSIZE = 6
    # X_CENTER = 1000
    # Y_CENTER = 1000
    n_agents = 3
    field_T = 1398
    field_width_T = 1959.5
    num_blocks = 10
    # buffer  = 133
    # buffer_train = 115
    X_MIN = 0.0399
    X_MAX = 99.9
    Y_MIN = 0.056
    Y_MAX = 99.89
    X_CENTER = 30
    Y_CENTER = 30
    field = 99.9
    field_width = 99.89
    buffer = 8
    buffer_train = 8

    court = "datasets/target/tank2.png"
    H_inv =None

    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/sdd/test__5_3_/centroids_agents.csv"
    groupnet_gt_path = 'G1/stats/sdd/_5_3/missions5_length120s_sim100_buf133_1_/centroids_agents.csv'
    sampler_gt_path = "Sampler/stats/sdd/False/_5_3/missions5_length120s_sim100_buf133_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/sdd/test__5_3_/GT_logged_data.npz'
    groupnet_gt_path_data = 'G1/stats/sdd/_5_3/missions5_length120s_sim100_buf133_1_/GT_logged_data.npz'
    sampler_gt_path_data = "Sampler/stats/sdd/False/_5_3/missions5_length120s_sim100_buf133_0_/GT_logged_data.npz"

class ConstantSDD6:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    # X_MIN = 9.5000
    # X_MAX = 1393
    # Y_MIN = 18
    # Y_MAX = 1928.5000
    # COL_WIDTH = 0.3
    # SCALE = 1.65
    # FONTSIZE = 6
    # X_CENTER = 1000
    # Y_CENTER = 1000
    n_agents = 4
    field_T = 1393
    field_width_T = 1928.5000
    num_blocks = 10
    # buffer = 131.76
    # buffer_train = 115

    X_MIN = 0.0399
    X_MAX = 99.9
    Y_MIN = 0.056
    Y_MAX = 99.89
    X_CENTER = 30
    Y_CENTER = 30
    field = 99.9
    field_width = 99.89
    buffer = 8
    buffer_train = 8
    court = "datasets/target/tank2.png"
    H_inv =None

    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/sdd/test__6_4_/centroids_agents.csv"
    groupnet_gt_path ='G1/stats/sdd/_6_4/missions5_length120s_sim100_buf131.76_1_/centroids_agents.csv'
    sampler_gt_path = "Sampler/stats/sdd/False/_6_4/missions5_length120s_sim100_buf131.76_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/test__6_4_/GT_logged_data.npz'
    groupnet_gt_path_data = 'G1/stats/sdd/_6_4/missions5_length120s_sim100_buf131.76_1_/GT_logged_data.npz'
    sampler_gt_path_data = "Sampler/stats/sdd/False/_6_4/missions5_length120s_sim100_buf131.76_0_/GT_logged_data.npz"

class ConstantSDD7:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 0
    # X_MIN = 178.5000
    # X_MAX = 1817
    # Y_MIN = 500.5000
    # Y_MAX = 1055.5000
    # COL_WIDTH = 0.3
    # SCALE = 1.65
    # FONTSIZE = 6
    # X_CENTER = 1000
    # Y_CENTER = 1000
    n_agents = 2
    field_T = 1980
    field_width_T = 1980
    num_blocks = 10
    # buffer = 87.74
    # buffer_train = 115

    X_MIN = 0.0399
    X_MAX = 99.9
    Y_MIN = 0.056
    Y_MAX = 99.89
    X_CENTER = 30
    Y_CENTER = 30
    field = 99.9
    field_width = 99.89
    buffer = 8
    buffer_train = 8

    court = "datasets/target/tank2.png"
    H_inv =None

    MESSAGE = 'You can rerun the script and choose any event from 0 to '
    gt_path = "GT/stats/sdd/test__7_2_/centroids_agents.csv"
    groupnet_gt_path = 'G1/stats/sdd/_7_2/missions5_length120s_sim100_buf87.74_1_/centroids_agents.csv'
    sampler_gt_path = "Sampler/stats/sdd/False/_7_2/missions5_length120s_sim100_buf87.74_0_/centroids_agents.csv"

    gt_path_data = 'GT/stats/sdd/test__7_2_/GT_logged_data.npz'
    groupnet_gt_path_data = 'G1/stats/sdd/_7_2/missions5_length120s_sim100_buf87.74_1_/GT_logged_data.npz'
    sampler_gt_path_data = "Sampler/stats/sdd/False/_7_2/missions5_length120s_sim100_buf87.74_0_/GT_logged_data.npz"

def return_the_sdd_scene(scene):
    if scene == 0:
        return ConstantSDD0
    elif scene == 1:
        return ConstantSDD1
    elif scene == 2:
        return ConstantSDD2

    elif scene == 3:
        return ConstantSDD3
    elif scene == 4:
        return ConstantSDD4

    elif scene == 5:
        return ConstantSDD5
    elif scene == 6:
        return ConstantSDD6
    elif scene == 7:
        return ConstantSDD7



def draw_result(constant, folder, args, future,past,mode='pre', mission_aware = None, GT_num = None, mission = None):
    # b n t 2
    # print('drawing...')
    trajs = np.concatenate((past,future), axis = 2)
    # print("traj",trajs.shape)
    batch = trajs.shape[0]
    for idx in range(batch):
        plt.clf()
        traj = trajs[idx]

        if args.dataset =="nba":
            traj = traj*(94/28)
            ax = plt.axes(xlim=(0,100),ylim=(0,50))

        elif args.dataset == "sdd":
            x_min = trajs[:, :, 0].min()
            x_max = trajs[:, :, 0].max()
            y_min = trajs[:, :, 1].min()
            y_max = trajs[:, :, 1].max()

            ax = plt.axes(xlim=(x_min, x_max), ylim=(y_min, y_max))
        else:
            ax = plt.axes(xlim=(constant.X_MIN,
                                constant.X_MAX),
                          ylim=(constant.Y_MIN,
                                constant.Y_MAX))

        H_inv = constant.H_inv
        actor_num = traj.shape[0]
        length = traj.shape[1]


        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)

        colorteam1 = 'dodgerblue'
        colorteam2 = 'orangered'
        colorball = 'limegreen'
        colorteam1_pre = 'skyblue'
        colorteam2_pre = 'lightsalmon'
        colorball_pre = 'mediumspringgreen'

        for j in range(actor_num):
            if j < 3:
                color = colorteam1
                color_pre = colorteam1_pre
            elif j < 6:
                color = colorteam2
                color_pre = colorteam2_pre
            else:
                color_pre = colorball_pre
                color = colorball

            if mission_aware and j in mission_aware:
                edge_color = 'black'
                linewidth = 0.8
            else:
                edge_color = color
                linewidth = 0.0  # transparent edge

            # if H_inv is not None:
            #     wpts = traj[j]
            #     uv = world2image(wpts, H_inv)  # (T,2) in pixel
            #     x_pix, y_pix = uv[:, 0], uv[:, 1]
            #     x_world = constant.X_MIN + (x_pix / court.shape[1]) * (constant.X_MAX - constant.X_MIN)
            #     y_world = constant.Y_MAX - (y_pix / court.shape[0]) * (constant.Y_MAX - constant.Y_MIN)
            #     traj[j] = np.stack([x_world, y_world], axis=1)

            for i in range(length):
                points = [(traj[j,i,0],traj[j,i,1])]
                (x, y) = zip(*points)
                # plt.scatter(x, y, color=color,s=20,alpha=0.3+i*((1-0.3)/length))
                if i < 5:
                    plt.scatter(x, y, color=color_pre, edgecolors=edge_color, linewidths=linewidth,s=20,alpha=1)
                else:
                    plt.scatter(x, y, color=color, edgecolors=edge_color, linewidths=linewidth,s=20,alpha=1)
            final_pos = traj[j, -1]
            if mission_aware and j in mission_aware:
                plt.text(final_pos[0], final_pos[1]+1, str(j), color='black',
                         fontsize=9, ha='center', va='center', fontweight='bold', zorder=3)

            for i in range(length-1):
                points = [(traj[j,i,0],traj[j,i,1]),(traj[j,i+1,0],traj[j,i+1,1])]
                (x, y) = zip(*points)
                # plt.plot(x, y, color=color,alpha=0.3+i*((1-0.3)/length),linewidth=2)
                if i < 4:
                    plt.plot(x, y, color=color_pre,alpha=0.5,linewidth=2)
                else:
                    plt.plot(x, y, color=color,alpha=1,linewidth=2)

        ax.set_aspect('equal')


        if args.dataset == 'nba':
            court = plt.imread(constant.court)
            plt.imshow(court, zorder=0, extent=[0, 100- constant.DIFF,
                                                0 ,50], alpha=0.5)
        elif args.dataset == "sdd":

            court = plt.imread(constant.court)
            plt.imshow(court, zorder=0, extent=[x_min, x_max,
                                                y_min, y_max], alpha=0.5)
        else:
            court = plt.imread(constant.court)
            plt.imshow(court, zorder=0, extent=[constant.X_MIN, constant.X_MAX - constant.DIFF,
                                                constant.Y_MIN, constant.Y_MAX], alpha=0.5)


        if mission_aware is not None and mission is not None:
            for m_idx, agent_idx in enumerate(mission_aware):
                mission_point = mission[m_idx]  # shape (2,)
                # plt.scatter(mission_point[0], mission_point[1], color='black', marker='x', s=50, linewidths=2, zorder=3)
                plt.text(mission_point[0], mission_point[1], str(agent_idx), color='black',
                         fontsize=10, fontweight='bold', ha='center', va='center', zorder=4)

        data_target = f'{folder}/imgs'
        if not os.path.exists(data_target):
            os.mkdir(data_target)
        if mode == 'pre':
            if mission_aware:
                plt.savefig(f'{folder}/imgs/'+str(GT_num)+'_pre.png')
            else:
                plt.savefig(f'{folder}/imgs/'+str(idx)+'_pre.png')
        else:
            if mission_aware:
                plt.savefig(f'{folder}/imgs/'+str(GT_num)+'_gt.png')
            else:
                plt.savefig(f'{folder}/imgs/' + str(idx) + '_gt.png')
    # print('ok')
    return




def saveModel_mission_Sampler(SM, args, epoch=''):
    output_dir = f'SM/{args.model_dir}/{args.dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    torch.save(SM.state_dict(), "%s/SM_%s_%s_%s_%s_%s_%s_%s_%s.pth" % (output_dir,args.scene,args.sdd_scene,args.training_type,args.test_mlp, epoch,  args.seed,args.how_far, args.info))

def saveModel_GAN_disc(D, args,epoch='', name = "groupnet-gan-disc"):
    if name =="groupnet-gan-disc":
        basename = os.path.basename(args.saved_models_GAN_GM)
        output_dir = f'GANG/{args.model_dir}/{args.dataset}/{args.disc_type}'
    else:
        basename = os.path.basename(args.saved_models_GAN_SM)
        output_dir = f'GANS/{args.model_dir}/{args.dataset}/{args.disc_type}'
    prefix = basename.split('_0.pth')[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    torch.save(D.state_dict(), "%s/D_%s_%s_%s_%s_%s_%s_%s.pth" % (output_dir,args.scene, args.sdd_scene, args.training_type ,epoch,  args.seed,prefix, args.info))



def saveModel_real(GM, D, args,epoch=''):
    numbers = re.findall(r'\d+', args.saved_models_GM)
    far = int(numbers[-1]) if numbers else None
    timestamp = args.timestamp
    output_dir = f'GANG/{args.model_dir}/{args.dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    torch.save(GM.state_dict(), "%s/GM_%s_%s_%s_%s_%s_%s_%s.pth" % (output_dir, args.scene,args.sdd_scene,args.training_type,epoch,  args.seed,args.how_far, args.info ))
    torch.save(D.state_dict(), "%s/D_%s_%s_%s_%s_%s_%s_%s.pth" % (output_dir, args.scene,args.sdd_scene,args.training_type,epoch,  args.seed,args.how_far, args.info))

def saveModel_Sampler_GAN(SM, D, args,epoch=''):
    numbers = re.findall(r'\d+', args.saved_models_GM)
    far = int(numbers[-1]) if numbers else None
    timestamp = args.timestamp
    output_dir = f'GANS/{args.model_dir}/{args.dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    torch.save(SM.state_dict(), "%s/SM_%s_%s_%s_%s_%s_%s_%s.pth" % (output_dir, args.scene,args.sdd_scene,args.training_type,epoch,  args.seed,args.how_far, args.info ))
    torch.save(D.state_dict(), "%s/D_%s_%s_%s_%s_%s_%s_%s.pth" % (output_dir, args.scene,args.sdd_scene,args.training_type,epoch,  args.seed,args.how_far, args.info))


def saveModel_S(S, args,epoch=''):

    output_dir = f'Sampler/{args.model_dir}/{args.dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    torch.save(S.state_dict(), "%s/G_%s_%s_%s_%s_%s_%s_%s.pth" % (output_dir, args.scene,args.sdd_scene ,args.training_type,epoch, args.seed, args.test_mlp, args.info))

def saveModel_D(S, args,epoch=''):

    output_dir = f'{args.model_dir}{args.dataset}/{args.scene}_{args.sdd_scene}_{args.training_type}_{args.info}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    torch.save(S.state_dict(), "%s/D_%s_%s_%s.pth" % (output_dir,epoch, args.seed, args.classifier_method))


def saveModel_DS(S, args,epoch=''):
    output_dir = f'{args.model_dir}Sampler/{args.dataset}/{args.scene}_{args.sdd_scene}_{args.training_type}_{args.info}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    torch.save(S.state_dict(), "%s/D_%s_%s_%s.pth" % (output_dir,epoch, args.seed, args.classifier_method))

def compute_features(args, x,time_step, fov_deg=190.0):
    disp = x[:, :, 1:, :] - x[:, :, :-1, :]  # (B, N, T-1, 2)

    velocity = torch.norm(disp, dim=-1)  # (B, N, T-1)

    # (angle of movement vector)
    direction = torch.atan2(disp[..., 1], disp[..., 0])  # (B, N, T-1)
    #########################NEW#############################################################################
    # direction = torch.stack([direction.cos(),  # (B, N, T-1)
    #                        direction.sin()],  # (B, N, T-1)
    #                       dim=-1)  # now (B, N, T-1, 2)

    B, N, T, _ = x.shape
    fov_rad = math.radians(fov_deg)
    half_fov = fov_rad / 2
    # print("half_fov", half_fov)
    visibility_seq = []
    count_seq = []

    for t in range(1, T):
        pos_prev = x[:, :, t - 1, :]  # (B, N, 2)
        pos_curr = x[:, :, t, :]  # (B, N, 2)

        movement = pos_curr - pos_prev  # (B, N, 2)
        dir_angle = torch.atan2(movement[..., 1], movement[..., 0])  # (B, N)
        agent_pos = pos_curr
        # print("dir_angle", dir_angle)
        # print("movement", movement)
        vis_matrix = torch.zeros((B, N, N), device=x.device)
        count_matrix = torch.zeros((B, N, N), device=x.device)

        for i in range(N):
            viewer_pos = agent_pos[:, i, :].unsqueeze(1)  # (B, 1, 2)
            viewer_angle = dir_angle[:, i].unsqueeze(1)  # (B, 1)

            # print("viewer", viewer_pos, viewer_angle)
            rel_vec = agent_pos - viewer_pos  # (B, N, 2)
            rel_dist = torch.norm(rel_vec, dim=-1)  # (B, N)

            epsilon = 1e-4
            self_mask = torch.zeros((B, N), device=rel_dist.device).bool()
            self_mask[:, i] = True
            zero_dist = rel_dist == 0
            need_fix = zero_dist & ~self_mask
            rel_dist = torch.where(need_fix, torch.full_like(rel_dist, epsilon), rel_dist)

            rel_angle = torch.atan2(rel_vec[..., 1], rel_vec[..., 0])  # (B, N)
            angle_diff = (rel_angle - viewer_angle + math.pi) % (2 * math.pi) - math.pi
            # print("rel_angle", rel_angle)
            # print("rel_dist", rel_dist)
            # print("angle_diff", angle_diff)
            in_fov = (angle_diff.abs() <= half_fov)
            # print("in_fov", in_fov)

            vis_matrix[:, i, :] = torch.where(in_fov, 1.0 / (rel_dist + epsilon), torch.zeros_like(rel_dist))
            count_matrix[:, i, :] = in_fov.float()
            vis_matrix[:, i, i] = 0.0
            count_matrix[:, i, i] = 0.0

        visibility_seq.append(vis_matrix.unsqueeze(1))  # Add time dimension
        count_seq.append(count_matrix.unsqueeze(1))

    if time_step == "past":
        final_dist_with_time = vis_matrix.squeeze(1) #B, N, N (only final matrix)
    final_counter = torch.cat(count_seq, dim=1).sum(dim=1)
    final_dist = torch.cat(visibility_seq, dim=1).sum(dim =1) #B, T-1, N, N ->B, N, N

    # print("final_dist", final_dist[0])
    # print("final_counter", final_counter[0])
    weighted_vis = final_dist * final_counter

    # print("weighted_vis", weighted_vis[0])
    row_max = weighted_vis.sum(dim=-1, keepdim=True)   # (B, N, 1)
    normalized_vis = weighted_vis / (row_max+ epsilon)

    # print("normalized_vis", normalized_vis.shape, normalized_vis[0])

    if time_step == "future":
        final_dist_with_time = normalized_vis

    #prepare edge features matrix
    edge_features = torch.zeros(B, N, dtype=torch.long).to(normalized_vis.device)
    edge_weights = torch.ones(B, N, dtype=torch.float).to(normalized_vis.device)
    for b in range(B):
        for e in range(N):
            agents_in_edge = torch.where(final_dist_with_time[b, e] > 0)[0]  # Indices of agents -> choose the last hypergraoh to out attenstion on it
            if args.dataset == "nba":
                if all(a < 5 for a in agents_in_edge):  # Team A edge
                    edge_features[b, e] = 0
                elif all(5 <= a < 10 for a in agents_in_edge):  # Team B edge
                    edge_features[b, e] = 1
                elif 10 in agents_in_edge:  # Edge contains the ball
                    edge_features[b, e] = 3
                    edge_weights[b, e] = 2.0
                else:
                    edge_features[b, e] = 2  # Mixed team edge
            else:
                edge_features[b, e] = 1

    # print("final shape", final.shape, final)
    return normalized_vis, velocity, direction, edge_features, edge_weights


def prepare_target_GAN(constant, args, device, traj_dataset, type="train"):

    x_min, x_max = constant.X_MIN + 1, constant.X_MAX - 1
    y_min, y_max = constant.Y_MIN + 1, constant.Y_MAX - 1

    path = f'datasets/{args.dataset}/tests/{args.length}_{args.sim_num}_{args.mission_num}_{constant.n_agents}_{args.sdd_scene}_{args.scene}_{args.training_type}_{type}.pt'

    if os.path.exists(path):
        print("Loading pre-generated targets")
        return torch.load(path, weights_only=False)
    else:
        print("Creating dataset")

    dx = x_max - x_min
    dy = y_max - y_min

    targets_list = []
    controlled_agents_list = []

    for sim_idx in range(len(traj_dataset)):
        num_agents = traj_dataset[sim_idx][0].shape[0]

        targets = torch.rand(num_agents, args.mission_num, 2, device=device)

        targets[..., 0] = targets[..., 0] * dx + x_min
        targets[..., 1] = targets[..., 1] * dy + y_min

        targets_list.append(targets)
        # controlled agents
        if args.dataset == 'nba':
            max_idx = num_agents - 1
        else:
            max_idx = num_agents
        num_controlled = random.randint(1, min(args.max_covert, max_idx))
        perm = torch.randperm(max_idx, device=device)
        controlled_agents = perm[:num_controlled]

        #paDDING controlled_agents to maximum possible length (-1 where unused)
        padded = torch.full((num_agents,), fill_value=-1, dtype=torch.long, device=device)
        padded[:controlled_agents.shape[0]] = controlled_agents
        controlled_agents_list.append(padded)


    #save both targets and controlled agents
    data = {
        "targets": targets_list,  # list of tensors: each (N, M, 2)
        "controlled_agents": controlled_agents_list     #list of tensors: each (N,) with padding
    }

    torch.save(data, path)
    print(f"Saved mission dataset to {path}")
    return data
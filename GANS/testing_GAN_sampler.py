import os

import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import scipy.stats as stats


import sys


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from config import parse_args
from model.models_GAN import TrajectoryClassifier

from torch.utils.data import random_split
from model.models_sampler import SamplerMission, Sampler, SamplerMLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data.dataloader_fish import FISHDataset
from data.dataloader_syn import SYNDataset
from data.dataloader_SDD import TrajectoryDatasetSDD
from data.dataloader_ETH import TrajectoryDatasetETH, GroupedBatchSampler, seq_collate
from torch.utils.data import DataLoader
from model.GroupNet_nba import GroupNet
from data.dataloader_GAN import TrajectoryDatasetGANTest, seq_collate_GANTest
from data.dataloader_nba import NBADataset
import time
from torch.utils.tensorboard import SummaryWriter
from utilis import *




def gan_d_loss(scores_fake, scores_real, agent_idx, criterion):
    y_real = torch.ones_like(scores_real) * random.uniform(1, 1.0)
    loss_real = criterion(scores_real, y_real)
    y_fake = torch.zeros_like(scores_fake)  #Only want to penalize controlled agents
    loss_fake = criterion(scores_fake, y_fake)
    return loss_real, loss_fake


def train(constant, writer, new_train_loader, args, SM, S, D, G,criterion):
    print(args.timestamp)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=args.lr_step, gamma=args.lr_gamma)
    SM.eval()
    S.eval()
    G.eval()

    D.train()

    train_scores_real = []
    train_scores_fake = []
    train_losses_d = []
    print("new_train_loader", len(new_train_loader))
    for i in range(args.epoch_continue, args.epoch):
        time_epoch = time.time()

        train_loss_d = 0

        fake_agent_counts = np.zeros(args.agent_num)
        real_agent_counts = np.zeros(args.agent_num)

        train_real_score = np.zeros(args.agent_num)
        train_fake_score = np.zeros(args.agent_num)

        batch_num = 0
        iter_num = 0

        for data in new_train_loader:
            B, current_N, T_p, _ = data['past_traj'].shape
            agents_tragets, agents_idx, error_tolerance, _ = prepare_targets_mission_net(constant, data, args.dataset,
                                                                                         args.device, 100, five_missions=True)



            traj_sample  = data['past_traj'].to(args.device) #B, N, T, 2

            G.args.sample_k =20
            prediction20, H = G.inference_simulator(traj_sample)
            normalized_vis, velocity, direction, edge_features, edge_weights = compute_features(args, traj_sample, "past")
            prediction_for_infer = prediction20.view(20, B,current_N, args.future_length, 2).permute(1, 2, 3, 4, 0)  # (B, N, T, 2, 20)

            prediction_regular, index_S = S.inference(traj_sample, normalized_vis, velocity, direction,edge_features,edge_weights, prediction_for_infer) #1, BN, T, 2
            indices_expanded_S = index_S.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1,args.future_length, 2, 1)
            selected_S = torch.gather(prediction_for_infer, dim=4, index=indices_expanded_S).squeeze(-1)
            prediction_regular = selected_S if args.classifier_method == 'sampler_selected' else prediction_regular  # B, N, T 2
            prediction_regular = prediction_regular.detach().to(args.device)
            # print("prediction_regular", prediction_regular.shape)


            if args.disc_type in ['sampler-gan', 'sampler-mission']:
                predictions_covert, indexSM = SM.inference(args.alpha, agents_idx, agents_tragets,constant.buffer, traj_sample, normalized_vis, velocity,
                                              direction, edge_features, edge_weights, prediction_for_infer)#1, BN, T, 2
                indices_expanded_SM = indexSM.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1,args.future_length,2, 1)
                selected_SM = torch.gather(prediction_for_infer, dim=4, index=indices_expanded_SM).squeeze(-1)
                predictions_covert = selected_SM if args.classifier_method == 'sampler_selected' else predictions_covert  # B, N, T 2
                predictions_covert = predictions_covert.detach().to(args.device).squeeze(0).view(B, current_N, args.future_length, 2)

            elif args.disc_type == 'baseline':
                # print("agents_idx", agents_idx)
                # print("one_mission", one_mission[0].shape)
                # print("traj_sample shape", traj_sample.shape)

                # print("BASELUNR")
                predictions_covert = baseline_mission_reach(traj_sample, agents_tragets[0], agents_idx, error_tolerance,args)  # B, N, T, 2
                predictions_covert = predictions_covert.to(args.device).squeeze(0).view(B, current_N, args.future_length, 2)
                # print("predictions_covert", predictions_covert[0, agents_idx[0]], predictions_covert.shape)
                # print("mission", one_mission[0,0 ])
                # print("BASE")
                # print(predictions_covert.shape, predictions_covert[0,0,:,:])

            elif args.disc_type == 'smooth':

                predictions_covert = baseline_mission_wave(traj_sample, agents_tragets[0], agents_idx, error_tolerance, args)#1, BN, T, 2
                predictions_covert = predictions_covert.to(args.device).squeeze(0).view(B, current_N, args.future_length, 2)
                # print("predictions_covert", predictions_covert[0, agents_idx[0]], predictions_covert.shape)
                # print("mission", one_mission[0,0 ])


            predictions_covert  = predictions_covert.view(B, current_N, args.future_length, 2)
            prediction_regular = prediction_regular.view(B, current_N, args.future_length, 2)
            optimizer_D.zero_grad()
            scores_real = D( prediction_regular) #B, N
            scores_fake = D( predictions_covert)#B, N
            loss_real, loss_fake = gan_d_loss(scores_fake, scores_real, agents_idx, criterion)  # BCEloss
            total_loss = loss_real + loss_fake
            total_loss.backward()
            optimizer_D.step()


            #stats
            train_loss_d += total_loss.item()  # the last one from n_critic
            scores_real_mean = scores_real.mean(dim=0).detach().cpu().numpy()  # N
            scores_fake_mean = scores_fake.mean(dim=0).detach().cpu().numpy()  # N,
            for a in range(current_N):
                train_real_score[a] += scores_real_mean[a]
                real_agent_counts[a] += 1

            for k, a in enumerate(agents_idx):
                train_fake_score[a] += scores_fake_mean[k]
                fake_agent_counts[a] += 1


            batch_num += 1

            if batch_num % args.iternum_print == 0:
                print("%%%%%%%%%%%%%% iter_num %%%%%%%%%%%%%%: ", batch_num, "out of ", len(new_train_loader), "\n")
                print("############## discriminator loss")
                print("total_loss_d, ", total_loss.item(), "loss_real, ", loss_real, " loss_fake, ", loss_fake)
                print("scores_fake, ", scores_fake_mean)
                print("scores_real", scores_real_mean)

        scheduler_D.step()

        train_losses_d.append(train_loss_d / batch_num) #per epoch - ie - few simulations
        train_scores_real.append(train_real_score / np.clip(real_agent_counts, 1, None))
        train_scores_fake.append(train_fake_score / np.clip(fake_agent_counts, 1, None))

        writer.add_scalar("Loss/Train_Discriminator", train_losses_d[-1], i)

        for agent_id in range(args.agent_num):
            writer.add_scalar(f"Score/Train_Real_Agent_{agent_id}", train_scores_real[-1][agent_id], i)
            writer.add_scalar(f"Score/Train_Fake_Agent_{agent_id}", train_scores_fake[-1][agent_id], i)

        time_end = time.time()
        if (i + 1) % args.save_every == 0:
            saveModel_GAN_disc( D, args, str(i + 1),"sampler-gan-disc")

        print(
            f"Epoch [{i + 1}/{args.epoch}] - "
            f"Train Loss D: {train_losses_d[-1]:.4f}, \n"

            f"######################## scores: \n"
            f"Train Real Score (targeted agents): {np.round(train_scores_real[-1], 4)},"
            f"Fake Score (targeted agents): {np.round(train_scores_fake[-1], 4)},"

            f"Time: {time_end - time_epoch:.2f}s"
        )


def non_mission_test(test_loader, args, SM, S, D, G,  constant):
    args.batch_size = 1

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    stats_path = create_stats_folder(args, constant, 'sampler-gan-disc')

    SM.eval()
    S.eval()
    G.eval()
    D.train()

    total_missions = 0
    missions_achieved = 0
    total_controlled = 0
    full_success_count = 0

    score_list_all = []

    sim_logs = []
    all_centroids = []
    mission_tolerance = constant.buffer

    all_traj = []

    all_valid_fake_centroids = []
    all_valid_fake_traj = []
    X_MIN, X_MAX = constant.X_MIN, constant.X_MAX
    Y_MIN, Y_MAX = constant.Y_MIN, constant.Y_MAX

    for sim_id in range(len(test_loader)):

        traj_sample, _, _ = test_loader.dataset[sim_id]  # traj_sample: [N, T, 2], missions: [N, M, 2], controlled: [max_controlled]
        score_list_sim = []

        if type(traj_sample) is list:
            traj_sample = traj_sample[0]
        traj_sample = traj_sample.unsqueeze(0).to(args.device)  # [1, N, T, 2]
        controlled = []  # valid controlled agent indices

        agents_idx = controlled.copy()  # []

        iter_num = 0
        max_steps = int(args.length / 0.4)
        valid_fake_chunks = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]
        future_traj = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]

        while (future_traj.shape[1] - args.past_length < max_steps):
            with torch.no_grad():
                prediction20, H = G.inference_simulator(traj_sample)
                normalized_vis, velocity, direction, edge_features, edge_weights = compute_features(args, traj_sample,
                                                                                                    "past")
                prediction_for_infer = prediction20.view(20, 1, args.agent_num, args.future_length, 2).permute(1, 2, 3,4,0)  # (B, N, T, 2, 20)
                one_mission = torch.zeros(1, 0, 2, device=args.device)
                fake_traj, indexSM = SM.inference(args.alpha,  torch.tensor(agents_idx, device=traj_sample.device), one_mission,constant.buffer, traj_sample, normalized_vis, velocity,
                                              direction, edge_features, edge_weights, prediction_for_infer)

                indices_expanded_SM = indexSM.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1,args.future_length, 2, 1)
                selected_SM = torch.gather(prediction_for_infer, dim=4, index=indices_expanded_SM).squeeze(-1)
                fake_traj = selected_SM if args.classifier_method == 'sampler_selected' else fake_traj  # B, N, T 2


                # fake_traj, _ = G.inference_simulator(traj_sample)

                scores = D(fake_traj).squeeze(0)  # B=1, N -> N

                if args.dataset in ['sdd', 'eth']:
                    fake_np = fake_traj[0].cpu().numpy()  # shape: [N, T, 2]
                    out_of_bounds_mask = (
                            (fake_traj[0][:, :, 0] < X_MIN) | (fake_traj[0][:, :, 0] > X_MAX) |
                            (fake_traj[0][:, :, 1] < Y_MIN) | (fake_traj[0][:, :, 1] > Y_MAX)
                    )
                    invalid_mask = (out_of_bounds_mask.sum(axis=1) >= (fake_np.shape[1] // 3))  # shape: [N]
                else:
                    invalid_mask = torch.zeros(scores.shape, dtype=torch.bool)

                scores_masked = scores.clone()
                scores_masked[invalid_mask] = float('nan')
                score_list_sim.append(scores_masked)
                if invalid_mask.any():
                    pass
                else:
                    valid_fake_chunks = np.concatenate((valid_fake_chunks, fake_traj[0].cpu().numpy()),
                                                       axis=1)  # N, T, 2

            future_traj = np.concatenate((future_traj, fake_traj[0].cpu().numpy()), axis=1)
            traj_sample = fake_traj[:, :, -args.past_length:, :]
            iter_num += 1

        if np.isnan(future_traj).any():
            continue
        centroids = future_traj.mean(axis=0)  # shape (T, 2)
        all_centroids.append(centroids)
        all_traj.append(future_traj)

        # print("real all_traj len", future_traj.shape[1])
        if args.dataset in ['sdd', 'eth']:
            valid_fake_centroids = valid_fake_chunks.mean(axis=0)  # [20, 2], [8,2]...
            all_valid_fake_centroids.append(valid_fake_centroids)  # S, T-changing, 2
            all_valid_fake_traj.append(valid_fake_chunks)
            # print("fake all_traj len", valid_fake_chunks.shape[1])

        scores_tensor = torch.stack(score_list_sim, dim=0)  # shape: [T, N]
        mask = ~torch.isnan(scores_tensor)  # shape: [T, N], True where valid
        scores_tensor = torch.nan_to_num(scores_tensor, nan=0.0)
        # Sum and count
        sum_scores = (scores_tensor * mask).sum(dim=0)  # [N]
        count_valid = mask.sum(dim=0).clamp(min=1)  # [N], avoid division by 0
        sim_scores = sum_scores / count_valid
        score_list_all.append(sim_scores.cpu().numpy())

        sim_missions = 0  # C*missions for each agent -> total misssoin
        total_controlled += 0
        sim_achieved = 0  # total missions achieved
        total_missions += 0  # missions in all simulations
        missions_achieved += 0  # missions achieved in all simulations

        sim_logs.append({
            "sim_id": sim_id,
            "controlled_agents": len(controlled),
            "achieved": sim_achieved,
            "total": sim_missions,
        })

        if sim_id == 0:
            vis_predictions_no_missions(constant, future_traj, args, stats_path)

    score_array = np.stack(score_list_all, axis=0)  # shape: [num_simulations, num_agents]
    scores_mean_agents = np.nanmean(score_array, axis=0)
    scores_std_agents = np.nanstd(score_array, axis=0)

    print("starting analysis of usage")
    df_centroids = analyze_usage(
        all_trajectories=all_traj if args.dataset not in ['eth', 'sdd'] else all_valid_fake_traj,
        all_centroids=all_centroids if args.dataset not in ['eth', 'sdd'] else all_valid_fake_centroids,
        # list of np.ndarray, shape [T, 2]
        field_length=constant.field if args.dataset != "sdd" else constant.field_T,
        field_width=constant.field_width if args.dataset != "sdd" else constant.field_width_T,
        num_blocks=constant.num_blocks,
        timestep_duration=0.4,
        args=args,
        folder=f'{stats_path}/no_mission/',
        constant=constant,
        gen_gt_type="sampler"
    )

    df_per_sim = pd.DataFrame(sim_logs)

    df_summary = pd.DataFrame({
        "Total Missions": [total_missions],
        "Total Controlled Agents": [total_controlled],
        "Total Missions Achieved": [missions_achieved],

        "Avg Scores for agents (Between Sims)": [scores_mean_agents],
        'Std Scores for agents (Between Sims)': [scores_std_agents],

        "Avg Scores for agents (Total)": [scores_mean_agents.mean()],
        'Std Scores for agents (Total)': [scores_std_agents.mean()],


    })

    print(f"\n--- Final Statistics over {len(test_loader)} simulations ---")
    print(df_summary.T)

    if args.dataset == "nba":
        df_centroids.to_csv(
            f"{stats_path}/no_mission/centroids_agents.csv")
    df_per_sim.to_csv(
        f"{stats_path}/no_mission/per_sim_stats.csv")
    df_summary.to_csv(
        f"{stats_path}/no_mission/overall_summary.csv")



def mission_test(testing_loader, args,SM, S, D, G, constant):
    args.batch_size = 1
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    stats_path = create_stats_folder(args, constant, 'sampler-gan-disc')

    G.eval()
    D.eval()
    SM.eval()
    S.eval()
    total_missions = 0
    missions_achieved = 0
    total_controlled = 0
    full_success_count = 0

    score_list_all = []
    visualized= False
    sim_logs = []
    all_centroids = []
    mission_tolerance = constant.buffer

    per_agent_mission_times = []
    all_traj = []
    all_mission_durations = []
    agent_full_success = []
    controlled_all = []
    sims_success_rate = []

    all_valid_fake_centroids = []
    all_valid_fake_traj = []
    X_MIN, X_MAX = constant.X_MIN, constant.X_MAX
    Y_MIN, Y_MAX = constant.Y_MIN, constant.Y_MAX
    print("len(testing_loader)", len(testing_loader))

    for sim_id in range(len(testing_loader)):

        traj_sample, missions, controlled = testing_loader.dataset[sim_id]  # traj_sample: [N, T, 2], missions: [N, M, 2], controlled: [max_controlled]

        if type(traj_sample) is list:
            traj_sample = traj_sample[0]
        traj_sample = traj_sample.unsqueeze(0).to(args.device)  # [1, N, T, 2]
        missions = missions.to(args.device)  # [N, M, 2]
        controlled = controlled[controlled != -1].tolist()  # valid controlled agent indices
        agent_missions = [0] * len(controlled)  # which missions collected for each agent
        agents_idx = controlled.copy()
        agents_idx_plot = controlled.copy()
        agents_targs = missions[agents_idx]  # [C, M, 2]

        mission_log = []
        target_status = {}
        iter_num = 0
        all_mission_accomplished = False
        max_steps = int(args.length / 0.4)
        score_list_sim = []
        invalid_score_mask_sim = []  # stores one [N] mask per T step
        valid_fake_chunks = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]

        future_traj = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]
        agent_mission_times = {a: [] for a in agents_idx}
        start_times = {a: args.past_length for a in agents_idx}

        while (future_traj.shape[1] - args.past_length < max_steps):
            with torch.no_grad():
                agent_missions_ten = torch.tensor(agent_missions, device=traj_sample.device)  # shape: [C]
                one_mission = agents_targs.gather(1, agent_missions_ten.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)

                # print("agents_idx", agents_idx)
                # print("one_mission", one_mission)
                # print("traj_sample shape", traj_sample.shape)
                # print("controlled target distance", (one_mission - traj_sample[:, agents_idx, -1, :]).norm(dim=-1))

                G.args.sample_k =20
                prediction20, H = G.inference_simulator(traj_sample)
                normalized_vis, velocity, direction, edge_features, edge_weights = compute_features(args, traj_sample,"past")
                prediction_for_infer = prediction20.view(20, 1, args.agent_num, args.future_length, 2).permute(1, 2, 3,4,0)  # (B, N, T, 2, 20)

                prediction_regular, index_S = S.inference(traj_sample, normalized_vis, velocity, direction,edge_features,edge_weights, prediction_for_infer) #1, BN, T, 2
                indices_expanded_S = index_S.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.future_length, 2, 1)
                selected_S = torch.gather(prediction_for_infer, dim=4, index=indices_expanded_S).squeeze(-1)
                prediction_regular = selected_S if args.classifier_method == 'sampler_selected' else prediction_regular  # B, N, T 2


                if args.disc_type in ['sampler-gan', 'sampler-mission']:
                    predictions_covert, indexSM = SM.inference(args.alpha,  torch.tensor(agents_idx, device=traj_sample.device), one_mission,constant.buffer, traj_sample, normalized_vis, velocity,
                                                  direction, edge_features, edge_weights, prediction_for_infer)#1, BN, T, 2
                    indices_expanded_SM = indexSM.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1,args.future_length,2, 1)
                    selected_SM = torch.gather(prediction_for_infer, dim=4, index=indices_expanded_SM).squeeze(-1)
                    predictions_covert = selected_SM if args.classifier_method == 'sampler_selected' else predictions_covert  # B, N, T 2

                elif args.disc_type == 'baseline':
                    if len(agents_idx) != 0:
                        predictions_covert = baseline_mission_reach(traj_sample.clone(), one_mission,  torch.tensor(agents_idx, device=traj_sample.device), mission_tolerance,args)  # 1, BN, T, 2
                    else:
                        # print("idx 0")
                        predictions_covert = torch.zeros(1, 0,args.future_length ,2).to(args.device)
                elif args.disc_type == 'closest-20':
                    # if len(agents_idx) != 0:
                    #     predictions_covert = best_of_20_mission_baseline(traj_sample, prediction20, one_mission, agents_idx, args)
                    # else:
                    #     predictions_covert = torch.zeros(1, 0,args.future_length ,2).to(args.device)
                    if len(agents_idx) !=0:
                        predictions_covert = baseline_mission_wave(traj_sample.clone(), one_mission,  torch.tensor(agents_idx, device=traj_sample.device), mission_tolerance, args)#1, BN, T, 2
                    else:
                        predictions_covert = torch.zeros(1, 0,args.future_length ,2).to(args.device)



                fake_traj = prediction_regular.clone()
                fake_traj[:, agents_idx, :, :] = predictions_covert[:, agents_idx, :, :]


                scores = D( fake_traj).squeeze(0)  # B=1, N -> N

                if args.dataset in ['sdd', 'eth']:
                    fake_np = fake_traj[0].cpu().numpy()  # shape: [N, T, 2]
                    out_of_bounds_mask = (
                            (fake_np[:, :, 0] < X_MIN) | (fake_np[:, :, 0] > X_MAX) |
                            (fake_np[:, :, 1] < Y_MIN) | (fake_np[:, :, 1] > Y_MAX)
                    )
                    invalid_mask = (out_of_bounds_mask.sum(axis=1) > (fake_np.shape[1] // 3))  # shape: [N]
                else:
                    invalid_mask = torch.zeros(scores.shape, dtype=torch.bool)

                # if sim_id == 0:
                # print("invalid_mask", invalid_mask)
                # print("fake_np", fake_np)
                scores_masked = scores.clone()
                scores_masked[invalid_mask] = float('nan')
                score_list_sim.append(scores_masked)
                if invalid_mask.any():  # if there is any invalid agent - dont add it to calc for fixed number of N
                    pass
                else:
                    valid_fake_chunks = np.concatenate((valid_fake_chunks, fake_traj[0].cpu().numpy()),
                                                       axis=1)  # N, T, 2

            agents_to_remove = []

            for i in reversed(range(len(agents_idx))):  # reversed to safely remove
                agent = agents_idx[i]
                mission_id = agent_missions[i]
                target = agents_targs[i, mission_id]
                agent_path = fake_traj[0, agent]  # shape: [T, 2]

                # distance check (line-to-point projection)
                p1 = agent_path[:-1]
                p2 = agent_path[1:]
                seg = p2 - p1
                seg_len = seg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                proj = ((target - p1) * seg).sum(-1, keepdim=True) / (seg_len ** 2)
                proj_clamped = proj.clamp(0, 1)
                closest = p1 + proj_clamped * seg
                dists = (closest - target).norm(dim=-1)  # [T-1]

                hit_steps = (dists < mission_tolerance).nonzero(as_tuple=True)[0]

                if len(hit_steps) > 0:
                    line_hit = hit_steps[0].item()
                    time_hit = args.past_length + iter_num * args.future_length + line_hit
                    mission_log.append((time_hit, agent, mission_id, target.cpu().numpy()))
                    target_status[(agent, mission_id)] = True
                    agent_mission_times[agent].append((time_hit - start_times[agent]) * 0.4)
                    all_mission_durations.append((time_hit - start_times[agent]) * 0.4)
                    start_times[agent] = time_hit  # starting time for next mission

                    if mission_id < args.mission_num - 1:
                        agent_missions[i] += 1
                    else:
                        agents_to_remove.append(i)

            for idx in agents_to_remove:
                a = agents_idx[idx]
                agent_full_success.append(a)
                del agents_idx[idx]
                del agent_missions[idx]
                agents_targs = torch.cat([agents_targs[:idx], agents_targs[idx + 1:]], dim=0)

            all_mission_accomplished = len(agents_idx) == 0
            future_traj = np.concatenate((future_traj, fake_traj[0].cpu().numpy()), axis=1)
            traj_sample = fake_traj[:, :, -args.past_length:, :]
            iter_num += 1

        if np.isnan(future_traj).any():
            print("NANAS")
            continue
        centroids = future_traj.mean(axis=0)  # shape (T, 2)
        all_centroids.append(centroids)  # S, T, 2
        all_traj.append(future_traj)

        # print("real all_traj len", future_traj.shape[1])
        if args.dataset in ['sdd', 'eth']:
            valid_fake_centroids = valid_fake_chunks.mean(axis=0)  # [20, 2], [8,2]...
            all_valid_fake_centroids.append(valid_fake_centroids)  # S, T-changing, 2
            all_valid_fake_traj.append(valid_fake_chunks)
            print("fake all_traj len", valid_fake_chunks.shape[1])

        scores_tensor = torch.stack(score_list_sim, dim=0)  # shape: [T, N]
        mask = ~torch.isnan(scores_tensor)  # shape: [T, N], True where valid
        scores_tensor = torch.nan_to_num(scores_tensor, nan=0.0)

        sum_scores = (scores_tensor * mask).sum(dim=0)  # [N]
        count_valid = mask.sum(dim=0).clamp(min=1)  # [N], avoid division by 0
        sim_scores = sum_scores / count_valid

        score_list_all.append(sim_scores.cpu().numpy())  # collect into main list
        controlled_all.append(agents_idx_plot.copy())

        sim_missions = len(controlled) * args.mission_num  # C*missions for each agent -> total misssoin
        total_controlled += len(controlled)
        sim_achieved = len(mission_log)  # total missions achieved
        total_missions += sim_missions  # missions in all simulations
        missions_achieved += sim_achieved  # missions achieved in all simulations
        if sim_achieved == sim_missions:
            full_success_count += 1  # in this sim all misssions achieved

        sim_success_rate = sim_achieved / sim_missions
        sims_success_rate.append(sim_success_rate)

        mission_durations = [t for a in agent_mission_times for t in agent_mission_times[a]]
        per_agent_mission_times.extend(mission_durations)

        sim_logs.append({
            "sim_id": sim_id,
            "controlled_agents": len(controlled),
            "achieved": sim_achieved,
            "total": sim_missions,
            "full_success": full_success_count,
            "min_mission_time": np.min(mission_durations) if mission_durations else None,
            "max_mission_time": np.max(mission_durations) if mission_durations else None,
            "mean_mission_time": np.mean(mission_durations) if mission_durations else None,
            "avg_missions_per_agent": sim_achieved / len(controlled),
            "mission_log": mission_log,
            "sim_success_rate": sim_success_rate
        })


        if len(agents_idx_plot) == args.agents_to_plot and not visualized:
            print("VIS")
            vis_predictions_missions(constant,future_traj, mission_log, target_status, args, missions.cpu(), agents_idx_plot, stats_path)
            visualized = True

    score_array = np.stack(score_list_all, axis=0)  # shape: [num_simulations, num_agents]
    controlled_mask = np.zeros_like(score_array, dtype=bool)  # shape [S, N]

    for i, c in enumerate(controlled_all):
        controlled_mask[i, c] = True
    controlled_scores = np.where(controlled_mask, score_array, np.nan)
    uncontrolled_scores = np.where(~controlled_mask, score_array, np.nan)

    valid_controlled_agents = ~np.all(np.isnan(controlled_scores), axis=0)
    valid_controlled_scores = controlled_scores[:, valid_controlled_agents]

    valid_uncontrolled_agents = ~np.all(np.isnan(uncontrolled_scores), axis=0)
    valid_uncontrolled_scores = uncontrolled_scores[:, valid_uncontrolled_agents]

    controlled_mean = np.nanmean(valid_controlled_scores, axis=0)
    controlled_std = np.nanstd(valid_controlled_scores, axis=0)
    uncontrolled_mean = np.nanmean(valid_uncontrolled_scores, axis=0)
    uncontrolled_std = np.nanstd(valid_uncontrolled_scores, axis=0)


    sim_num = valid_controlled_scores.shape[0]

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for i in range(sim_num):
        y_scores_controlled = valid_controlled_scores[i]
        y_scores_uncontrolled = valid_uncontrolled_scores[i]

        y_true = np.concatenate([
            np.zeros_like(y_scores_controlled),  # 0 = controlled
            np.ones_like(y_scores_uncontrolled)  # 1 = uncontrolled
        ])

        y_pred = np.concatenate([
            (y_scores_controlled > 0.5).astype(int),
            (y_scores_uncontrolled > 0.5).astype(int)
        ])

        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, pos_label=0, zero_division=0))


    y_scores_controlled = valid_controlled_scores.flatten()
    y_scores_uncontrolled = valid_uncontrolled_scores.flatten()

    y_true = np.concatenate([
        np.zeros_like(y_scores_controlled),
        np.ones_like(y_scores_uncontrolled)
    ])

    y_pred = np.concatenate([
        (y_scores_controlled > 0.5).astype(int),
        (y_scores_uncontrolled > 0.5).astype(int)
    ])

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=0)
    recall = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred, pos_label=0)

    metrics = {
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "precision_mean": np.mean(precisions),
        "precision_std": np.std(precisions),
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "recall_all": recall,
        "accuracy_all": accuracy,
        "precision_all": precision,
        "f1_all": f1,
    }

    print(f'Recall: {metrics["recall_mean"]:.4f}, {metrics["recall_std"]}, all {recall}')



    df_per_sim = pd.DataFrame(sim_logs)

    avg_per_agent_all = df_per_sim["avg_missions_per_agent"].mean()
    std_per_agent_all = df_per_sim["avg_missions_per_agent"].std()
    avg_sim_success_rate = df_per_sim["sim_success_rate"].mean()
    std_sim_success_rate = df_per_sim["sim_success_rate"].std()

    #####
    n = len(df_per_sim)

    # 95% confidence interval
    confidence = 0.95
    z = stats.norm.ppf(1 - (1 - confidence) / 2)  # ≈ 1.96
    margin_of_error_sim = z * std_sim_success_rate / np.sqrt(n)
    margin_of_error_agent = z * std_per_agent_all / np.sqrt(n)


    grouped = df_per_sim.groupby("controlled_agents")
    covert_stats = grouped.agg(
        {"avg_missions_per_agent": ['mean', 'std'], "mean_mission_time": ['mean', 'std']}).reset_index()
    covert_stats.columns = [
        "num_covert_agents",
        "avg_missions_mean", "avg_missions_std",
        "mission_time_mean", "mission_time_std"
    ]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(covert_stats["num_covert_agents"] - 0.15, covert_stats["avg_missions_mean"],
            yerr=covert_stats["avg_missions_std"], width=0.3, label="Avg Missions per Agent")
    ax1.set_ylabel("Avg Missions per Agent")
    ax1.set_xlabel("Number of Covert Agents")
    ax1.set_xticks(covert_stats["num_covert_agents"])

    ax2 = ax1.twinx()
    ax2.bar(covert_stats["num_covert_agents"] + 0.15, covert_stats["mission_time_mean"],
            yerr=covert_stats["mission_time_std"], width=0.3, color='orange', label="Avg Mission Time")
    ax2.set_ylabel("Avg Mission Time (s)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Mission Performance by Number of Covert Agents")
    plt.tight_layout()
    plt.savefig(os.path.join(stats_path, "covert_agent_stats.png"))
    plt.close()

    if len(per_agent_mission_times) > 0:
        min_time = np.min(per_agent_mission_times)
        max_time = np.max(per_agent_mission_times)
        mean_mission_time = np.mean(per_agent_mission_times)
        std_mission_time = np.std(per_agent_mission_times)
    else:
        min_time = None
        std_mission_time = None
        mean_mission_time = None
        max_time = None

    df_summary = pd.DataFrame({
        "Total Missions": [total_missions],
        "Avg Missions per Simulation": [total_missions / len(testing_loader)],
        "Total Controlled Agents": [total_controlled],
        "Total Missions Achieved": [missions_achieved],
        "Overall Success Rate": [missions_achieved / total_missions],
        "Simulations with Full Success": [full_success_count],
        "Avg Missions Achieved per Simulation (General)": [missions_achieved / len(testing_loader)],
        "avg simimulation success rate": [avg_sim_success_rate],
        "Std simimulation success rate": [std_sim_success_rate],
        "margin_of_error_sim: ": [margin_of_error_sim],

        "Mean Mission Time (All Agents)": [mean_mission_time],
        "Std Mission Time (All Agents)": [std_mission_time],
        "Std of Mean Mission Time (Between Sims)": [df_per_sim["mean_mission_time"].std()],

        "Min Mission Time (All Agents)": [min_time],
        "Min of Mean Mission Times (Between Sims)": [df_per_sim["mean_mission_time"].min()],

        "Max Mission Time (All Agents)": [max_time],
        "Max of Mean Mission Times (Between Sims)": [df_per_sim["mean_mission_time"].max()],

        "Avg Missions per Agent (All Sims)": [avg_per_agent_all],
        "Std of Avg Missions per Agent (Between Sims)": [std_per_agent_all],
        "margin_of_error_agent: ": [margin_of_error_agent],

        "Avg Scores for controlled agents (Between Sims)": [controlled_mean],
        "Std Scores for controlled agents (Between Sims)": [controlled_std],
        "Avg Scores for uncontrolled agents (Between Sims)": [uncontrolled_mean],
        "Std of Scores for uncontrolled agents (Between Sims)": [uncontrolled_std],

        "Avg Scores for controlled agents (Between Sims) (Total)": [controlled_mean.mean()],
        "Std Scores for controlled agents (Between Sims) (Total)": [controlled_std.mean()],
        "Avg Scores for uncontrolled agents (Between Sims) (Total)": [uncontrolled_mean.mean()],
        "Std of Scores for uncontrolled agents (Between Sims) (Total)": [uncontrolled_std.mean()],

    })

    print(f"\n--- Final Statistics over {len(testing_loader)} simulations ---")
    print(df_summary.T)

    print("starting analysis of usage")

    df_centroids = analyze_usage(
        all_trajectories=all_traj if args.dataset not in ['eth', 'sdd'] else all_valid_fake_traj,
        all_centroids=all_centroids if args.dataset not in ['eth', 'sdd'] else all_valid_fake_centroids,
        # list of np.ndarray, shape [T, 2]
        field_length=constant.field if args.dataset != "sdd" else constant.field_T,
        field_width=constant.field_width if args.dataset != "sdd" else constant.field_width_T,
        num_blocks=constant.num_blocks,
        timestep_duration=0.4,
        args=args,
        folder=stats_path,
        constant=constant,
        gen_gt_type= "sampler"
    )

    if args.dataset == "nba":
        df_centroids.to_csv(f"{stats_path}/centroids_agents.csv")
    df_per_sim.to_csv(f"{stats_path}/per_sim_stats.csv")
    df_summary.to_csv(f"{stats_path}/overall_summary.csv")
    with open(f"{stats_path}/metrics_summary.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")




def create_dataset_split_flex(constant, train_loader, save_path,train_size=300, test_size=100, ):
    train_file = os.path.join(save_path, "train_dataset.pt")
    test_file = os.path.join(save_path, "test_dataset.pt")
    if os.path.exists(train_file) and os.path.exists(test_file):
        print("Loading saved split datasets...")
        train_dataset = torch.load(train_file, weights_only=False)
        test_dataset = torch.load(test_file, weights_only=False)
        return train_dataset, test_dataset

    os.makedirs(save_path, exist_ok=True)

    past_traj_list_train = []
    past_traj_list_test = []
    future_traj_list_train = []
    future_traj_list_test = []

    seq_start_end_train = []
    seq_start_end_test = []

    grouped_seq_indices_train = defaultdict(list)
    grouped_seq_indices_test = defaultdict(list)

    agent_counter_train = 0
    agent_counter_test = 0

    scene_counter_train = 0
    scene_counter_test = 0

    train_scene_count = 0
    test_scene_count = 0

    for data in train_loader:
        B, N, T, _ = data['past_traj'].shape
        past_traj = data['past_traj'].view(B * N, T, 2).detach().cpu()

        for b in range(B):
            start_idx = b * N
            end_idx = (b + 1) * N
            single_scene = past_traj[start_idx:end_idx]  # (N, T, 2)

            if test_scene_count < test_size:
                if N == constant.n_agents:
                    past_traj_list_test.append(single_scene)
                    seq_start_end_test.append((agent_counter_test, agent_counter_test + N))
                    grouped_seq_indices_test[N].append(scene_counter_test)

                    agent_counter_test += N
                    scene_counter_test += 1
                    test_scene_count += 1
                else:
                    continue

            elif train_scene_count < train_size:
                past_traj_list_train.append(single_scene)
                seq_start_end_train.append((agent_counter_train, agent_counter_train + N))
                grouped_seq_indices_train[N].append(scene_counter_train)

                agent_counter_train += N
                scene_counter_train += 1
                train_scene_count += 1

            if train_scene_count >= train_size and test_scene_count >= test_size:
                break

        if train_scene_count >= train_size and test_scene_count >= test_size:
            break

    #conc train/test agent trajectories
    past_traj_all_train = torch.cat(past_traj_list_train, dim=0)  # (total_train_agents, T, 2)
    past_traj_all_test = torch.cat(past_traj_list_test, dim=0)  # (total_test_agents, T, 2)

    #final datasets
    train_dataset = TrajectoryDatasetGANTest(past_traj_all_train, seq_start_end_train, grouped_seq_indices_train)
    test_dataset = TrajectoryDatasetGANTest(past_traj_all_test, seq_start_end_test, grouped_seq_indices_test)

    torch.save(train_dataset, train_file)
    torch.save(test_dataset, test_file)

    print(
        f"Saved disjoint train ({train_scene_count} scenes) and test ({test_scene_count} scenes) datasets to {save_path}")
    return train_dataset, test_dataset


def create_dataset_split(dataset, save_path,train_size=300, test_size=100, ):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if os.path.exists(f"{save_path}_train_subset.pt"):
        print(f"Loading dataset split from {save_path}")
        subset_400 = torch.utils.data.Subset(dataset, list(range(train_size + test_size)))
        train_subset = torch.load(f"{save_path}_train_subset.pt", weights_only=False)
        test_subset = torch.load(f"{save_path}_test_subset.pt", weights_only=False)
        train_subset.dataset = subset_400
        test_subset.dataset = subset_400
    else:
        print("Creating new dataset split...")
        total_size = train_size + test_size
        assert len(dataset) >= total_size, f"Dataset too small ({len(dataset)} < {total_size})"

        subset_400 = torch.utils.data.Subset(dataset, list(range(train_size+test_size)))

        train_subset, test_subset = random_split(
            subset_400,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        torch.save(train_subset, f"{save_path}_train_subset.pt")
        torch.save(test_subset, f"{save_path}_test_subset.pt")

    return train_subset, test_subset


if __name__ == '__main__':
    args = parse_args()

    """ setup """
    names = [x for x in args.model_names.split(',')]

    if args.dataset == 'nba':
        args.model_save_dir = 'G1/saved_models/nba'
        args.agent_num = 11
        args.edge_num = 12
        args.mission_num =12
        args.length = 120
        train_size = 12000
        d_set = NBADataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=args.training_type)

        constant = ConstantNBA

    elif args.dataset == 'fish':
        args.model_save_dir = 'G1/saved_models/fish'
        args.agent_num = 8
        args.edge_num = 16
        args.mission_num =12
        args.length = 120
        train_size = 4000
        d_set = FISHDataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=args.training_type)

        constant = ConstantFish

    elif args.dataset == 'syn':
        args.agent_num = 6
        args.model_save_dir = 'G1/saved_models/syn'
        d_set = SYNDataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=args.training_type)
        if args.training_type == 'train_rigid':
            val_part = 'val_rigid'
            constant = ConstantSYNR
        else:
            val_part = 'val_smooth'
            constant = ConstantSYNS


    elif args.dataset == 'sdd':
        args.past_length = 8
        args.future_length = 12
        args.agent_num = 8
        args.mission_num = 8
        args.length = 60
        train_size = 12000
        args.model_save_dir = 'G1/saved_models/sdd'
        d_set = TrajectoryDatasetSDD(data_dir="datasets/sdd", obs_len=args.past_length,
                                     pred_len=args.future_length, skip=1,
                                     min_ped=1, delim='space', save_path="datasets/sdd/SDD.pt",
                                     mode=args.training_type)

        constant = ConstantSDD2
    elif args.dataset == 'eth':
        args.past_length = 8
        args.future_length = 12
        args.agent_num = 8
        args.model_save_dir = 'G1/saved_models/eth'
        d_set = TrajectoryDatasetETH(data_dir="datasets/eth", obs_len=args.past_length,
                                     pred_len=args.future_length, skip=1, min_ped=1,
                                     delim='space', test_scene=args.scene, save_path=args.scene,
                                     mode=args.training_type)
        constant = return_the_eth_scene(args.scene)



    for name in names:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        """ model """
        saved_path = os.path.join(args.model_save_dir, str(name) + '.p')
        print('load model from:', saved_path)
        checkpoint = torch.load(saved_path, map_location=args.device, weights_only=False)
        training_args = checkpoint['model_cfg']

#for sampler
        G = GroupNet(training_args, args.device)
        G.set_device(args.device)
        G.eval()
        G.load_state_dict(checkpoint['model_dict'], strict=True)
        G.args.sample_k = 20

        # regular simulator
        if args.test_mlp:
            S =SamplerMLP(args, args.device, args.past_length - 1, args.future_length - 1, 32, 64,
                        args.past_length - 1
                        , args.future_length - 1,
                        32, 128, 1, 2, 8, edge_dim=16, bias=True).to(args.device)
        else:
            S = Sampler(args, args.device, args.past_length - 1, args.future_length - 1, 32, 64,
                        args.past_length - 1
                        , args.future_length - 1,
                        32, 128, 1, 2, 8, edge_dim=16, bias=True).to(args.device)
        saved_path = f"Sampler/saved_model/{args.dataset}/{args.saved_models_SAM}.pth"
        S.load_state_dict(torch.load(saved_path, map_location=args.device, weights_only=False))
        S.to(args.device)
        S.eval()

        #mission aware simulator
        SM = SamplerMission(args, args.device, args.past_length - 1, args.future_length - 1, 32, 64,
                            args.past_length - 1
                            , args.future_length - 1,
                            32, 128, 1, 2, 8, edge_dim=16, bias=True).to(args.device)
        if args.disc_type in ['sampler-gan']:
            SM_path = f"GANS/saved_model/{args.dataset}/{args.saved_models_GAN_SM}.pth"
            SM.load_state_dict(torch.load(SM_path, weights_only=False))
            SM.to(args.device)
            SM.eval()
        else:
            SM.to(args.device)
            SM.eval()

        test_size = args.sim_num #can be same as groupnets'
        save_path = f"GANS/data/train_test_indexes_for_GAN_groupnet_{args.dataset}_{args.training_type}_{args.scene}_{args.sdd_scene}_"

        if args.dataset in ['nba', 'fish', 'syn']:
            train_dataset, test_dataset = create_dataset_split(d_set,save_path,train_size, test_size)
            new_train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=seq_collate,
                pin_memory=True)

        else:
            #more advanced spliting...
            train_loader = DataLoader(d_set,
                                      batch_sampler=GroupedBatchSampler(d_set.grouped_seq_indices_test,
                                                                        batch_size=args.batch_size,
                                                                        shuffle=False,
                                                                        drop_last=False), collate_fn=seq_collate)
            train_dataset, test_dataset = create_dataset_split_flex(constant, train_loader, save_path, train_size, test_size)
            new_train_loader = DataLoader(train_dataset,
                                      batch_sampler=GroupedBatchSampler(train_dataset.grouped_seq_indices,
                                                                        batch_size=args.batch_size,
                                                                        shuffle=False,
                                                                        drop_last=False), collate_fn=seq_collate_GANTest)

        if args.mode == 'train':


            criterion = torch.nn.BCELoss()
            D = TrajectoryClassifier(args.device) #new classifier
            if args.epoch_continue > 0:
                saved_path = f"./GAN/GAN_saved_model/sampler-gan-disc/{args.dataset}/{args.disc_type}/{args.saved_models_DIS}.pth"
                print(f'load model from: {saved_path}')
                Dstate_dict  = torch.load(saved_path, map_location=args.device, weights_only=False)
                D.load_state_dict(Dstate_dict)

            D.to(args.device)
            writer = SummaryWriter(log_dir=f"runs/GAN_testing_disc_sampler/{args.timestamp}_{args.disc_type}_{args.dataset}_{args.scene}_{args.sdd_scene}_{args.training_type}_{args.seed}_{args.classifier_method}")
            train(constant, writer, new_train_loader, args, SM, S, D, G, criterion)


        else:
            SEED = args.seed
            g = torch.Generator()
            g.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


            D = TrajectoryClassifier(args.device).to(args.device)
            saved_path = f"GANS/saved_model/{args.dataset}/{args.disc_type}/{args.saved_models_non_trained_DIS_SM}.pth"

            D.load_state_dict(torch.load(saved_path, map_location=args.device, weights_only=False))
            D.to(args.device)


            data = prepare_target(args, args.device, test_dataset, constant, "test")

            wrapped_test = MissionTestDatasetGAN(test_dataset, data)
            testing_loader = DataLoader(
                wrapped_test,
                batch_size=args.batch_size,  # simulate one scenario at a time
                shuffle=False,
                num_workers=0,
                pin_memory=True, generator=g,
            )
            mission_test(testing_loader, args, SM, S, D, G, constant)
            if args.disc_type in ['sampler-gan']:
                print("starting non mission test")
                non_mission_test(testing_loader, args, SM, S, D, G, constant)



import argparse
import os
import sys
import datetime


from torch.utils.data import Dataset
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from data.dataloader_nba import NBADataset
from data.dataloader_fish import FISHDataset
from model.GroupNet_nba_mission import GroupNetM
from data.dataloader_SDD import TrajectoryDatasetSDD
from data.dataloader_ETH import TrajectoryDatasetETH, GroupedBatchSampler, seq_collate
from torch.utils.data import random_split, DataLoader
from utilis import *
from data.dataloader_syn import SYNDataset





def test_one_traj(folder, test_loader, model, args,constant, simple_dist_plot =False, dist_plot=False, measure_res=False, plot_20 = False, XGB=False):
    total_num_pred = 0
    all_num = 0
    l2error_overall = 0
    l2error_dest = 0
    l2error_avg_04s = 0
    l2error_dest_04s = 0
    l2error_avg_08s = 0
    l2error_dest_08s = 0
    l2error_avg_12s = 0
    l2error_dest_12s = 0
    l2error_avg_16s = 0
    l2error_dest_16s = 0
    l2error_avg_20s = 0
    l2error_dest_20s = 0
    l2error_avg_24s = 0
    l2error_dest_24s = 0
    l2error_avg_28s = 0
    l2error_dest_28s = 0
    l2error_avg_32s = 0
    l2error_dest_32s = 0
    l2error_avg_36s = 0
    l2error_dest_36s = 0


    l2error_overall_base = 0
    l2error_dest_base = 0
    l2error_avg_04s_base = 0
    l2error_dest_04s_base = 0
    l2error_avg_08s_base = 0
    l2error_dest_08s_base = 0
    l2error_avg_12s_base = 0
    l2error_dest_12s_base = 0
    l2error_avg_16s_base = 0
    l2error_dest_16s_base = 0
    l2error_avg_20s_base = 0
    l2error_dest_20s_base = 0
    l2error_avg_24s_base = 0
    l2error_dest_24s_base = 0
    l2error_avg_28s_base= 0
    l2error_dest_28s_base = 0
    l2error_avg_32s_base = 0
    l2error_dest_32s_base = 0
    l2error_avg_36s_base = 0
    l2error_dest_36s_base = 0
    ade = 0
    fde = 0
    ade_base = 0
    fde_base = 0
    all_X = []
    all_y = []

    tcc_base = []
    tcc_new = []
    tcc_group = []
    iteration = 0
    mission_tolerance = constant.buffer
    model.eval()


    for sim_id in range(len(test_loader)):

        traj_sample, missions, controlled, future = test_loader.dataset[sim_id]  # traj_sample: [N, T, 2], missions: [N, M, 2], controlled: [max_controlled]
        traj_sample = traj_sample.unsqueeze(0).to(args.device)  # [1, N, T, 2]
        missions = missions.to(args.device)  # [N, M, 2]
        controlled = controlled[controlled != -1].tolist()  # valid controlled agent indices
        agent_missions = [0] * len(controlled) # which missions collected for each agent

        agents_idx = controlled.copy()
        agents_targs = missions[agents_idx]  # [C, M, 2]


        with torch.no_grad():
            current_mission_ids = torch.tensor(agent_missions, device=args.device)
            current_targets = agents_targs[torch.arange(len(agents_idx)), current_mission_ids]  # [C, 2]
            prediction, _ = model.inference_simulator(traj_sample, current_targets, torch.tensor(agents_idx, device=traj_sample.device), mission_tolerance)  # prediction: [1, N, T, 2]

        if args.dataset == "sdd":
            normalized_traj_sample = traj_sample
            traj_sample = denormalize_traj(traj_sample)
            past_traj = traj_sample.detach().cpu().numpy() * args.traj_scale
            normalized_past = normalized_traj_sample.detach().cpu().numpy() * args.traj_scale

        else:
            traj_sample = normalized_traj_sample = traj_sample
            past_traj = normalized_past = traj_sample.detach().cpu().numpy() * args.traj_scale


        last_5_steps = past_traj[:, :, -5:, :]
        avg_velocity = np.mean(np.diff(last_5_steps, axis=2), axis=2, keepdims=True)
        last_position = last_5_steps[:, :, -1:, :]
        baseline_prediction = np.concatenate(
            [last_position + i * avg_velocity for i in range(1, args.future_length+1)], axis=2
        )

        future_traj = future.unsqueeze(0).detach().cpu().numpy() * args.traj_scale  # B,N,T,2
        batch = future_traj.shape[0]
        actor_num = future_traj.shape[1]
        prediction = prediction * args.traj_scale
        BN = batch * actor_num

        if prediction.shape[0] == 20:
            random_indices = torch.randint(0, 20, (BN,), device=prediction.device)
            prediction_new = prediction[random_indices, torch.arange(BN)].detach().cpu().numpy()
        else:
            prediction_new = prediction.squeeze(0).detach().cpu().numpy() # (20,BN,T,2)

        baseline_prediction = baseline_prediction.reshape(BN, args.future_length, 2)
        future_traj = np.reshape(future_traj, (BN, args.future_length, 2))

        if args.dataset == "sdd":
            future_traj_normalized = future_traj
            future_traj = denormalize_traj(future_traj)
            prediction_new_normalized = prediction_new
            prediction_new = denormalize_traj(prediction_new)

        else:
            future_traj_normalized = future_traj
            prediction_new_normalized = prediction_new


        if iteration < 15:
            future_ploting = np.reshape(future_traj_normalized, (batch, actor_num, args.future_length, 2))
            previous_3D = np.reshape(normalized_past, (batch, actor_num, args.past_length, 2))
            best = np.reshape(prediction_new_normalized, (batch, actor_num, args.future_length, 2))
            draw_result(constant, folder, args, best, previous_3D, mission_aware = agents_idx,GT_num =iteration , mission=current_targets.cpu().numpy())
            draw_result(constant, folder, args, future_ploting, previous_3D, mode='gt', mission_aware = agents_idx,GT_num = iteration, mission=current_targets.cpu().numpy())

        tcc_new.extend(calc_TCC(prediction_new, future_traj, 1))
        tcc_base.extend(calc_TCC(baseline_prediction, future_traj, 1))

        if args.dataset == 'nba' or args.dataset == 'fish' or args.dataset == 'syn':
            l2error_avg_04s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :1, :] - prediction_new[:, :1, :], axis=2), axis=1)) * batch  # 012
            l2error_dest_04s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 0:1, :] - prediction_new[:, 0:1, :], axis=2), axis=1)) * batch  # 012
            l2error_avg_08s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :2, :] - prediction_new[:, :2, :], axis=2), axis=1)) * batch  # 024
            l2error_dest_08s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 1:2, :] - prediction_new[:, 1:2, :], axis=2), axis=1)) * batch  # 024
            l2error_avg_12s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :3, :] - prediction_new[:, :3, :], axis=2), axis=1)) * batch  # 0.036
            l2error_dest_12s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 2:3, :] - prediction_new[:, 2:3, :], axis=2),
                        axis=1)) * batch  # 0.036
            l2error_avg_16s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :4, :] - prediction_new[:, :4, :], axis=2), axis=1)) * batch  # 0.48
            l2error_dest_16s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 3:4, :] - prediction_new[:, 3:4, :], axis=2), axis=1)) * batch  # 0.48
            l2error_avg_20s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :5, :] - prediction_new[:, :5, :], axis=2), axis=1)) * batch  # 1
            l2error_dest_20s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 4:5, :] - prediction_new[:, 4:5, :], axis=2), axis=1)) * batch  # 1
            l2error_avg_24s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :6, :] - prediction_new[:, :6, :], axis=2), axis=1)) * batch
            l2error_dest_24s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 5:6, :] - prediction_new[:, 5:6, :], axis=2), axis=1)) * batch  # 1.12
            l2error_avg_28s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :7, :] - prediction_new[:, :7, :], axis=2), axis=1)) * batch
            l2error_dest_28s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 6:7, :] - prediction_new[:, 6:7, :], axis=2), axis=1)) * batch  # 1.24
            l2error_avg_32s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :8, :] - prediction_new[:, :8, :], axis=2), axis=1)) * batch  # 1.36
            l2error_dest_32s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 7:8, :] - prediction_new[:, 7:8, :], axis=2), axis=1)) * batch
            l2error_avg_36s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :9, :] - prediction_new[:, :9, :], axis=2), axis=1)) * batch
            l2error_dest_36s += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 8:9, :] - prediction_new[:, 8:9, :], axis=2), axis=1)) * batch  # 1.48
            l2error_overall += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :10, :] - prediction_new[:, :10, :], axis=2), axis=1)) * batch  # 2~!
            l2error_dest += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 9:10, :] - prediction_new[:, 9:10, :], axis=2), axis=1)) * batch

            l2error_avg_04s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :1, :] - baseline_prediction[:, :1, :], axis=2),
                        axis=1)) * batch  # 012
            l2error_dest_04s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 0:1, :] - baseline_prediction[:, 0:1, :], axis=2),
                        axis=1)) * batch  # 012
            l2error_avg_08s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :2, :] - baseline_prediction[:, :2, :], axis=2),
                        axis=1)) * batch  # 024
            l2error_dest_08s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 1:2, :] - baseline_prediction[:, 1:2, :], axis=2),
                        axis=1)) * batch  # 024
            l2error_avg_12s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :3, :] - baseline_prediction[:, :3, :], axis=2),
                        axis=1)) * batch  # 0.036
            l2error_dest_12s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 2:3, :] - baseline_prediction[:, 2:3, :], axis=2),
                        axis=1)) * batch  # 0.036
            l2error_avg_16s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :4, :] - baseline_prediction[:, :4, :], axis=2),
                        axis=1)) * batch  # 0.48
            l2error_dest_16s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 3:4, :] - baseline_prediction[:, 3:4, :], axis=2),
                        axis=1)) * batch  # 0.48
            l2error_avg_20s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :5, :] - baseline_prediction[:, :5, :], axis=2), axis=1)) * batch  # 1
            l2error_dest_20s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 4:5, :] - baseline_prediction[:, 4:5, :], axis=2),
                        axis=1)) * batch  # 1
            l2error_avg_24s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :6, :] - baseline_prediction[:, :6, :], axis=2), axis=1)) * batch
            l2error_dest_24s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 5:6, :] - baseline_prediction[:, 5:6, :], axis=2),
                        axis=1)) * batch  # 1.12
            l2error_avg_28s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :7, :] - baseline_prediction[:, :7, :], axis=2), axis=1)) * batch
            l2error_dest_28s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 6:7, :] - baseline_prediction[:, 6:7, :], axis=2),
                        axis=1)) * batch  # 1.24
            l2error_avg_32s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :8, :] - baseline_prediction[:, :8, :], axis=2),
                        axis=1)) * batch  # 1.36
            l2error_dest_32s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 7:8, :] - baseline_prediction[:, 7:8, :], axis=2), axis=1)) * batch
            l2error_avg_36s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :9, :] - baseline_prediction[:, :9, :], axis=2), axis=1)) * batch
            l2error_dest_36s_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 8:9, :] - baseline_prediction[:, 8:9, :], axis=2),
                        axis=1), ) * batch  # 1.48
            l2error_overall_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :10, :] - baseline_prediction[:, :10, :], axis=2),
                        axis=1)) * batch  # 2~!
            l2error_dest_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 9:10, :] - baseline_prediction[:, 9:10, :], axis=2), axis=1)) * batch
        else:
            ade += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :12, :] - prediction_new[:, :12, :], axis=2), axis=1)
            ) * batch
            fde += np.mean(
                np.linalg.norm(future_traj[:, 11, :] - prediction_new[:, 11, :], axis=1)
            ) * batch
            ade_base += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :12, :] - baseline_prediction[:, :12, :], axis=2), axis=1)
            ) * batch
            fde_base += np.mean(
                np.linalg.norm(future_traj[:, 11, :] - baseline_prediction[:, 11, :], axis=1)
            ) * batch
        all_num += batch
        iteration += 1

    tcc_base = np.mean(tcc_base)
    tcc_new = np.mean(tcc_new)

    if args.dataset == 'nba' or args.dataset == 'fish' or args.dataset == 'syn':
        print(all_num)
        l2error_overall /= all_num
        l2error_dest /= all_num
        l2error_avg_04s /= all_num
        l2error_dest_04s /= all_num
        l2error_avg_08s /= all_num
        l2error_dest_08s /= all_num
        l2error_avg_12s /= all_num
        l2error_dest_12s /= all_num
        l2error_avg_16s /= all_num
        l2error_dest_16s /= all_num
        l2error_avg_20s /= all_num
        l2error_dest_20s /= all_num
        l2error_avg_24s /= all_num
        l2error_dest_24s /= all_num
        l2error_avg_28s /= all_num
        l2error_dest_28s /= all_num
        l2error_avg_32s /= all_num
        l2error_dest_32s /= all_num
        l2error_avg_36s /= all_num
        l2error_dest_36s /= all_num

        l2error_overall_base /= all_num
        l2error_dest_base  /= all_num
        l2error_avg_04s_base  /= all_num
        l2error_dest_04s_base  /= all_num
        l2error_avg_08s_base  /= all_num
        l2error_dest_08s_base /= all_num
        l2error_avg_12s_base  /= all_num
        l2error_dest_12s_base  /= all_num
        l2error_avg_16s_base  /= all_num
        l2error_dest_16s_base  /= all_num
        l2error_avg_20s_base  /= all_num
        l2error_dest_20s_base  /= all_num
        l2error_avg_24s_base /= all_num
        l2error_dest_24s_base  /= all_num
        l2error_avg_28s_base  /= all_num
        l2error_dest_28s_base /= all_num
        l2error_avg_32s_base  /= all_num
        l2error_dest_32s_base /= all_num
        l2error_avg_36s_base  /= all_num
        l2error_dest_36s_base /= all_num

        results = [
            # TCC
            ("Regular", "TCC",  tcc_new),
            ("Base","TCC", tcc_base),

            # Regular
            ("Regular", "ADE 1.0s", (l2error_avg_08s + l2error_avg_12s) / 2),
            ("Regular", "ADE 2.0s", l2error_avg_20s),
            ("Regular", "ADE 3.0s", (l2error_avg_32s + l2error_avg_28s) / 2),
            ("Regular", "ADE 4.0s", l2error_overall),
            ("Regular", "FDE 1.0s", (l2error_dest_08s + l2error_dest_12s) / 2),
            ("Regular", "FDE 2.0s", l2error_dest_20s),
            ("Regular", "FDE 3.0s", (l2error_dest_28s + l2error_dest_32s) / 2),
            ("Regular", "FDE 4.0s", l2error_dest),

            # Base
            ("Base", "ADE 1.0s", (l2error_avg_08s_base + l2error_avg_12s_base) / 2),
            ("Base", "ADE 2.0s", l2error_avg_20s_base),
            ("Base", "ADE 3.0s", (l2error_avg_32s_base + l2error_avg_28s_base) / 2),
            ("Base", "ADE 4.0s", l2error_overall_base),
            ("Base", "FDE 1.0s", (l2error_dest_08s_base + l2error_dest_12s_base) / 2),
            ("Base", "FDE 2.0s", l2error_dest_20s_base),
            ("Base", "FDE 3.0s", (l2error_dest_28s_base + l2error_dest_32s_base) / 2),
            ("Base", "FDE 4.0s", l2error_dest_base),

            # Discrepancies
            ("Discrepancy", "ADE 1.0s",
             (((l2error_avg_08s_base + l2error_avg_12s_base) / 2) - (l2error_avg_08s + l2error_avg_12s) / 2)),
            ("Discrepancy %", "ADE 1.0s",
             ((((l2error_avg_08s_base + l2error_avg_12s_base) / 2) - (l2error_avg_08s + l2error_avg_12s) / 2)) / (
                     (l2error_avg_08s + l2error_avg_12s) / 2) * 100),

            ("Discrepancy", "ADE 2.0s", l2error_avg_20s_base - l2error_avg_20s),
            ("Discrepancy %", "ADE 2.0s", (l2error_avg_20s_base - l2error_avg_20s) / l2error_avg_20s * 100),

            ("Discrepancy", "ADE 3.0s",
             ((l2error_avg_32s_base + l2error_avg_28s_base) / 2) - (l2error_avg_32s + l2error_avg_28s) / 2),
            ("Discrepancy %", "ADE 3.0s",
             (((l2error_avg_32s_base + l2error_avg_28s_base) / 2) - (l2error_avg_32s + l2error_avg_28s) / 2) / (
                     (l2error_avg_32s + l2error_avg_28s) / 2) * 100),

            ("Discrepancy", "ADE 4.0s", l2error_overall_base - l2error_overall),
            ("Discrepancy %", "ADE 4.0s", (l2error_overall_base - l2error_overall) / l2error_overall * 100),

            ("Discrepancy", "FDE 1.0s",
             ((l2error_dest_08s_base + l2error_dest_12s_base) / 2) - (l2error_dest_08s + l2error_dest_12s) / 2),
            ("Discrepancy %", "FDE 1.0s",
             (((l2error_dest_08s_base + l2error_dest_12s_base) / 2) - (l2error_dest_08s + l2error_dest_12s) / 2) / (
                     (l2error_dest_08s + l2error_dest_12s) / 2) * 100),

            ("Discrepancy", "FDE 2.0s", l2error_dest_20s_base - l2error_dest_20s),
            ("Discrepancy %", "FDE 2.0s", (l2error_dest_20s_base - l2error_dest_20s) / l2error_dest_20s * 100),

            ("Discrepancy", "FDE 3.0s",
             ((l2error_dest_28s_base + l2error_dest_32s_base) / 2) - (l2error_dest_28s + l2error_dest_32s) / 2),
            ("Discrepancy %", "FDE 3.0s",
             (((l2error_dest_28s_base + l2error_dest_32s_base) / 2) - (l2error_dest_28s + l2error_dest_32s) / 2) / (
                     (l2error_dest_28s + l2error_dest_32s) / 2) * 100),

            ("Discrepancy", "FDE 4.0s", l2error_dest_base - l2error_dest),
            ("Discrepancy %", "FDE 4.0s", (l2error_dest_base - l2error_dest) / l2error_dest * 100),
        ]
    else:
        fde /= all_num
        ade /= all_num
        fde_base /= all_num
        ade_base /= all_num

        results = [
            ("Regular", "TCC", tcc_new),
            ("Base", "TCC", tcc_base),

            ("Regular", "ADE 4.8s", ade),
            ("Regular", "FDE 4.8s", fde),
            ("Base", "ADE 4.8s", ade_base),
            ("Base", "FDE 4.8s", fde_base),

            # Discrepancies
            ("Discrepancy", "ADE 4.8s", ade_base - ade),
            ("Discrepancy %", "ADE 4.8s", ((ade_base - ade) / (ade) * 100))]

    df = pd.DataFrame(results, columns=["Type", "Metric", "Value"])
    df.to_csv(f"{folder}/evaluation_FDE_results.csv", index=False)
    df.to_string(buf=open(f"{folder}/evaluation_FDE_results.txt", "w"), index=False)

    return


def non_mission_test(test_loader, args, model, constant):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    stats_path = create_stats_folder(args, constant, "groupnet-mission")

    model.eval()
    total_missions = 0
    missions_achieved = 0
    total_controlled = 0
    sim_logs = []
    all_centroids = []
    # mission_tolerance = args.mission_buffer
    mission_tolerance = constant.buffer

    all_traj = []

    all_valid_fake_centroids = []
    all_valid_fake_traj= []
    X_MIN, X_MAX = constant.X_MIN, constant.X_MAX
    Y_MIN, Y_MAX = constant.Y_MIN, constant.Y_MAX

    for sim_id in range(len(test_loader)):

        traj_sample, _, _, _ = test_loader.dataset[sim_id]  # traj_sample: [N, T, 2], missions: [N, M, 2], controlled: [max_controlled]


        traj_sample = traj_sample.unsqueeze(0).to(args.device)  # [1, N, T, 2]
        controlled = []  # valid controlled agent indices

        agents_idx = controlled.copy() # []

        iter_num = 0
        max_steps = int(args.length / 0.4)
        valid_fake_chunks = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]

        future_traj = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]

        while (future_traj.shape[1] - args.past_length < max_steps):
            with torch.no_grad():

                one_mission =  torch.zeros(1, 0, 2, device=args.device)
                prediction, _ = model.inference_simulator(traj_sample, one_mission,torch.tensor(agents_idx, device=traj_sample.device),
                                                          mission_tolerance)  # prediction: [1, N, T, 2]

            new_traj = prediction
            if args.dataset in ['sdd', 'eth']:
                fake_np = new_traj[0].cpu().numpy()  # shape: [N, T, 2]
                out_of_bounds_mask = (
                        (fake_np[:, :, 0] < X_MIN) | (fake_np[:, :, 0] > X_MAX) |
                        (fake_np[:, :, 1] < Y_MIN) | (fake_np[:, :, 1] > Y_MAX)
                )
                invalid_mask = (out_of_bounds_mask.sum(axis=1) > (fake_np.shape[1] // 3))  # shape: [N]
            else:
                invalid_mask = torch.zeros(new_traj[0].shape[0], dtype=torch.bool)

            if invalid_mask.any():  # if there is any invalid agent - dont add it to calc for fixed number of N
                pass
            else:
                valid_fake_chunks = np.concatenate((valid_fake_chunks, new_traj[0].cpu().numpy()), axis=1)  # N, T, 2

            future_traj = np.concatenate((future_traj, new_traj[0].cpu().numpy()), axis=1)
            traj_sample = new_traj[:, :, -args.past_length:, :]
            iter_num += 1

        if np.isnan(future_traj).any():
            continue

        centroids = future_traj.mean(axis=0)  # shape (T, 2)
        all_centroids.append(centroids)
        all_traj.append(future_traj)

        # print("real all_traj len", future_traj.shape[1])
        if args.dataset in ['sdd', 'eth']:
            valid_fake_centroids = valid_fake_chunks.mean(axis=0) # [20, 2], [8,2]...
            all_valid_fake_centroids.append(valid_fake_centroids)  # S, T-changing, 2
            all_valid_fake_traj.append(valid_fake_chunks)
            # print("fake all_traj len", valid_fake_chunks.shape[1])

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
            vis_predictions_no_missions(constant,future_traj, args, stats_path)



    print("starting analysis of usage")
    df_centroids = analyze_usage(
        all_trajectories=all_traj if args.dataset not in ['eth', 'sdd'] else all_valid_fake_traj,
        all_centroids=all_centroids if args.dataset not in ['eth', 'sdd'] else all_valid_fake_centroids,  # list of np.ndarray, shape [T, 2]
        field_length=constant.field if args.dataset != "sdd" else constant.field_T,
        field_width=constant.field_width if args.dataset != "sdd" else constant.field_width_T,
        num_blocks=constant.num_blocks,
        timestep_duration=0.4,
        args=args,
        folder=f'{stats_path}/no_mission',
        constant=constant,
        gen_gt_type="groupnet"
    )

    df_per_sim = pd.DataFrame(sim_logs)



    df_summary = pd.DataFrame({
        "Total Missions": [total_missions],
        "Total Controlled Agents": [total_controlled],
        "Total Missions Achieved": [missions_achieved],

    })

    print(f"\n------ Final Statistics over {len(test_loader)} simulations -----")
    print(df_summary.T)

    if args.dataset == "nba":
        df_centroids.to_csv(
            f"{stats_path}/no_mission/centroids_agents.csv")
    df_per_sim.to_csv(
        f"{stats_path}/no_mission/per_sim_stats.csv")
    df_summary.to_csv(
        f"{stats_path}/no_mission/overall_summary.csv")




def mission_test( test_loader, other_test, args,model, constant):

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    stats_path = create_stats_folder(args, constant, "groupnet-mission")

    test_one_traj(stats_path, other_test, model, args, constant)
    model.eval()
    total_missions = 0
    missions_achieved = 0
    total_controlled = 0
    full_success_count = 0
    sim_logs = []
    all_centroids = []
    # mission_tolerance = args.mission_buffer
    mission_tolerance = constant.buffer

    per_agent_mission_times = []
    all_traj = []
    all_mission_durations = []
    agent_full_success = []

    all_valid_fake_centroids = []
    all_valid_fake_traj= []
    X_MIN, X_MAX = constant.X_MIN, constant.X_MAX
    Y_MIN, Y_MAX = constant.Y_MIN, constant.Y_MAX
    visualized = False
    for sim_id in range(len(test_loader)):

        traj_sample, missions, controlled, _ = test_loader.dataset[sim_id]  # traj_sample: [N, T, 2], missions: [N, M, 2], controlled: [max_controlled]
        traj_sample = traj_sample.unsqueeze(0).to(args.device)  # [1, N, T, 2]
        missions = missions.to(args.device)  # [N, M, 2]
        controlled = controlled[controlled != -1].tolist()  # valid controlled agent indices

        agent_missions = [0] * len(controlled) # which missions collected for each agent
        agents_idx = controlled.copy()
        agents_idx_plot = controlled.copy()
        agents_targs = missions[agents_idx]  # [C, M, 2]


        mission_log = []
        target_status = {}
        iter_num = 0
        all_mission_accomplished = False
        max_steps = int(args.length / 0.4)
        valid_fake_chunks = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]

        future_traj = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]
        agent_mission_times = {a: [] for a in agents_idx}
        start_times = {a: args.past_length for a in agents_idx}


        while (future_traj.shape[1] - args.past_length < max_steps):
            with torch.no_grad():
                agent_missions_ten = torch.tensor(agent_missions, device=traj_sample.device)  # shape: [C]
                one_mission = agents_targs.gather(1, agent_missions_ten.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
                prediction, _ = model.inference_simulator(traj_sample, one_mission, torch.tensor(agents_idx, device=traj_sample.device),mission_tolerance)  # prediction: [1, N, T, 2]

            new_traj = prediction
            if args.dataset in ['sdd', 'eth']:
                fake_np = new_traj[0].cpu().numpy()  # shape: [N, T, 2]
                out_of_bounds_mask = (
                        (fake_np[:, :, 0] < X_MIN) | (fake_np[:, :, 0] > X_MAX) |
                        (fake_np[:, :, 1] < Y_MIN) | (fake_np[:, :, 1] > Y_MAX)
                )
                invalid_mask = (out_of_bounds_mask.sum(axis=1) > (fake_np.shape[1] // 3))  # shape: [N]
            else:
                invalid_mask = torch.zeros(new_traj[0].shape[0], dtype=torch.bool)

            if invalid_mask.any():  # if there is any invalid agent - dont add it to calc for fixed number of N
                pass
            else:
                valid_fake_chunks = np.concatenate((valid_fake_chunks, new_traj[0].cpu().numpy()), axis=1)  # N, T, 2
            agents_to_remove = []

            for i in reversed(range(len(agents_idx))):  # reversed to safely remove
                agent = agents_idx[i]
                mission_id = agent_missions[i]
                target = agents_targs[i, mission_id]
                agent_path = new_traj[0, agent]  # shape: [T, 2]

                #distance check (line-to-point projection)
                p1 = agent_path[:-1]
                p2 = agent_path[1:]
                seg = p2 - p1
                seg_len = seg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                proj = ((target - p1) * seg).sum(-1, keepdim=True) / (seg_len ** 2)
                proj_clamped = proj.clamp(0, 1)
                closest = p1 + proj_clamped * seg
                dists = (closest - target).norm(dim=-1)  # [T-1]
                #Locate first hit (accurate)
                hit_steps = (dists < mission_tolerance).nonzero(as_tuple=True)[0]

                if len(hit_steps) > 0:
                    line_hit = hit_steps[0].item()
                    time_hit = args.past_length + iter_num * args.future_length +line_hit
                    mission_log.append((time_hit, agent, mission_id, target.cpu().numpy()))
                    target_status[(agent, mission_id)] = True
                    agent_mission_times[agent].append((time_hit - start_times[agent])*0.4)
                    all_mission_durations.append((time_hit - start_times[agent])*0.4)
                    start_times[agent] = time_hit #starting time for next mission

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
            future_traj = np.concatenate((future_traj, new_traj[0].cpu().numpy()), axis=1)
            traj_sample = new_traj[:, :, -args.past_length:, :]


            iter_num += 1

        if np.isnan(future_traj).any():
            continue

        centroids = future_traj.mean(axis=0)  # shape (T, 2)
        all_centroids.append(centroids)
        all_traj.append(future_traj)
        
        # print("real all_traj len", future_traj.shape[1])
        if args.dataset in ['sdd', 'eth']:
            valid_fake_centroids = valid_fake_chunks.mean(axis=0) # [20, 2], [8,2]...
            all_valid_fake_centroids.append(valid_fake_centroids)  # S, T-changing, 2
            all_valid_fake_traj.append(valid_fake_chunks)
            # print("fake all_traj len", valid_fake_chunks.shape[1])

        sim_missions = len(controlled) * args.mission_num # C*missions for each agent -> total misssoin
        total_controlled += len(controlled)
        sim_achieved = len(mission_log) #total missions achieved
        total_missions += sim_missions # missions in all simulations
        missions_achieved += sim_achieved # missions achieved in all simulations
        if sim_achieved == sim_missions:
            full_success_count += 1 # in this sim all misssions achieved

        sim_success_rate = sim_achieved / sim_missions

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
            "avg_missions_per_agent": sim_achieved/ len(controlled),
            "mission_log": mission_log,
            "sim_success_rate": sim_success_rate
        })

        print("sim", sim_id)
        if len(agents_idx_plot) == args.agents_to_plot and not visualized:
            print("VIS")
            vis_predictions_missions(constant,future_traj, mission_log, target_status, args, missions.cpu(), agents_idx_plot, stats_path)
            visualized = True


    df_per_sim = pd.DataFrame(sim_logs)

    avg_per_agent_all = df_per_sim["avg_missions_per_agent"].mean()
    std_per_agent_all = df_per_sim["avg_missions_per_agent"].std()
    avg_sim_success_rate = df_per_sim["sim_success_rate"].mean()
    std_sim_success_rate = df_per_sim["sim_success_rate"].std()

    # Group by number of covert agents
    grouped = df_per_sim.groupby("controlled_agents")
    covert_stats = grouped.agg({"avg_missions_per_agent": ['mean', 'std'],"mean_mission_time": ['mean', 'std'] }).reset_index()
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
    "Avg Missions per Simulation": [total_missions / len(test_loader)],
    "Total Controlled Agents": [total_controlled],
    "Total Missions Achieved": [missions_achieved],
    "Overall Success Rate": [missions_achieved / total_missions],
    "Simulations with Full Success": [full_success_count],
    "Avg Missions Achieved per Simulation (General)": [missions_achieved / len(testing_loader)],
    "avg simimulation success rate": [avg_sim_success_rate],
    "Std simimulation success rate": [std_sim_success_rate],

    "Mean Mission Time (All Agents)":mean_mission_time,
    "Std Mission Time (All Agents)": std_mission_time,
    "Std of Mean Mission Time (Between Sims)": [df_per_sim["mean_mission_time"].std()],

    "Min Mission Time (All Agents)": min_time,
    "Min of Mean Mission Times (Between Sims)": [df_per_sim["mean_mission_time"].min()],

    "Max Mission Time (All Agents)": max_time,
    "Max of Mean Mission Times (Between Sims)": [df_per_sim["mean_mission_time"].max()],

    "Avg Missions per Agent (All Sims)": [avg_per_agent_all],
    "Std of Avg Missions per Agent (Between Sims)": [std_per_agent_all],
    })

    print(f"\n----- Final Statistics over {len(test_loader)} simulations -------")
    print(df_summary.T)


    print("starting analysis of usage")

    df_centroids = analyze_usage(
        all_trajectories=all_traj if args.dataset not in ['eth', 'sdd'] else all_valid_fake_traj,
        all_centroids=all_centroids if args.dataset not in ['eth', 'sdd'] else all_valid_fake_centroids,  # list of np.ndarray, shape [T, 2]
        field_length=constant.field if args.dataset != "sdd" else constant.field_T,
        field_width=constant.field_width if args.dataset != "sdd" else constant.field_width_T,
        num_blocks=constant.num_blocks,
        timestep_duration=0.4,
        args=args,
        folder=stats_path,
        constant=constant,
        gen_gt_type="groupnet"
    )

    if args.dataset == "nba":
        df_centroids.to_csv(f"{stats_path}/centroids_agents.csv")
    df_per_sim.to_csv(f"{stats_path}/per_sim_stats.csv")
    df_summary.to_csv(f"{stats_path}/overall_summary.csv")

    # return df_summary, df_per_sim, df_centroids





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model_names', default="mission_aware100_None_nba_train_1_1_")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_save_dir', default='saved_models/nba')
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--traj_scale', type=int, default=1)
    parser.add_argument('--sample_k', type=int, default=1)
    parser.add_argument('--past_length', type=int, default=5)
    parser.add_argument('--future_length', type=int, default=10)
    parser.add_argument('--training_type', type=str, default="test")
    parser.add_argument('--length', type=int, default=120)
    parser.add_argument('--dataset', type=str, default="nba")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--agent_num', type=int, default=11)
    parser.add_argument('--sim_num', type=int, default=100)
    parser.add_argument('--mission_num', type=int, default=12)
    parser.add_argument('--testing',action='store_true', default=False)
    parser.add_argument('--mission_buffer',type=float, default=1)
    parser.add_argument('--agent',  nargs='+', type=int, default=[0 , 2])
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument('--info', type=str, default="")
    parser.add_argument('--max_covert', type=int, default=5)
    parser.add_argument('--scene', type=str, default="")
    parser.add_argument('--learn_prior', action='store_true', default=False)
    parser.add_argument('--how_far',  type=float, default=1)
    parser.add_argument('--sdd_scene', type=int, default=None)
    parser.add_argument('--pca_comp', type=int, default=2)
    parser.add_argument('--agents_to_plot', type=int, default=1)


    args = parser.parse_args()
    """ setup """
    names = [x for x in args.model_names.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)
    args.device = device
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset == 'nba':
        args.model_save_dir = 'GM/saved_models/nba'
        args.agent_num = 11
        args.mission_num = 12
        args.length = 120
        test_dset = NBADataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=args.training_type)


    elif args.dataset == 'fish':
        args.model_save_dir = 'GM/saved_models/fish'
        args.agent_num = 8
        args.mission_num = 12
        args.length = 120
        test_dset = FISHDataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=args.training_type)


    elif args.dataset == 'syn':
        args.model_save_dir = 'GM/saved_models/syn'
        test_dset = SYNDataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=args.training_type)

    elif args.dataset == 'sdd':
        args.past_length = 8
        args.future_length = 12
        args.mission_num = 8
        args.length = 60
        args.model_save_dir = 'GM/saved_models/sdd'
        test_dset = TrajectoryDatasetSDD(data_dir="datasets/SDD", obs_len=args.past_length,
                                         pred_len=args.future_length, skip=1,
                                         min_ped=1, delim='space', save_path="datasets/sdd/SDD.pt",
                                         mode=args.training_type)

    elif args.dataset == 'eth':
        args.past_length = 8
        args.future_length = 12
        args.model_save_dir = 'GM/saved_models/eth'
        test_dset = TrajectoryDatasetETH(data_dir="datasets/eth", obs_len=args.past_length,
                                         pred_len=args.future_length, skip=1, min_ped=1,
                                         delim='space', test_scene=args.scene, save_path=args.scene,
                                         mode=args.training_type)
    SEED = args.seed
    g = torch.Generator()
    g.manual_seed(SEED)

    if args.dataset == 'nba' or args.dataset == 'fish' or args.dataset == 'syn':
        test_loader = DataLoader(
            test_dset,
            batch_size=args.batch_size*32,
            shuffle=False,
            num_workers=0,
            collate_fn=seq_collate,
            pin_memory=True,generator=g,)

    else:
        test_loader = DataLoader(test_dset, batch_sampler=GroupedBatchSampler(test_dset.grouped_seq_indices_test,
                                                                               batch_size=args.batch_size*32, shuffle=False,
                                                                               drop_last=False), collate_fn=seq_collate,generator=g)

    for name in names:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        """ model """
        saved_path = os.path.join(args.model_save_dir,str(name)+'.p')
        print('load model from:',saved_path)
        checkpoint = torch.load(saved_path, map_location='cpu')
        training_args = checkpoint['model_cfg']

        model = GroupNetM(training_args,device)
        model.set_device(device)
        model.eval()
        model.load_state_dict(checkpoint['model_dict'], strict=True)
        model.args.sample_k = args.sample_k

        if args.dataset in ['nba', 'syn', 'fish']:
            traj_dataset = torch.utils.data.Subset(test_dset, range(args.sim_num))
            if args.dataset == 'nba':
                constant = ConstantNBA
            elif args.dataset == 'fish':
                constant = ConstantFish
            elif args.dataset == 'syn' and args.training_type == 'test_rigid':
                constant = ConstantSYNR
            elif args.dataset == 'syn' and args.training_type != 'test_rigid':
                constant = ConstantSYNS

        else:
            if args.dataset == 'eth':
                constant = return_the_eth_scene(args.scene)
            else:
                constant = return_the_sdd_scene(args.sdd_scene)
            traj_dataset = Subset(test_dset, constant.n_agents, args.sim_num, args.sdd_scene)
        data = prepare_target(args, args.device, traj_dataset, constant, "test")

        wrapped_test = MissionTestDataset(traj_dataset, data)
        testing_loader = DataLoader(
            wrapped_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,generator=g,
        )

        other_testing_loader = DataLoader(
            wrapped_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,generator=g,
        )

        mission_test(testing_loader,other_testing_loader, args, model, constant)
        print("starting non mission test")
        non_mission_test(testing_loader, args, model, constant)
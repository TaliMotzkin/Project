import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from data.dataloader_nba import NBADataset
from data.dataloader_fish import FISHDataset
from data.dataloader_syn import SYNDataset
from config import parse_args
from model.models_sampler import Sampler, SamplerMLP

from torch.utils.data import DataLoader
from model.GroupNet_nba import GroupNet
from loss.loss_sampler import LossCompute
from data.dataloader_GAN import seq_collate_sampler, TrajectoryDatasetFlexN, TrajectoryDatasetNonFlexN
import time
from utilis import *
from data.dataloader_SDD import TrajectoryDatasetSDD
from data.dataloader_ETH import TrajectoryDatasetETH, GroupedBatchSampler, seq_collate




def create_traj_flex_N(test_loader, model, args):
    past_list, fut_list, old_fut_list = [], [], []
    group_list = []
    ew_list, ef_list, dir_list, vel_list, vis_list, idx_list = [], [], [], [], [], []
    iter = 0
    print("test_loader", len(test_loader))
    for data in test_loader:
        B, N, _, _ = data['past_traj'].shape
        # print("data['past_traj'].shape", data['past_traj'].shape)

        with torch.no_grad():
            # (20, B*N, T, 2)
            pred20, _ = model.inference(data)

            vis, vel, dirc, e_f, e_w = compute_features(args, data['past_traj'], "past")

            # best‐indexes per sample
            y = data['future_traj'].reshape(B*N, args.future_length, 2).unsqueeze(0).repeat(20,1,1,1).to(args.device)
            errs = torch.norm(y - pred20, dim=3).mean(dim=2)        # (20, B*N)
            best = errs.argmin(dim=0).view(B, N)                        # (B, N)

            pred_per_scene  = (pred20.permute(1, 2, 3, 0).reshape(B, N, args.future_length, 2, 20))

            past_list.extend(data['past_traj'].cpu() )
            fut_list.extend(data['future_traj'].cpu() )
            group_list.extend(pred_per_scene.cpu() )
            ew_list.extend(e_w.cpu() )
            ef_list.extend(e_f.cpu() )
            dir_list.extend(dirc.cpu() )
            vel_list.extend(vel.cpu() )
            vis_list.extend(vis.cpu() )
            idx_list.extend(best.cpu() )

            iter += 1
            if iter % 100 == 0:
                print("iteration:", iter, "in data creation")

    seq_start_end = []
    grouped_seq_indices = defaultdict(list)
    agent_counter = 0
    for scene_i, traj in enumerate(past_list):
        Ni = traj.shape[0]
        start = agent_counter
        end = start + Ni
        seq_start_end.append((start, end))
        grouped_seq_indices[Ni].append(scene_i)
        agent_counter = end

    device = args.device
    all_past = torch.cat(past_list, dim=0).to(device)  # (sum Ni, T, 2)
    all_future = torch.cat(fut_list, dim=0).to(device)  # (sum Ni, T, 2)
    all_groupnet = torch.cat(group_list, dim=0).to(device)  # (sum Ni, T,2,20)
    all_edge_w = torch.cat(ew_list, dim=0).to(device)  # (sum Ni,)
    all_edge_f = torch.cat(ef_list, dim=0).to(device)
    all_dir = torch.cat(dir_list, dim=0).to(device)  # (sum Ni,T)
    all_vel = torch.cat(vel_list, dim=0).to(device)
    all_vis = vis_list  # (S, N, N)
    all_indices = torch.cat(idx_list, dim=0).to(device)  # (sum Ni,)

    traj_dataset = TrajectoryDatasetFlexN(all_past, all_future,all_groupnet, all_edge_w, all_edge_f,
        all_dir, all_vel, all_vis, all_indices , grouped_seq_indices,seq_start_end)

    return traj_dataset


def create_traj(test_loader, model, args):

    iter = 0
    actor_num = args.agent_num
    past_traj = torch.empty((0, actor_num, args.past_length, 2)).to(args.device)  # total samples, 8 agents, 5 time steps, 2 coords
    group_net = torch.empty((0, actor_num, args.future_length, 2, 20)).to(args.device)  # (Total_samples, N, 10, 2, 20)
    future_traj = torch.empty((0, actor_num, args.future_length, 2)).to(args.device)

    edge_weights_past_list = torch.empty((0, actor_num)).to(args.device)
    edge_features_past_list = torch.empty((0, actor_num)).to(args.device)
    direction_past_list = torch.empty((0, actor_num, args.past_length-1)).to(args.device)
    velocity_past_list = torch.empty((0, actor_num, args.past_length -1)).to(args.device)
    visability_mat_past_list = torch.empty((0, actor_num, actor_num)).to(args.device)


    indexes_list = torch.empty((0, actor_num)).to(args.device)

    print("data length", len(test_loader))
    for data in test_loader:

        with torch.no_grad():
            prediction, H = model.inference(data)  # (20, B*N,10,2)

            visability_mat_past, velocity_past, direction_past, edge_features_past, edge_weights_past = compute_features(args,
                data['past_traj'], "past")

            y = data['future_traj'].reshape(data['future_traj'].shape[0] * actor_num, args.future_length,2)  # 200, 10, 2
            y = y.unsqueeze(0).repeat(20, 1, 1, 1).to(args.device)
            error = torch.norm(y - prediction, dim=3).mean(dim=2)  # 20, BN,
            indices = torch.argmin(error, dim=0)  # BN


        indexes_list = torch.cat((indexes_list, indices.view(data['future_traj'].shape[0], actor_num)), dim=0)

        edge_weights_past_list = torch.cat((edge_weights_past_list, edge_weights_past.to(args.device)), dim=0)
        edge_features_past_list = torch.cat((edge_features_past_list, edge_features_past.to(args.device)), dim=0)
        direction_past_list = torch.cat((direction_past_list, direction_past.to(args.device)), dim=0)
        velocity_past_list = torch.cat((velocity_past_list, velocity_past.to(args.device)), dim=0)
        visability_mat_past_list = torch.cat((visability_mat_past_list, visability_mat_past.to(args.device)), dim=0)



        future_traj = torch.cat((future_traj, data['future_traj'].to(args.device)), dim=0)
        prediction = prediction.permute(1, 2, 3, 0).view(data['future_traj'].shape[0], actor_num, 10, 2, 20)
        group_net = torch.cat((group_net, prediction.to(args.device)), dim=0)
        past_traj = torch.cat((past_traj, data['past_traj'].to(args.device)), dim=0)

        iter += 1
        if iter % 100 == 0:
            print("iteration:", iter, "in data creation")
    # print("future_traj", future_traj.shape, "past_traj", past_traj.shape, "group_net", group_net.shape, "H_list", H_list.shape)
    traj_dataset = TrajectoryDatasetNonFlexN(past_traj, future_traj, group_net, edge_weights_past_list, edge_features_past_list,
                                         direction_past_list, velocity_past_list, visability_mat_past_list,indexes_list)

    return traj_dataset



def train(train_loader, args, S, writer):
    lossfn = LossCompute(args, S)
    optimizer_S = torch.optim.Adam(S.parameters(), lr=args.lr)
    scheduler_S = torch.optim.lr_scheduler.StepLR(optimizer_S, step_size=args.lr_step, gamma=args.lr_gamma)

    # Track losses
    train_losses = []

    mid_epoch = int(args.epoch / 2)
    for i in range(args.epoch):
        iter_num = 0

        S.train()
        train_loss = 0
        num_batches = 0
        pred_loss_list = 0
        score_lost_list = 0
        time_epoch = time.time()

        for batch in train_loader:
            past = batch['past_traj'].to(args.device)
            prediction = batch['group_net'].to(args.device)
            future_traj = batch['future_traj'].to(args.device)

            edge_weights_past = batch['edge_weights_past'].to(args.device)
            edge_features_past = batch['edge_features_past'].to(args.device)
            direction_past = batch['direction_past'].to(args.device)
            velocity_past = batch['velocity_past'].to(args.device)
            visability_mat_past = batch['visability_mat_past'].to(args.device)

            indices = batch["indexes_list"].to(args.device)

            optimizer_S.zero_grad()
            total_loss, pred_loss, loss_score = lossfn.compute_sampler_loss(i, mid_epoch,
                                                                                                 future_traj, past,
                                                                                                 edge_weights_past,
                                                                                                 edge_features_past,
                                                                                                 direction_past,
                                                                                                 velocity_past,
                                                                                                 visability_mat_past,
                                                                                                 prediction, indices)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(S.parameters(), max_norm=5.0)
            optimizer_S.step()

            iter_num += 1
            num_batches += 1
            train_loss += total_loss.item()
            pred_loss_list += pred_loss.item()
            score_lost_list += loss_score.item()
            if iter_num % args.iternum_print == 0:
                print(
                    'Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Total loss: {:03f}| Loss_pred: {:03f}| loss_score: {:03f}'
                    .format(i, args.epoch, iter_num, len(train_loader), total_loss.item(), pred_loss.item(), loss_score.item() ))
        time_end = time.time()
        writer.add_scalar("Loss/Total", train_loss / num_batches, i)
        writer.add_scalar("Loss/Prediction", pred_loss_list / num_batches, i)
        writer.add_scalar("Loss/Score", score_lost_list / num_batches, i)

        scheduler_S.step()
        train_losses.append(train_loss / num_batches)

        # save model checkpoint every epoch
        if (i + 1) % args.save_every == 0:
            saveModel_S(S, args, str(i + 1))

        print(
            f"Epoch [{i + 1}/{args.epoch}] - Train Loss S: {train_losses[-1]:.4f}, Time: {time_end - time_epoch}")
        writer.close()
    plot_losses_gen(args, train_losses)


def plot_losses_gen(args, train_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Sampler Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Sampler Loss')
    plt.legend()
    plt.title('Sampler Loss Progression')
    plt.savefig(f"Sampler/plots/Sampler_Loss_{args.dataset}_{args.seed}_{args.timestamp}_{args.test_mlp}.png")
    # plt.show()





def testing_ADE_FDE(constant,test_loader, args, S, folder, model):
    S.eval()
    model.eval()

    iteration = 0

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
    l2error_avg_28s_base = 0
    l2error_dest_28s_base = 0
    l2error_avg_32s_base = 0
    l2error_dest_32s_base = 0
    l2error_avg_36s_base = 0
    l2error_dest_36s_base = 0

    l2error_overall_group = 0
    l2error_dest_group = 0
    l2error_avg_04s_group = 0
    l2error_dest_04s_group = 0
    l2error_avg_08s_group = 0
    l2error_dest_08s_group = 0
    l2error_avg_12s_group = 0
    l2error_dest_12s_group = 0
    l2error_avg_16s_group = 0
    l2error_dest_16s_group = 0
    l2error_avg_20s_group = 0
    l2error_dest_20s_group = 0
    l2error_avg_24s_group = 0
    l2error_dest_24s_group = 0
    l2error_avg_28s_group = 0
    l2error_dest_28s_group = 0
    l2error_avg_32s_group = 0
    l2error_dest_32s_group = 0
    l2error_avg_36s_group = 0
    l2error_dest_36s_group = 0

    l2error_overall_selected = 0
    l2error_dest_selected = 0
    l2error_avg_04s_selected = 0
    l2error_dest_04s_selected = 0
    l2error_avg_08s_selected = 0
    l2error_dest_08s_selected = 0
    l2error_avg_12s_selected = 0
    l2error_dest_12s_selected = 0
    l2error_avg_16s_selected = 0
    l2error_dest_16s_selected = 0
    l2error_avg_20s_selected = 0
    l2error_dest_20s_selected = 0
    l2error_avg_24s_selected = 0
    l2error_dest_24s_selected = 0
    l2error_avg_28s_selected = 0
    l2error_dest_28s_selected = 0
    l2error_avg_32s_selected = 0
    l2error_dest_32s_selected = 0
    l2error_avg_36s_selected = 0
    l2error_dest_36s_selected = 0

    ade = 0
    fde = 0
    ade_base = 0
    fde_base = 0
    ade_group = 0
    fde_group = 0
    ade_selected = 0
    fde_selected = 0

    tcc_base = []
    tcc_new = []
    tcc_group = []
    tcc_selected = []

    for data in test_loader:
        past = data['past_traj'].to(args.device)
        prediction = data['group_net'].to(args.device)
        future_traj = data['future_traj'].to(args.device)


        edge_weights_past = data['edge_weights_past'].to(args.device)
        edge_features_past = data['edge_features_past'].to(args.device)
        direction_past = data['direction_past'].to(args.device)
        velocity_past = data['velocity_past'].to(args.device)
        visability_mat_past = data['visability_mat_past'].to(args.device)


        with torch.no_grad():
            prediction_new, indixes = S.inference(past, visability_mat_past, velocity_past, direction_past,
                                                  edge_features_past,
                                                  edge_weights_past, prediction)  # B, N, 10, 2


        batch = future_traj.shape[0]
        actor_num = future_traj.shape[1]
        future_traj = future_traj.view(future_traj.shape[0] * future_traj.shape[1], future_traj.shape[2],
                                       future_traj.shape[3]).cpu().numpy()
        prediction_new =prediction_new.view(prediction_new.shape[0] * prediction_new.shape[1], prediction_new.shape[2],
                                       prediction_new.shape[3]).cpu().numpy()





        # indixes = torch.randint(0, 20, (batch, actor_num)).to(args.device) #for random choice check

        indices_expanded = indixes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.future_length, 2, 1)

        past_traj = data['past_traj'].detach().cpu().numpy()


        if args.dataset == "sdd":
            # print("prediction range before:", prediction.min(), prediction.max())

            normalized_future_traj = future_traj
            future_traj = denormalize_traj(future_traj)
            normalized_prediction_new = prediction_new
            prediction_new = denormalize_traj(prediction_new)
            normalized_past_traj = past_traj
            past_traj = denormalize_traj(past_traj)
            prediction = denormalize_traj(prediction, "prediction") # (Total_samples, N, 10, 2, 20)

            # print("prediction range:", prediction.min(), prediction.max())
            # print("future_traj range:", future_traj.min(), future_traj.max())

        else:
            normalized_future_traj = future_traj
            normalized_prediction_new = prediction_new
            normalized_past_traj = past_traj

        last_5_steps = past_traj[:, :, -5:, :]
        avg_velocity = np.mean(np.diff(last_5_steps, axis=2), axis=2, keepdims=True)
        last_position = last_5_steps[:, :, -1:, :]
        baseline_prediction = np.concatenate(
            [last_position + i * avg_velocity for i in range(1,  args.future_length +1)], axis=2
        )
        BN = batch * actor_num
        baseline_prediction = baseline_prediction.reshape(BN,  args.future_length, 2)
        selected = torch.gather(prediction, dim=4, index=indices_expanded).squeeze(-1).view(BN,  args.future_length, 2).detach().cpu().numpy()
        # print("selected range:", selected.min(), selected.max())

        y = future_traj
        y = y[None].repeat(20, axis=0)
        # print("prediction", prediction.shape)
        prediction = np.array(prediction.permute(4,0,1,2,3).view(20,BN,  args.future_length, 2).detach().cpu().numpy())


        # print("pred_20", pred_20[0,0])
        # print("prediction from data", prediction[0,0])

        if iteration == 0:
            future_ploting = np.reshape(normalized_future_traj, (batch, actor_num, args.future_length, 2))
            previous_3D = np.reshape(normalized_past_traj, (batch, actor_num, args.past_length, 2))
            best = np.reshape(normalized_prediction_new, (batch, actor_num, args.future_length, 2))

            draw_result(constant,folder, args, best, previous_3D)
            draw_result(constant,folder, args, future_ploting, previous_3D, mode='gt')

        tcc_new.extend(calc_TCC(prediction_new, future_traj, 1))
        tcc_base.extend(calc_TCC(baseline_prediction, future_traj, 1))
        tcc_group.extend(calc_TCC(prediction, future_traj, 20))
        tcc_selected.extend(calc_TCC(selected, future_traj, 1))

        if args.dataset == 'nba' or args.dataset == "fish" or args.dataset == 'syn':
            l2error_avg_04s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, :1, :] - prediction[:, :, :1, :], axis=3), axis=2),
                       axis=0)) * batch  # 012
            l2error_dest_04s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, 0:1, :] - prediction[:, :, 0:1, :], axis=3), axis=2),
                       axis=0)) * batch  # 012
            l2error_avg_08s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, :2, :] - prediction[:, :, :2, :], axis=3), axis=2),
                       axis=0)) * batch  # 024
            l2error_dest_08s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, 1:2, :] - prediction[:, :, 1:2, :], axis=3), axis=2),
                       axis=0)) * batch  # 024
            l2error_avg_12s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, :3, :] - prediction[:, :, :3, :], axis=3), axis=2),
                       axis=0)) * batch  # 0.036
            l2error_dest_12s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, 2:3, :] - prediction[:, :, 2:3, :], axis=3), axis=2),
                       axis=0)) * batch  # 0.036
            l2error_avg_16s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, :4, :] - prediction[:, :, :4, :], axis=3), axis=2),
                       axis=0)) * batch  # 0.48
            l2error_dest_16s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, 3:4, :] - prediction[:, :, 3:4, :], axis=3), axis=2),
                       axis=0)) * batch  # 0.48
            l2error_avg_20s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, :5, :] - prediction[:, :, :5, :], axis=3), axis=2),
                       axis=0)) * batch  # 1
            l2error_dest_20s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, 4:5, :] - prediction[:, :, 4:5, :], axis=3), axis=2),
                       axis=0)) * batch  # 1
            l2error_avg_24s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, :6, :] - prediction[:, :, :6, :], axis=3), axis=2), axis=0)) * batch
            l2error_dest_24s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, 5:6, :] - prediction[:, :, 5:6, :], axis=3), axis=2),
                       axis=0)) * batch  # 1.12
            l2error_avg_28s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, :7, :] - prediction[:, :, :7, :], axis=3), axis=2), axis=0)) * batch
            l2error_dest_28s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, 6:7, :] - prediction[:, :, 6:7, :], axis=3), axis=2),
                       axis=0)) * batch  # 1.24
            l2error_avg_32s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, :8, :] - prediction[:, :, :8, :], axis=3), axis=2),
                       axis=0)) * batch  # 1.36
            l2error_dest_32s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, 7:8, :] - prediction[:, :, 7:8, :], axis=3), axis=2), axis=0)) * batch
            l2error_avg_36s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, :9, :] - prediction[:, :, :9, :], axis=3), axis=2), axis=0)) * batch
            l2error_dest_36s_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, 8:9, :] - prediction[:, :, 8:9, :], axis=3), axis=2),
                       axis=0)) * batch  # 1.48
            l2error_overall_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, :10, :] - prediction[:, :, :10, :], axis=3), axis=2),
                       axis=0)) * batch  # 2~!
            l2error_dest_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, 9:10, :] - prediction[:, :, 9:10, :], axis=3), axis=2),
                       axis=0)) * batch


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
                np.mean(np.linalg.norm(future_traj[:, 2:3, :] - prediction_new[:, 2:3, :], axis=2), axis=1)) * batch  # 0.036
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

            l2error_avg_04s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :1, :] - selected[:, :1, :], axis=2), axis=1)) * batch  # 012
            l2error_dest_04s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 0:1, :] - selected[:, 0:1, :], axis=2), axis=1)) * batch  # 012
            l2error_avg_08s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :2, :] - selected[:, :2, :], axis=2), axis=1)) * batch  # 024
            l2error_dest_08s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 1:2, :] - selected[:, 1:2, :], axis=2), axis=1)) * batch  # 024
            l2error_avg_12s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :3, :] - selected[:, :3, :], axis=2), axis=1)) * batch  # 0.036
            l2error_dest_12s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 2:3, :] - selected[:, 2:3, :], axis=2),
                        axis=1)) * batch  # 0.036
            l2error_avg_16s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :4, :] - selected[:, :4, :], axis=2), axis=1)) * batch  # 0.48
            l2error_dest_16s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 3:4, :] - selected[:, 3:4, :], axis=2), axis=1)) * batch  # 0.48
            l2error_avg_20s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :5, :] - selected[:, :5, :], axis=2), axis=1)) * batch  # 1
            l2error_dest_20s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 4:5, :] - selected[:, 4:5, :], axis=2), axis=1)) * batch  # 1
            l2error_avg_24s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :6, :] - selected[:, :6, :], axis=2), axis=1)) * batch
            l2error_dest_24s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 5:6, :] - selected[:, 5:6, :], axis=2), axis=1)) * batch  # 1.12
            l2error_avg_28s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :7, :] - selected[:, :7, :], axis=2), axis=1)) * batch
            l2error_dest_28s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 6:7, :] - selected[:, 6:7, :], axis=2), axis=1)) * batch  # 1.24
            l2error_avg_32s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :8, :] - selected[:, :8, :], axis=2), axis=1)) * batch  # 1.36
            l2error_dest_32s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 7:8, :] - selected[:, 7:8, :], axis=2), axis=1)) * batch
            l2error_avg_36s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :9, :] - selected[:, :9, :], axis=2), axis=1)) * batch
            l2error_dest_36s_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 8:9, :] - selected[:, 8:9, :], axis=2), axis=1)) * batch  # 1.48
            l2error_overall_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :10, :] - selected[:, :10, :], axis=2), axis=1)) * batch  # 2~!
            l2error_dest_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, 9:10, :] - selected[:, 9:10, :], axis=2), axis=1)) * batch


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

            ade_group += np.mean(
            np.min(np.mean(np.linalg.norm(y[:, :, :12, :] - prediction[:, :, :12, :], axis=3), axis=2),
                   axis=0)) * batch
            fde_group += np.mean(
                np.min(np.mean(np.linalg.norm(y[:, :, 11:12, :] - prediction[:, :,  11:12, :], axis=3), axis=2),
                       axis=0)) * batch
            ade_selected += np.mean(
                np.mean(np.linalg.norm(future_traj[:, :12, :] - selected[:, :12, :], axis=2), axis=1)
            ) * batch
            fde_selected += np.mean(
                np.linalg.norm(future_traj[:, 11, :] - selected[:, 11, :], axis=1)
            ) * batch

        all_num += batch
        iteration += 1

    tcc_base = np.mean(tcc_base)
    tcc_group = np.mean(tcc_group)
    tcc_new = np.mean(tcc_new)
    tcc_selected = np.mean(tcc_selected)
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
        l2error_dest_base /= all_num
        l2error_avg_04s_base /= all_num
        l2error_dest_04s_base /= all_num
        l2error_avg_08s_base /= all_num
        l2error_dest_08s_base /= all_num
        l2error_avg_12s_base /= all_num
        l2error_dest_12s_base /= all_num
        l2error_avg_16s_base /= all_num
        l2error_dest_16s_base /= all_num
        l2error_avg_20s_base /= all_num
        l2error_dest_20s_base /= all_num
        l2error_avg_24s_base /= all_num
        l2error_dest_24s_base /= all_num
        l2error_avg_28s_base /= all_num
        l2error_dest_28s_base /= all_num
        l2error_avg_32s_base /= all_num
        l2error_dest_32s_base /= all_num
        l2error_avg_36s_base /= all_num
        l2error_dest_36s_base /= all_num

        l2error_overall_group /= all_num
        l2error_dest_group /= all_num
        l2error_avg_04s_group /= all_num
        l2error_dest_04s_group /= all_num
        l2error_avg_08s_group /= all_num
        l2error_dest_08s_group /= all_num
        l2error_avg_12s_group /= all_num
        l2error_dest_12s_group /= all_num
        l2error_avg_16s_group /= all_num
        l2error_dest_16s_group /= all_num
        l2error_avg_20s_group /= all_num
        l2error_dest_20s_group /= all_num
        l2error_avg_24s_group /= all_num
        l2error_dest_24s_group /= all_num
        l2error_avg_28s_group /= all_num
        l2error_dest_28s_group /= all_num
        l2error_avg_32s_group /= all_num
        l2error_dest_32s_group /= all_num
        l2error_avg_36s_group /= all_num
        l2error_dest_36s_group /= all_num

        l2error_overall_selected /= all_num
        l2error_dest_selected /= all_num
        l2error_avg_04s_selected /= all_num
        l2error_dest_04s_selected /= all_num
        l2error_avg_08s_selected /= all_num
        l2error_dest_08s_selected /= all_num
        l2error_avg_12s_selected /= all_num
        l2error_dest_12s_selected /= all_num
        l2error_avg_16s_selected /= all_num
        l2error_dest_16s_selected /= all_num
        l2error_avg_20s_selected /= all_num
        l2error_dest_20s_selected /= all_num
        l2error_avg_24s_selected /= all_num
        l2error_dest_24s_selected /= all_num
        l2error_avg_28s_selected /= all_num
        l2error_dest_28s_selected /= all_num
        l2error_avg_32s_selected /= all_num
        l2error_dest_32s_selected /= all_num
        l2error_avg_36s_selected /= all_num
        l2error_dest_36s_selected /= all_num

        results = [
            ("Regular", "TCC",  tcc_new),
            ("Base","TCC", tcc_base),
            ("GroupNet","TCC", tcc_group),
            ("Selected","TCC", tcc_selected),

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

            # Group
            ("GroupNet", "ADE 1.0s", (l2error_avg_08s_group + l2error_avg_12s_group) / 2),
            ("GroupNet", "ADE 2.0s", l2error_avg_20s_group),
            ("GroupNet", "ADE 3.0s", (l2error_avg_32s_group + l2error_avg_28s_group) / 2),
            ("GroupNet", "ADE 4.0s", l2error_overall_group),
            ("GroupNet", "FDE 1.0s", (l2error_dest_08s_group + l2error_dest_12s_group) / 2),
            ("GroupNet", "FDE 2.0s", l2error_dest_20s_group),
            ("GroupNet", "FDE 3.0s", (l2error_dest_28s_group + l2error_dest_32s_group) / 2),
            ("GroupNet", "FDE 4.0s", l2error_dest_group),

            # Selected
            ("Selected", "ADE 1.0s", (l2error_avg_08s_selected + l2error_avg_12s_selected) / 2),
            ("Selected", "ADE 2.0s", l2error_avg_20s_selected),
            ("Selected", "ADE 3.0s", (l2error_avg_32s_selected + l2error_avg_28s_selected) / 2),
            ("Selected", "ADE 4.0s", l2error_overall_selected),
            ("Selected", "FDE 1.0s", (l2error_dest_08s_selected + l2error_dest_12s_selected) / 2),
            ("Selected", "FDE 2.0s", l2error_dest_20s_selected),
            ("Selected", "FDE 3.0s", (l2error_dest_28s_selected + l2error_dest_32s_selected) / 2),
            ("Selected", "FDE 4.0s", l2error_dest_selected),

            # Discrepancies
            ("Discrepancy regular", "ADE 1.0s",
             (((l2error_avg_08s_base + l2error_avg_12s_base) / 2) - (l2error_avg_08s + l2error_avg_12s) / 2)),
            ("Discrepancy regular %", "ADE 1.0s",
             ((((l2error_avg_08s_base + l2error_avg_12s_base) / 2) - (l2error_avg_08s + l2error_avg_12s) / 2)) / (
                     (l2error_avg_08s + l2error_avg_12s) / 2) * 100),

            ("Discrepancy regular", "ADE 2.0s", l2error_avg_20s_base - l2error_avg_20s),
            ("Discrepancy regular %", "ADE 2.0s", (l2error_avg_20s_base - l2error_avg_20s) / l2error_avg_20s * 100),

            ("Discrepancy regular", "ADE 3.0s",
             ((l2error_avg_32s_base + l2error_avg_28s_base) / 2) - (l2error_avg_32s + l2error_avg_28s) / 2),
            ("Discrepancy regular %", "ADE 3.0s",
             (((l2error_avg_32s_base + l2error_avg_28s_base) / 2) - (l2error_avg_32s + l2error_avg_28s) / 2) / (
                     (l2error_avg_32s + l2error_avg_28s) / 2) * 100),

            ("Discrepancy regular", "ADE 4.0s", l2error_overall_base - l2error_overall),
            ("Discrepancy regular %", "ADE 4.0s", (l2error_overall_base - l2error_overall) / l2error_overall * 100),

            ("Discrepancy regular", "FDE 1.0s",
             ((l2error_dest_08s_base + l2error_dest_12s_base) / 2) - (l2error_dest_08s + l2error_dest_12s) / 2),
            ("Discrepancy regular %", "FDE 1.0s",
             (((l2error_dest_08s_base + l2error_dest_12s_base) / 2) - (l2error_dest_08s + l2error_dest_12s) / 2) / (
                     (l2error_dest_08s + l2error_dest_12s) / 2) * 100),

            ("Discrepancy regular", "FDE 2.0s", l2error_dest_20s_base - l2error_dest_20s),
            ("Discrepancy regular %", "FDE 2.0s", (l2error_dest_20s_base - l2error_dest_20s) / l2error_dest_20s * 100),

            ("Discrepancy regular", "FDE 3.0s",
             ((l2error_dest_28s_base + l2error_dest_32s_base) / 2) - (l2error_dest_28s + l2error_dest_32s) / 2),
            ("Discrepancy regular %", "FDE 3.0s",
             (((l2error_dest_28s_base + l2error_dest_32s_base) / 2) - (l2error_dest_28s + l2error_dest_32s) / 2) / (
                     (l2error_dest_28s + l2error_dest_32s) / 2) * 100),

            ("Discrepancy regular", "FDE 4.0s", l2error_dest_base - l2error_dest),
            ("Discrepancy regular %", "FDE 4.0s", (l2error_dest_base - l2error_dest) / l2error_dest * 100),

            #desc groupnet
            ("Discrepancy groupnet", "ADE 1.0s",
             (((l2error_avg_08s_base + l2error_avg_12s_base) / 2) - (l2error_avg_08s_group + l2error_avg_12s_group) / 2)),
            ("Discrepancy groupnet %", "ADE 1.0s",
             ((((l2error_avg_08s_base + l2error_avg_12s_base) / 2) - (l2error_avg_08s_group + l2error_avg_12s_group) / 2)) / (
                     (l2error_avg_08s_group + l2error_avg_12s_group) / 2) * 100),

            ("Discrepancy groupnet", "ADE 2.0s", l2error_avg_20s_base - l2error_avg_20s_group),
            ("Discrepancy groupnet %", "ADE 2.0s", (l2error_avg_20s_base - l2error_avg_20s_group) / l2error_avg_20s_group * 100),

            ("Discrepancy groupnet", "ADE 3.0s",
             ((l2error_avg_32s_base + l2error_avg_28s_base) / 2) - (l2error_avg_32s_group + l2error_avg_28s_group) / 2),
            ("Discrepancy groupnet %", "ADE 3.0s",
             (((l2error_avg_32s_base + l2error_avg_28s_base) / 2) - (l2error_avg_32s_group + l2error_avg_28s_group) / 2) / (
                     (l2error_avg_32s_group + l2error_avg_28s_group) / 2) * 100),

            ("Discrepancy groupnet", "ADE 4.0s", l2error_overall_base - l2error_overall_group),
            ("Discrepancy groupnet %", "ADE 4.0s", (l2error_overall_base - l2error_overall_group) / l2error_overall_group * 100),

            ("Discrepancy groupnet", "FDE 1.0s",
             ((l2error_dest_08s_base + l2error_dest_12s_base) / 2) - (l2error_dest_08s_group + l2error_dest_12s_group) / 2),
            ("Discrepancy groupnet %", "FDE 1.0s",
             (((l2error_dest_08s_base + l2error_dest_12s_base) / 2) - (l2error_dest_08s_group + l2error_dest_12s_group) / 2) / (
                     (l2error_dest_08s_group + l2error_dest_12s_group) / 2) * 100),

            ("Discrepancy groupnet", "FDE 2.0s", l2error_dest_20s_base - l2error_dest_20s_group),
            ("Discrepancy groupnet %", "FDE 2.0s", (l2error_dest_20s_base - l2error_dest_20s_group) / l2error_dest_20s_group * 100),

            ("Discrepancy groupnet", "FDE 3.0s",
             ((l2error_dest_28s_base + l2error_dest_32s_base) / 2) - (l2error_dest_28s_group + l2error_dest_32s_group) / 2),
            ("Discrepancy groupnet %", "FDE 3.0s",
             (((l2error_dest_28s_base + l2error_dest_32s_base) / 2) - (l2error_dest_28s_group + l2error_dest_32s_group) / 2) / (
                     (l2error_dest_28s_group + l2error_dest_32s_group) / 2) * 100),

            ("Discrepancy groupnet", "FDE 4.0s", l2error_dest_base - l2error_dest_group),
            ("Discrepancy groupnet %", "FDE 4.0s", (l2error_dest_base - l2error_dest_group) / l2error_dest_group * 100),
            ######### discrepancies selected
            ("Discrepancy", "ADE 1.0s",
             (((l2error_avg_08s_base + l2error_avg_12s_base) / 2) - (l2error_avg_08s_selected + l2error_avg_12s_selected) / 2)),
            ("Discrepancy %", "ADE 1.0s",
             ((((l2error_avg_08s_base + l2error_avg_12s_base) / 2) - (l2error_avg_08s_selected + l2error_avg_12s_selected) / 2)) / (
                     (l2error_avg_08s_selected + l2error_avg_12s_selected) / 2) * 100),

            ("Discrepancy Selected", "ADE 2.0s", l2error_avg_20s_base - l2error_avg_20s_selected),
            ("Discrepancy Selected %", "ADE 2.0s", (l2error_avg_20s_base - l2error_avg_20s_selected) / l2error_avg_20s_selected * 100),

            ("Discrepancy Selected", "ADE 3.0s",
             ((l2error_avg_32s_base + l2error_avg_28s_base) / 2) - (l2error_avg_32s_selected + l2error_avg_28s_selected) / 2),
            ("Discrepancy Selected %", "ADE 3.0s",
             (((l2error_avg_32s_base + l2error_avg_28s_base) / 2) - (l2error_avg_32s_selected + l2error_avg_28s_selected) / 2) / (
                     (l2error_avg_32s_selected + l2error_avg_28s_selected) / 2) * 100),

            ("Discrepancy Selected", "ADE 4.0s", l2error_overall_base - l2error_overall_selected),
            ("Discrepancy Selected %", "ADE 4.0s", (l2error_overall_base - l2error_overall_selected) / l2error_overall_selected * 100),

            ("Discrepancy Selected", "FDE 1.0s",
             ((l2error_dest_08s_base + l2error_dest_12s_base) / 2) - (l2error_dest_08s_selected + l2error_dest_12s_selected) / 2),
            ("Discrepancy Selected %", "FDE 1.0s",
             (((l2error_dest_08s_base + l2error_dest_12s_base) / 2) - (l2error_dest_08s_selected + l2error_dest_12s_selected) / 2) / (
                     (l2error_dest_08s_selected + l2error_dest_12s_selected) / 2) * 100),

            ("Discrepancy Selected", "FDE 2.0s", l2error_dest_20s_base - l2error_dest_20s_selected),
            ("Discrepancy Selected %", "FDE 2.0s", (l2error_dest_20s_base - l2error_dest_20s_selected) / l2error_dest_20s_selected * 100),

            ("Discrepancy Selected", "FDE 3.0s",
             ((l2error_dest_28s_base + l2error_dest_32s_base) / 2) - (l2error_dest_28s_selected + l2error_dest_32s_selected) / 2),
            ("Discrepancy Selected %", "FDE 3.0s",
             (((l2error_dest_28s_base + l2error_dest_32s_base) / 2) - (l2error_dest_28s_selected + l2error_dest_32s_selected) / 2) / (
                     (l2error_dest_28s_selected + l2error_dest_32s_selected) / 2) * 100),

            ("Discrepancy Selected", "FDE 4.0s", l2error_dest_base - l2error_dest_selected),
            ("Discrepancy Selected %", "FDE 4.0s", (l2error_dest_base - l2error_dest_selected) / l2error_dest_selected * 100),
        ]

    else:
        fde /= all_num
        ade /= all_num
        fde_base /= all_num
        ade_base /= all_num
        fde_selected /= all_num
        ade_selected /= all_num
        fde_group /= all_num
        ade_group /= all_num

        results = [
            ("Regular", "TCC",  tcc_new),
            ("Base","TCC", tcc_base),
            ("GroupNet","TCC", tcc_group),
            ("Selected","TCC", tcc_selected),

            ("Regular", "ADE 4.8s", ade),
            ("Regular", "FDE 4.8s", fde),
            ("Base", "ADE 4.8s", ade_base),
            ("Base", "FDE 4.8s", fde_base),

            ("Selected", "ADE 4.8s", ade_selected),
            ("Selected", "FDE 4.8s", fde_selected),
            ("Groupnet", "ADE 4.8s", ade_group),
            ("Groupnet", "FDE 4.8s", fde_group),

            # Discrepancies
            ("Discrepancy", "ADE 4.8s", ade_base - ade),
            ("Discrepancy %", "ADE 4.8s",((ade_base - ade )/ (ade) * 100)),
            ("Discrepancy Selected", "ADE 4.8s", ade_base - ade_selected),
            ("Discrepancy Selected%", "ADE 4.8s", ((ade_base - ade_selected) / (ade_selected) * 100))
        ]

    df = pd.DataFrame(results, columns=["Type", "Metric", "Value"])
    df.to_csv(f"{folder}/evaluation_FDE_results.csv", index=False)
    df.to_string(buf=open(f"{folder}/evaluation_FDE_results.txt", "w"), index=False)
    return


def mission_test(test_loader, older_test, args,model, S, constant):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    stats_path = create_stats_folder(args, constant,"Sampler")

    testing_ADE_FDE(constant, older_test, args, S, stats_path, model)

    S.eval()
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

    for sim_id in range(len(test_loader)):#


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
        groupnet_normalized_std_x_list = []
        selected_normalized_std_x_list = []
        while (future_traj.shape[1] - args.past_length < max_steps):
            with torch.no_grad():
                prediction, _ = model.inference_simulator(traj_sample)  # prediction: [1, N, T, 2]
                _, N, _, _ = prediction.shape
                normalized_vis, velocity, direction, edge_features, edge_weights = compute_features(args, traj_sample, "past")
                prediction_for_infer = prediction.view(20, 1, N, args.future_length, 2).permute(1, 2, 3, 4,0)  # (B, N, T, 2, 20)
                prediction_new, indixes = S.inference(traj_sample, normalized_vis, velocity, direction,
                                                      edge_features,
                                                      edge_weights, prediction_for_infer)  # B, N, 10, 2

                indices_expanded = indixes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.future_length, 2, 1)
                selected = torch.gather(prediction_for_infer, dim=4, index=indices_expanded).squeeze(-1)

            new_traj = selected if args.classifier_method == 'sampler_selected' else prediction_new  # B, N, T 2

            ##### STD ANALYSIS
            prediction_np = prediction_for_infer[0].cpu().numpy()
            groupnet_std_x = np.std(prediction_np.flatten())

            selected_np = new_traj[0].cpu().numpy()  # (N, T, 2)
            selected_std_x = np.std(selected_np.flatten())

            max_range = max(constant.X_MAX, constant.Y_MAX)
            min_range = min(constant.X_MIN, constant.Y_MIN)
            traj_range_x = max_range - min_range

            normalized_groupnet_std_x = groupnet_std_x / (traj_range_x)

            normalized_selected_std_x = selected_std_x / (traj_range_x)

            if np.isfinite(normalized_groupnet_std_x) and normalized_groupnet_std_x < 100:
                groupnet_normalized_std_x_list.append(normalized_groupnet_std_x)
            else:
                print(f"skipping suspicious GroupNet normalized std value: {normalized_groupnet_std_x}")

            if np.isfinite(normalized_selected_std_x) and normalized_selected_std_x < 100:
                selected_normalized_std_x_list.append(normalized_selected_std_x)
            else:
                print(f"skipping suspicious Selected normalized std value: {normalized_selected_std_x}")


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

        sim_missions = len(controlled) * args.mission_num  # C*missions for each agent -> total misssoin
        total_controlled += len(controlled)
        sim_achieved = len(mission_log)  # total missions achieved
        total_missions += sim_missions  # missions in all simulations
        missions_achieved += sim_achieved  # missions achieved in all simulations
        if sim_achieved == sim_missions:
            full_success_count += 1  # in this sim all misssions achieved

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
            "avg_missions_per_agent": sim_achieved / len(controlled),
            "mission_log": mission_log,
            "sim_success_rate": sim_success_rate
        })

        if sim_id == 3:
            # vis_predictions_missions(constant, future_traj, mission_log, target_status, args, missions.cpu(), agents_idx_plot,
            #                          stats_path)
            vis_predictions_no_missions(constant, future_traj,  args,stats_path)

    groupnet_normalized_std_x_array = np.array(groupnet_normalized_std_x_list)
    groupnet_normalized_std_x_cleaned = groupnet_normalized_std_x_array[np.isfinite(groupnet_normalized_std_x_array)]
    overall_groupnet_normalized_std_x = np.nanmean(groupnet_normalized_std_x_cleaned)

    selected_normalized_std_x_array = np.array(selected_normalized_std_x_list)
    selected_normalized_std_x_cleaned = selected_normalized_std_x_array[np.isfinite(selected_normalized_std_x_array)]
    overall_selected_normalized_std_x = np.nanmean(selected_normalized_std_x_cleaned)

    print("\n--- Normalized Std Statistics ---")
    print(f"GroupNet Normalized Std (X): {overall_groupnet_normalized_std_x:.4f}")
    print(f"Selected Normalized Std (X): {overall_selected_normalized_std_x:.4f}")


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
    )

    df_per_sim = pd.DataFrame(sim_logs)

    avg_per_agent_all = df_per_sim["avg_missions_per_agent"].mean()
    std_per_agent_all = df_per_sim["avg_missions_per_agent"].std()
    avg_sim_success_rate = df_per_sim["sim_success_rate"].mean()
    std_sim_success_rate = df_per_sim["sim_success_rate"].std()

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
        "Avg Missions per Simulation": [total_missions / len(test_loader)],
        "Total Controlled Agents": [total_controlled],
        "Total Missions Achieved": [missions_achieved],
        "Overall Success Rate": [missions_achieved / total_missions],
        "Simulations with Full Success": [full_success_count],
        "Avg Missions Achieved per Simulation (General)": [missions_achieved / len(test_loader)],
        "avg simimulation success rate": [avg_sim_success_rate],
        "Std simimulation success rate": [std_sim_success_rate],

        "Mean Mission Time (All Agents)": [mean_mission_time],
        "Std Mission Time (All Agents)": [std_mission_time],
        "Std of Mean Mission Time (Between Sims)": [df_per_sim["mean_mission_time"].std()],

        "Min Mission Time (All Agents)": [min_time],
        "Min of Mean Mission Times (Between Sims)": [df_per_sim["mean_mission_time"].min()],

        "Max Mission Time (All Agents)": [max_time],
        "Max of Mean Mission Times (Between Sims)": [df_per_sim["mean_mission_time"].max()],

        "Avg Missions per Agent (All Sims)": [avg_per_agent_all],
        "Std of Avg Missions per Agent (Between Sims)": [std_per_agent_all],
        "Total Sims": [len(test_loader)]
    })

    print(f"\n--- Final Statistics over {len(test_loader)} simulations ---")
    print(df_summary.T)

    if args.dataset == "nba":
        df_centroids.to_csv(
        f"{stats_path}/centroids_agents.csv")
    df_per_sim.to_csv(
        f"{stats_path}/per_sim_stats.csv")
    df_summary.to_csv(
        f"{stats_path}/overall_summary.csv")


def load_dataset(type, test_loader, args, model):
    DATASET_PATH = f"datasets/{args.dataset}/data/trajectory_dataset_sampler_{type}_{args.training_type}_{args.scene}_{args.sdd_scene}.pt"

    if os.path.exists(DATASET_PATH):
        print("Loading existing dataset...")
        traj_dataset = torch.load(DATASET_PATH, weights_only=False)
    else:
        print("Creating new dataset...")
        if args.dataset in ['nba', 'fish', 'syn']:
            traj_dataset = create_traj(test_loader, model, args)
        else:
            traj_dataset = create_traj_flex_N(test_loader, model, args)
        torch.save(traj_dataset, DATASET_PATH)

    print("traj_dataset", len(traj_dataset))
    if type == "train":
        new_batch_size = args.batch_size
    else:
        new_batch_size = args.batch_size*32

    if args.dataset in ['nba', 'fish', 'syn']:
        loader = DataLoader(
            traj_dataset,
            batch_size=new_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=seq_collate_sampler)
    else:
        loader = DataLoader(traj_dataset, batch_sampler=GroupedBatchSampler(traj_dataset.grouped_seq_indices,
                                                                               batch_size=new_batch_size,
                                                                               shuffle=True,
                                                                               drop_last=False),collate_fn=seq_collate_sampler)
    return loader



if __name__ == '__main__':
    args = parse_args()
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=f"runs/Sampler_training/{args.timestamp}_{args.dataset}_{args.scene}_{args.sdd_scene}_{args.training_type}_{args.seed}_{args.test_mlp}")
    """ setup """
    names = [x for x in args.model_names.split(',')]

    for name in names:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        if args.dataset == 'nba':
            args.model_save_dir = 'G1/saved_models/nba'
            args.agent_num = 11
            args.mission_num =12
            args.length = 120
        elif args.dataset == 'fish':
            args.model_save_dir = 'G1/saved_models/fish'
            args.agent_num = 8
            args.mission_num =12
            args.length = 120
        elif args.dataset == 'syn':
            args.model_save_dir = 'G1/saved_models/syn'
            args.agent_num = 6
        elif args.dataset == 'sdd':
            args.model_save_dir = 'G1/saved_models/sdd'
            args.past_length = 8
            args.future_length = 12
            args.agent_num = 8
            args.mission_num =8
            args.length = 60
        elif args.dataset == 'eth':
            args.model_save_dir = 'G1/saved_models/eth'
            args.past_length = 8
            args.future_length = 12
            args.agent_num = 8

        """ model """
        saved_path = os.path.join(args.model_save_dir, str(name) + '.p')
        print('load model from:', saved_path)
        checkpoint = torch.load(saved_path)#weights_only=False
        training_args = checkpoint['model_cfg']

        model = GroupNet(training_args, args.device)
        model.set_device(args.device)
        model.eval()
        model.load_state_dict(checkpoint['model_dict'], strict=True)
        model.args.sample_k = 20

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

        if args.dataset == 'nba':
            d_set = NBADataset(
                obs_len=args.past_length,
                pred_len=args.future_length,
                dataset_part=args.training_type)
        elif args.dataset == 'fish':
            d_set = FISHDataset(
                obs_len=args.past_length,
                pred_len=args.future_length,
                dataset_part=args.training_type)

        elif args.dataset == 'syn':
            d_set = SYNDataset(
                obs_len=args.past_length,
                pred_len=args.future_length,
                dataset_part=args.training_type)
        elif args.dataset == 'sdd':
            d_set = TrajectoryDatasetSDD(data_dir="datasets/raw/SDD", obs_len=args.past_length,
                                             pred_len=args.future_length, skip=1,
                                             min_ped=1, delim='space', save_path="datasets/sdd/SDD.pt",
                                             mode=args.training_type)
        elif args.dataset == 'eth':
            d_set = TrajectoryDatasetETH(data_dir="datasets/raw/eth", obs_len=args.past_length,
                                             pred_len=args.future_length, skip=1, min_ped=1,
                                             delim='space', test_scene=args.scene, save_path=args.scene,
                                             mode=args.training_type)



        if args.mode == 'train':

            if args.dataset == 'nba' or args.dataset == 'fish' or args.dataset == 'syn':
                loader = DataLoader(
                    d_set,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=seq_collate)
            else:
                loader = DataLoader(d_set, batch_sampler=GroupedBatchSampler(d_set.grouped_seq_indices_train,
                                                                             batch_size=args.batch_size,
                                                                             shuffle=True,
                                                                             drop_last=False), collate_fn=seq_collate)

            extra_train_loader = load_dataset("train", loader, args, model)
            train(extra_train_loader, args, S, writer)

        else:
            SEED = args.seed
            g = torch.Generator()
            g.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            if args.dataset == 'nba' or args.dataset == 'fish' or args.dataset == 'syn':
                loader = DataLoader(
                    d_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=seq_collate,generator=g,)
            else:
                print("dste", d_set.mode)
                loader = DataLoader(d_set, batch_sampler=GroupedBatchSampler(d_set.grouped_seq_indices_test,
                                                                             batch_size=args.batch_size,
                                                                             shuffle=False,
                                                                             drop_last=False), collate_fn=seq_collate,generator=g,)


            S_path = f"Sampler/saved_model/{args.dataset}/{args.saved_models_SAM}.pth"
            S.load_state_dict(torch.load(S_path, map_location=args.device))#, weights_only=False

            S.to(args.device)
            final_test_loader = load_dataset("test", loader, args, model)

            if args.dataset in ['nba', 'syn', 'fish']:
                traj_dataset = torch.utils.data.Subset(d_set, range(args.sim_num))
                if args.dataset == 'nba':
                    constant = ConstantNBA
                elif args.dataset == 'fish':
                    constant = ConstantFish
                elif args.training_type == 'test_rigid' and args.dataset == 'syn':
                    constant = ConstantSYNR
                elif args.training_type == 'test_smooth' and args.dataset == 'syn':
                    constant = ConstantSYNS

            else:
                if args.dataset == 'eth':
                    constant = return_the_eth_scene(args.scene)
                else:
                    constant = return_the_sdd_scene(args.sdd_scene)
                traj_dataset = Subset(d_set, constant.n_agents, args.sim_num, args.sdd_scene)
            print(f"Missions number {args.mission_num}")
            data = prepare_target(args, args.device, traj_dataset,constant, "test")


            wrapped_test = MissionTestDataset(traj_dataset, data)
            testing_loader = DataLoader(
                wrapped_test,
                batch_size=args.batch_size,  # simulate one scenario at a time
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            mission_test(testing_loader,final_test_loader, args, model, S, constant)

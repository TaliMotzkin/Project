import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import scipy.stats as stats


import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from config import parse_args
from model.models_GAN import  TrajectoryClassifier
from model.models_sampler import SamplerMission, Sampler, SamplerMLP
from data.dataloader_fish import FISHDataset
from torch.utils.data import DataLoader
from model.GroupNet_nba import GroupNet
from loss.loss_sampler import LossCompute
from data.dataloader_nba import NBADataset
import time
from torch.utils.tensorboard import SummaryWriter
import math
from utilis import *
from data.dataloader_GAN import seq_collate_sampler_GAN, TrajectoryDatasetFlexN, TrajectoryDatasetNonFlexN
from data.dataloader_SDD import TrajectoryDatasetSDD
from data.dataloader_ETH import TrajectoryDatasetETH, GroupedBatchSampler, seq_collate
from data.dataloader_syn import SYNDataset



def create_traj(test_loader, model, args, S):

    iter = 0
    actor_num = args.agent_num
    past_traj = torch.empty((0, actor_num, args.past_length, 2)).to(args.device)  # total samples, 8 agents, 5 time steps, 2 coords
    group_net = torch.empty((0, actor_num, args.future_length, 2, 20)).to(args.device)  # (Total_samples, N, 10, 2, 20)
    future_traj = torch.empty((0, actor_num, args.future_length, 2)).to(args.device)
    predictions_real = torch.empty((0, args.agent_num, args.future_length, 2)).to(args.device)
    indexes_list_selected = torch.empty((0, args.agent_num)).to(args.device)


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

            past_traj_ten = data['past_traj'].to(args.device)
            visability_mat_past = visability_mat_past.to(args.device)
            velocity_past = velocity_past.to(args.device)
            direction_past = direction_past.to(args.device)
            edge_features_past = edge_features_past.to(args.device)
            edge_weights_past = edge_weights_past.to(args.device)

            y = data['future_traj'].reshape(data['future_traj'].shape[0] * actor_num, args.future_length,2)  # 200, 10, 2
            y = y.unsqueeze(0).repeat(20, 1, 1, 1).to(args.device)
            error = torch.norm(y - prediction, dim=3).mean(dim=2)  # 20, BN,
            indices = torch.argmin(error, dim=0)  # BN
            prediction_for_infer = prediction.view(20, data['future_traj'].shape[0], args.agent_num, args.future_length, 2).permute(1, 2, 3, 4,0)  # (B, N, T, 2, 20)


            prediction_new, indixes = S.inference( past_traj_ten, visability_mat_past, velocity_past, direction_past,
                                                  edge_features_past,
                                                  edge_weights_past, prediction_for_infer)
            indices_expanded = indixes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.future_length, 2,1)
            selected = torch.gather(prediction_for_infer, dim=4, index=indices_expanded).squeeze(-1)


            y_s =selected.reshape(data['future_traj'].shape[0] * args.agent_num, args.future_length,2)  # 200, 10, 2
            y_s = y_s.unsqueeze(0).repeat(20, 1, 1, 1).to(args.device)
            error_s = torch.norm(y_s - prediction, dim=3).mean(dim=2)  # 20, BN,
            indices_selected = torch.argmin(error_s, dim=0).to(args.device)  # BN

        indexes_list = torch.cat((indexes_list, indices.view(data['future_traj'].shape[0], actor_num)), dim=0)
        indexes_list_selected = torch.cat((indexes_list_selected, indices_selected.view(data['future_traj'].shape[0], args.agent_num)), dim=0)

        edge_weights_past_list = torch.cat((edge_weights_past_list, edge_weights_past.to(args.device)), dim=0)
        edge_features_past_list = torch.cat((edge_features_past_list, edge_features_past.to(args.device)), dim=0)
        direction_past_list = torch.cat((direction_past_list, direction_past.to(args.device)), dim=0)
        velocity_past_list = torch.cat((velocity_past_list, velocity_past.to(args.device)), dim=0)
        visability_mat_past_list = torch.cat((visability_mat_past_list, visability_mat_past.to(args.device)), dim=0)
        predictions_real = torch.cat((predictions_real, selected), dim=0)

        future_traj = torch.cat((future_traj, data['future_traj'].to(args.device)), dim=0)
        prediction = prediction.permute(1, 2, 3, 0).view(data['future_traj'].shape[0], actor_num, args.future_length, 2, 20)
        group_net = torch.cat((group_net, prediction.to(args.device)), dim=0)
        past_traj = torch.cat((past_traj, data['past_traj'].to(args.device)), dim=0)

        iter += 1
        if iter % 100 == 0:
            print("iteration:", iter, "in data creation")
    # print("future_traj", future_traj.shape, "past_traj", past_traj.shape, "group_net", group_net.shape, "H_list", H_list.shape)
    traj_dataset = TrajectoryDatasetNonFlexN(past_traj, future_traj, group_net, edge_weights_past_list, edge_features_past_list,
                                         direction_past_list, velocity_past_list, visability_mat_past_list,indexes_list,predictions_real, indexes_list_selected)

    return traj_dataset

def create_traj_flex_N(test_loader, model, args, S):
    past_list, fut_list, old_fut_list, indexes_list_selected = [], [], [], []
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
            past_traj_ten =  data['past_traj'].to(args.device)
            vis = vis.to(args.device)
            vel =vel.to(args.device)
            dirc = dirc.to(args.device)
            e_f = e_f.to(args.device)
            e_w = e_w.to(args.device)

            # best‐indexes per sample
            y = data['future_traj'].reshape(B*N, args.future_length, 2).unsqueeze(0).repeat(20,1,1,1).to(args.device)
            errs = torch.norm(y - pred20, dim=3).mean(dim=2)        # (20, B*N)
            best = errs.argmin(dim=0).view(B, N)                        # (B, N)

            pred_per_scene  = (pred20.permute(1, 2, 3, 0).reshape(B, N, args.future_length, 2, 20))

            prediction_new, indixes = S.inference(past_traj_ten, vis, vel, dirc,e_f,e_w, pred_per_scene)
            indices_expanded = indixes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.future_length, 2,1)
            selected = torch.gather(pred_per_scene, dim=4, index=indices_expanded).squeeze(-1)

            y_s = selected.reshape(B * N, args.future_length, 2)  # 200, 10, 2
            y_s = y_s.unsqueeze(0).repeat(20, 1, 1, 1).to(args.device)
            error_s = torch.norm(y_s - pred20, dim=3).mean(dim=2)  # 20, BN,
            indices_selected = torch.argmin(error_s, dim=0).view(B, N).to(args.device)  # BN

            past_list.extend(data['past_traj'].cpu() )
            old_fut_list.extend(data['future_traj'].cpu() )
            fut_list.extend(selected.cpu() )
            group_list.extend(pred_per_scene.cpu() )
            ew_list.extend(e_w.cpu() )
            ef_list.extend(e_f.cpu() )
            dir_list.extend(dirc.cpu() )
            vel_list.extend(vel.cpu() )
            vis_list.extend(vis.cpu() )
            idx_list.extend(best.cpu() )
            indexes_list_selected.extend(indices_selected.cpu() )

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
    all_indeces_selcted = torch.cat(indexes_list_selected, dim =0).to(device)
    all_old_fut_list = torch.cat(old_fut_list, dim=0).to(device)

    traj_dataset = TrajectoryDatasetFlexN(all_past, all_old_fut_list,all_groupnet, all_edge_w, all_edge_f,
        all_dir, all_vel, all_vis, all_indices , grouped_seq_indices,seq_start_end, all_future, all_indeces_selcted)

    return traj_dataset

def plot_losses(args, train_losses_g, train_losses_d, train_scores_real, train_scores_fake,train_scores_unco):
    epochs = range(1, len(train_losses_g) + 1)
    num_agents =args.agent_num

    train_scores_real = np.array([t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in train_scores_real])
    train_scores_fake = np.array([t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in train_scores_fake])
    train_scores_unco = np.array([t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in train_scores_unco])

    # val_scores_real = np.array([t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in val_scores_real])
    # val_scores_fake = np.array([t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in val_scores_fake])

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses_g, label='Train Generator Loss', color='blue')
    # plt.plot(epochs, val_losses_g, label='Val Generator Loss', color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('Generator Loss')
    plt.legend()
    plt.title('Generator Loss Progression')
    plt.savefig(f"GANS/plots/GAN_Generator_Loss_Sampler_{args.dataset}_{args.classifier_method}_{args.scene}_{args.sdd_scene}_{args.training_type}_how_far_{args.how_far}_{args.test_mlp}_{args.info}.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses_d, label='Train Discriminator Loss', color='red')
    # plt.plot(epochs, val_losses_d, label='Val Discriminator Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Discriminator Loss')
    plt.legend()
    plt.title('Discriminator Loss Progression')
    plt.savefig(f"GANS/plots/GAN_Discriminator_Loss_Sampler_{args.dataset}_{args.classifier_method}_{args.scene}_{args.sdd_scene}_{args.training_type}_how_far_{args.how_far}_{args.test_mlp}_{args.info}.png")
    plt.close()

    color_map = get_agent_colors(args, args.agent_num)
    controlled_agents = args.agent if isinstance(args.agent, (list, tuple)) else [args.agent]
    cols = 3
    rows = math.ceil(num_agents / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True)

    axs = axs.flatten()

    for agent_id in range(num_agents):
        ax = axs[agent_id]
        color = color_map[agent_id]
        # lw = 2.5 if agent_id in controlled_agents else 1.5
        # marker = 'o' if agent_id in controlled_agents else None
        # facecolor = 'white' if agent_id in controlled_agents else None
        # edgecolor = 'black' if agent_id in controlled_agents else None
        lw = 1.5
        marker = None
        facecolor = None
        edgecolor = None

        ax.plot(epochs, train_scores_real[:, agent_id], label='Real',
                linestyle='-', color=color, linewidth=lw, marker=marker,
                markerfacecolor=facecolor, markeredgecolor=edgecolor)
        ax.plot(epochs, train_scores_fake[:, agent_id], label='Fake',
                linestyle='--', color=color, linewidth=lw, marker=marker,
                markerfacecolor=facecolor, markeredgecolor=edgecolor)
        ax.plot(epochs, train_scores_unco[:, agent_id], label='Uncontrolled',
                linestyle=':', color=color, linewidth=lw, marker=marker,
                markerfacecolor=facecolor, markeredgecolor=edgecolor)
        ax.set_title(f'Agent {agent_id}')
        ax.set_ylabel("Score")
        ax.grid(True)
        ax.legend(fontsize='x-small')

    for ax in axs[num_agents:]:
        ax.axis("off")

    plt.xlabel("Epochs")
    plt.suptitle("Train Discriminator Scores Per Agent", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        f"GANS/plots/Train_Discriminator_Scores_Per_Agent_Faceted_GroupNet_{args.dataset}_{args.classifier_method}_{args.scene}_{args.sdd_scene}_{args.training_type}_how_far_{args.how_far}_{args.test_mlp}_{args.info}.png")
    plt.close()

    # fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True)
    #
    # axs = axs.flatten()
    #
    # for agent_id in range(num_agents):
    #     ax = axs[agent_id]
    #     color = color_map[agent_id]
    #     lw = 2.5 if agent_id in controlled_agents else 1.5
    #     marker = 'o' if agent_id in controlled_agents else None
    #     facecolor = 'white' if agent_id in controlled_agents else None
    #     edgecolor = 'black' if agent_id in controlled_agents else None
    #
    #     ax.plot(epochs, val_scores_real[:, agent_id], label='Real',
    #             linestyle='-', color=color, linewidth=lw, marker=marker,
    #             markerfacecolor=facecolor, markeredgecolor=edgecolor)
    #     ax.plot(epochs, val_scores_fake[:, agent_id], label='Fake',
    #             linestyle='--', color=color, linewidth=lw, marker=marker,
    #             markerfacecolor=facecolor, markeredgecolor=edgecolor)
    #     ax.set_title(f'Agent {agent_id}')
    #     ax.set_ylabel("Score")
    #     ax.grid(True)
    #     ax.legend(fontsize='x-small')
    #
    # for ax in axs[num_agents:]:
    #     ax.axis("off")
    #
    # plt.xlabel("Epochs")
    # plt.suptitle("Val Discriminator Scores Per Agent", fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig(
    #     f"GANGS/plots/Val_Discriminator_Scores_Per_Agent_Faceted_{args.timestamp}_{args.dataset}_{args.classifier_method}.png")
    # plt.close()

def train(constant, writer, train_loader, args, SM, D):
    print(args.timestamp)
    lossfn = LossCompute(args, SM=SM, D=D)
    optimizer_SM = torch.optim.Adam(SM.parameters(), lr=2e-4)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)
    scheduler_SM = torch.optim.lr_scheduler.StepLR(optimizer_SM, step_size=args.lr_step, gamma=args.lr_gamma)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=args.lr_step, gamma=args.lr_gamma)

    train_losses_g = []
    train_losses_d = []

    train_scores_real = []
    train_scores_fake = []
    train_scores_unco = []


    n_critic = args.n_critic
    for i in range(args.epoch):
        iter_num = 0
        time_epoch = time.time()
        SM.train()
        D.train()
        train_loss_g = 0
        train_loss_d = 0

        fake_agent_counts = np.zeros(args.agent_num)
        fake_agent_counts_val = np.zeros(args.agent_num)
        unco_agent_counts = np.zeros(args.agent_num)
        unco_agent_counts_val = np.zeros(args.agent_num)
        real_agent_counts = np.zeros(args.agent_num)

        train_real_score = np.zeros(args.agent_num)
        train_fake_score = np.zeros(args.agent_num)
        train_unco_score = np.zeros(args.agent_num)

        mission_loss_mean = 0
        discriminator_loss_mean=0
        num_batches = 0
        loss_pred_mean = 0
        for batch in train_loader:
            current_N = batch['past_traj'].shape[1]
            iter_num += 1
            past = batch['past_traj'].to(args.device)
            prediction = batch['group_net'].to(args.device)

            if args.classifier_method == 'sampler_selected':
                future_traj = batch["future_traj"].to(args.device)
                indices = batch['indexes_list_selected'].to(args.device)

            elif args.classifier_method == 'future':
                future_traj = batch['old_future'].to(args.device)
                indices = batch["indexes_list"].to(args.device)

            edge_weights_past = batch['edge_weights_past'].to(args.device)
            edge_features_past = batch['edge_features_past'].to(args.device)
            direction_past = batch['direction_past'].to(args.device)
            velocity_past = batch['velocity_past'].to(args.device)
            visability_mat_past = batch['visability_mat_past'].to(args.device)

            agents_tragets, agents_idx, error_tolerance, _ = prepare_targets_mission_net(constant, batch, args.dataset, args.device, i)
            for _ in range(n_critic):
                optimizer_D.zero_grad()

                total_loss_d, loss_real, loss_fake, scores_fake, scores_real ,scores_uncontrolled= lossfn.compute_discriminator_loss(agents_tragets.to(args.device),agents_idx.to(args.device),error_tolerance,
                                                                                                 future_traj, past,edge_weights_past,edge_features_past,
                                                                                                 direction_past,velocity_past,visability_mat_past,prediction)


                total_loss_d.backward()
                optimizer_D.step()

            optimizer_SM.zero_grad()
            tuple_loss = lossfn.compute_generator_loss(i, agents_tragets,agents_idx,error_tolerance,future_traj, past,edge_weights_past,edge_features_past,
                                                   direction_past,velocity_past,visability_mat_past,prediction,indices)

            total_loss_g, pred_eqin_loss ,loss_score , discriminator_loss, mission_loss = tuple_loss
            mission_loss_mean += mission_loss
            discriminator_loss_mean += discriminator_loss
            loss_pred_mean += pred_eqin_loss

            total_loss_g.backward()
            optimizer_SM.step()
            train_loss_g += total_loss_g.item()
            train_loss_d += total_loss_d.item()

            scores_real_mean = scores_real.mean(dim=0).detach().cpu().numpy()  # N
            scores_fake_mean = scores_fake.mean(dim=0).detach().cpu().numpy()  # N,

            if agents_idx.numel() != 0:
                scores_uncontrolled_mean  = scores_uncontrolled.mean(dim=0).detach().cpu().numpy() #N
                uncontrolled_idx = [i for i in range(current_N) if i not in agents_idx.tolist()]
                for k, a in enumerate(uncontrolled_idx):  # the order is not important when dataset is not nba\target
                    train_unco_score[a] += scores_uncontrolled_mean[k]
                    unco_agent_counts[a] += 1


            for a in range(current_N):
                train_real_score[a] += scores_real_mean[a]
                real_agent_counts[a] += 1

            for k, a in enumerate(agents_idx):
                train_fake_score[a] += scores_fake_mean[k]
                fake_agent_counts[a] += 1


            num_batches += 1
            if iter_num % args.iternum_print == 0:
                print("%%%%%%%%%%%%%% iter_num %%%%%%%%%%%%%%: ", iter_num, "out of ", len(train_loader), "\n")
                print("############## discriminator loss")
                print("loss_d, ", total_loss_d.item(), "loss_real, " , loss_real, " loss_fake, ", loss_fake)
                print("scores_fake, ",  scores_fake_mean)
                print("scores_real", scores_real_mean)
                print("scores_uncontrolled", scores_uncontrolled_mean)

                print("############## generator loss ")
                print("total_loss ", total_loss_g.item(),  " pred_eqin_loss ,", pred_eqin_loss,  "loss_score ", loss_score
                      , "discriminator_loss " ,discriminator_loss, "mission_loss ", mission_loss)


        scheduler_SM.step()
        scheduler_D.step()
        # print(output) #64, 10, 2

        train_losses_g.append(train_loss_g / num_batches)
        train_losses_d.append(train_loss_d / num_batches)
        train_scores_real.append(train_real_score / np.clip(real_agent_counts, 1, None))
        train_scores_fake.append(train_fake_score / np.clip(fake_agent_counts, 1, None))
        train_scores_unco.append(train_unco_score / np.clip(unco_agent_counts, 1, None))
        mission_loss_mean =mission_loss_mean/num_batches
        discriminator_loss_mean =discriminator_loss_mean/num_batches
        loss_pred_mean = loss_pred_mean/num_batches

        writer.add_scalar("Loss/Train_Generator", train_losses_g[-1], i)
        writer.add_scalar("Loss/Train_Discriminator", train_losses_d[-1], i)

        for agent_id in range(args.agent_num):
            writer.add_scalar(f"Score/Train_Real_Agent_{agent_id}", train_scores_real[-1][agent_id], i)
            writer.add_scalar(f"Score/Train_Fake_Agent_{agent_id}", train_scores_fake[-1][agent_id], i)
            writer.add_scalar(f"Score/Train_Uncontrolled_Agent_{agent_id}", train_scores_unco[-1][agent_id], i)


        time_end = time.time()
        if (i + 1) % args.save_every == 0:
            saveModel_Sampler_GAN(SM, D, args, str(i + 1))

        print(
            f"Epoch [{i + 1}/{args.epoch}] - "
            f"Train Loss G: {train_losses_g[-1]:.4f}, D: {train_losses_d[-1]:.4f}, "
            f"Train Real Score (targeted agents): {np.round(train_scores_real[-1], 4)}, "
            f"Fake Score (targeted agents): {np.round(train_scores_fake[-1], 4)},"
            f"Uncontrolled Score (targeted agents): {np.round(train_scores_unco[-1], 4)},"
            f"average mission_loss : {mission_loss_mean} , "
            f"discriminator_loss_mean: {discriminator_loss_mean} , "
            f"loss_pred_mean: {loss_pred_mean} , "
            f"Time: {time_end - time_epoch:.2f}s"
        )

    plot_losses(args, train_losses_g, train_losses_d, train_scores_real, train_scores_fake,train_scores_unco)


def plot_score_list(score_list, args):
    score_array = np.array(score_list)  # shape: (steps, N)
    num_agents = score_array.shape[1]
    timesteps = 5 * 0.4 + np.arange(len(score_array)) * 10 * 0.4

    color_map = get_agent_colors(args)
    controlled_agents = args.agent if isinstance(args.agent, (list, tuple)) else [args.agent]

    cols = 3
    rows = math.ceil(num_agents / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True)
    axs = axs.flatten()

    for agent_id in range(num_agents):
        ax = axs[agent_id]
        color = color_map[agent_id]
        scores = score_array[:, agent_id]

        if agent_id in controlled_agents:
            ax.plot(timesteps, scores, label=f'Agent {agent_id}',
                    linewidth=2.5, color=color,
                    marker='o', markerfacecolor='white', markeredgecolor='black')
        else:
            ax.plot(timesteps, scores, label=f'Agent {agent_id}',
                    linewidth=1.5, color=color)

        ax.set_title(f"Agent {agent_id}")
        ax.set_ylabel("Score")
        ax.grid(True)
        ax.legend(loc='best', fontsize='small')

    for ax in axs[num_agents:]:
        ax.axis("off")

    axs[0].set_xlabel("Time (s)")
    plt.suptitle("Discriminator Scores per Agent Over Time", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"GANS/plots/faceted_score_list_{args.timestamp}_{args.dataset}_{args.classifier_method}.png")
    plt.close()




def non_mission_test(testing_loader, args, model, S, SM, D ,constant):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    stats_path = create_stats_folder(args, constant, 'sampler-gan')

    SM.eval()
    S.eval()
    D.eval()
    model.eval()

    total_missions = 0
    missions_achieved = 0
    total_controlled = 0

    score_list_all = []

    sim_logs = []
    all_centroids = []
    mission_tolerance = constant.buffer

    all_traj = []
    all_valid_fake_centroids = []
    all_valid_fake_traj= []
    X_MIN, X_MAX = constant.X_MIN, constant.X_MAX
    Y_MIN, Y_MAX = constant.Y_MIN, constant.Y_MAX


    for sim_id in range(len(testing_loader)):

        traj_sample, missions, controlled , _= testing_loader.dataset[sim_id]  # traj_sample: [N, T, 2], missions: [N, M, 2], controlled: [max_controlled]
        N, _, _ = traj_sample.shape
        score_list_sim = []
        traj_sample = traj_sample.unsqueeze(0).to(args.device)  # [1, N, T, 2]
        controlled = []  # valid controlled agent indices

        iter_num = 0
        agents_idx = controlled.copy() # []

        iter_num = 0
        max_steps = int(args.length / 0.4)
        valid_fake_chunks = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]

        future_traj = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]

        while (future_traj.shape[1] - args.past_length < max_steps):
            with torch.no_grad():
                one_mission =  torch.zeros(1, 0, 2, device=args.device)

                prediction, H = model.inference_simulator(traj_sample)
                normalized_vis, velocity, direction, edge_features, edge_weights = compute_features(args, traj_sample, "past")

                prediction_for_infer = prediction.view(20, 1, args.agent_num, args.future_length, 2).permute(1, 2, 3, 4, 0)  # (B, N, T, 2, 20)

                eq_in, index = SM.inference(args.alpha, agents_idx, one_mission, constant.buffer, traj_sample, normalized_vis, velocity,
                                              direction, edge_features, edge_weights, prediction_for_infer)
                indices_expanded_SM = index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.future_length, 2, 1)
                selected_SM = torch.gather(prediction_for_infer, dim=4, index=indices_expanded_SM).squeeze(-1)
                fake_traj = selected_SM if args.classifier_method == 'sampler_selected' else eq_in  # B, N, T 2

                scores = D(torch.cat([traj_sample, fake_traj], dim = 2)).squeeze(0) # B=1, N -> N


                if args.dataset in ['sdd', 'eth']:
                    fake_np = fake_traj[0].cpu().numpy()  # shape: [N, T, 2]
                    out_of_bounds_mask = (
                            (fake_np[:, :, 0] < X_MIN) | (fake_np[:, :, 0] > X_MAX) |
                            (fake_np[:, :, 1] < Y_MIN) | (fake_np[:, :, 1] > Y_MAX)
                    )
                    invalid_mask = (out_of_bounds_mask.sum(axis=1) > (fake_np.shape[1] // 3))  # shape: [N]
                else:
                    invalid_mask = torch.zeros(scores.shape, dtype=torch.bool)

                scores_masked = scores.clone()
                scores_masked[invalid_mask] = float('nan')
                score_list_sim.append(scores_masked)
                if invalid_mask.any(): # if there is any invalid agent - dont add it to calc for fixed number of N
                    pass
                else:
                    valid_fake_chunks = np.concatenate((valid_fake_chunks, fake_traj[0].cpu().numpy()), axis=1) #N, T, 2

            future_traj = np.concatenate((future_traj, fake_traj[0].cpu().numpy()), axis=1)
            traj_sample = fake_traj[:, :, -args.past_length:, :]
            iter_num += 1


        if np.isnan(future_traj).any():
            continue
        centroids = future_traj.mean(axis=0)  # shape (T, 2)
        all_centroids.append(centroids)
        all_traj.append(future_traj)

        print("real all_traj len", future_traj.shape[1])
        if args.dataset in ['sdd', 'eth']:
            valid_fake_centroids = valid_fake_chunks.mean(axis=0) # [20, 2], [8,2]...
            all_valid_fake_centroids.append(valid_fake_centroids)  # S, T-changing, 2
            all_valid_fake_traj.append(valid_fake_chunks)
            print("fake all_traj len", valid_fake_chunks.shape[1])

        scores_tensor = torch.stack(score_list_sim, dim=0)  # shape: [T, N]
        mask = ~torch.isnan(scores_tensor)  # shape: [T, N], True where valid
        scores_tensor = torch.nan_to_num(scores_tensor, nan=0.0)
        #Sum and count
        sum_scores = (scores_tensor * mask).sum(dim=0)  # [N]
        count_valid = mask.sum(dim=0).clamp(min=1)  # [N], avoid division by 0
        sim_scores = sum_scores / count_valid
        score_list_all.append(sim_scores.cpu().numpy())  # collect into main list

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
        all_centroids=all_centroids if args.dataset not in ['eth', 'sdd'] else all_valid_fake_centroids,  # list of np.ndarray, shape [T, 2]
        field_length=constant.field if args.dataset != "sdd" else constant.field_T,
        field_width=constant.field_width if args.dataset != "sdd" else constant.field_width_T,
        num_blocks=constant.num_blocks,
        timestep_duration=0.4,
        args=args,
        folder=f'{stats_path}/no_mission/',
        constant=constant
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

    print(f"\n--- Final Statistics over {len(testing_loader)} simulations ---")
    print(df_summary.T)

    if args.dataset == "nba":
        df_centroids.to_csv(
            f"{stats_path}/no_mission/centroids_agents.csv")
    df_per_sim.to_csv(
        f"{stats_path}/no_mission/per_sim_stats.csv")
    df_summary.to_csv(
        f"{stats_path}/no_mission/overall_summary.csv")




def mission_test(testing_loader, args, model, S, SM, D ,constant):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    stats_path = create_stats_folder(args, constant, 'sampler-gan')

    SM.eval()
    S.eval()
    D.eval()
    model.eval()

    total_missions = 0
    missions_achieved = 0
    total_controlled = 0
    full_success_count = 0

    score_list_all = []

    sim_logs = []
    all_centroids = []
    mission_tolerance = constant.buffer

    per_agent_mission_times = []
    all_traj = []
    all_mission_durations = []
    agent_full_success = []
    controlled_all = []
    sims_success_rate = []

    agent_scores_real_sum = torch.zeros(args.agent_num, device=args.device)
    agent_scores_fake_sum = torch.zeros(args.agent_num, device=args.device)
    agent_real_counts = torch.zeros(args.agent_num, device=args.device)
    agent_fake_counts = torch.zeros(args.agent_num, device=args.device)

    all_valid_fake_centroids = []
    all_valid_fake_traj= []
    X_MIN, X_MAX = constant.X_MIN, constant.X_MAX
    Y_MIN, Y_MAX = constant.Y_MIN, constant.Y_MAX


    for sim_id in range(len(testing_loader)):#

        traj_sample, missions, controlled , _= testing_loader.dataset[sim_id]  # traj_sample: [N, T, 2], missions: [N, M, 2], controlled: [max_controlled]
        N, _, _ = traj_sample.shape

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
        score_list_sim = []

        future_traj = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]
        agent_mission_times = {a: [] for a in agents_idx}
        start_times = {a: args.past_length for a in agents_idx}



        while (future_traj.shape[1] - args.past_length < max_steps):
            with torch.no_grad():

                prediction, H = model.inference_simulator(traj_sample)
                normalized_vis, velocity, direction, edge_features, edge_weights = compute_features(args, traj_sample, "past")

                prediction_for_infer = prediction.view(20, 1, args.agent_num, args.future_length, 2).permute(1, 2, 3, 4, 0)  # (B, N, T, 2, 20)
                agent_missions_ten = torch.tensor(agent_missions, device=traj_sample.device)  # shape: [C]
                one_mission = agents_targs.gather(1, agent_missions_ten.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)


                eq_in, index = SM.inference(args.alpha, agents_idx, one_mission,constant.buffer, traj_sample, normalized_vis, velocity,
                                              direction, edge_features, edge_weights, prediction_for_infer)
                indices_expanded_SM = index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.future_length, 2, 1)
                selected_SM = torch.gather(prediction_for_infer, dim=4, index=indices_expanded_SM).squeeze(-1)
                new_traj_SM = selected_SM if args.classifier_method == 'sampler_selected' else eq_in  # B, N, T 2


                prediction_new_S, indixes_S = S.inference(traj_sample, normalized_vis, velocity, direction,edge_features,
                                                      edge_weights, prediction_for_infer)  # B, N, 10, 2
                indices_expanded_S = indixes_S.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.future_length, 2, 1)
                selected_S = torch.gather(prediction_for_infer, dim=4, index=indices_expanded_S).squeeze(-1)
                new_traj_S = selected_S if args.classifier_method == 'sampler_selected' else prediction_new_S  # B, N, T 2


                fake_traj = new_traj_S.clone()
                fake_traj[:, agents_idx, :, :] = new_traj_SM[:, agents_idx, :, :]
                scores = D(torch.cat([traj_sample, fake_traj], dim = 2)).squeeze(0) # B=1, N -> N


                if args.dataset in ['sdd', 'eth']:
                    fake_np = fake_traj[0].cpu().numpy()  # shape: [N, T, 2]
                    out_of_bounds_mask = (
                            (fake_np[:, :, 0] < X_MIN) | (fake_np[:, :, 0] > X_MAX) |
                            (fake_np[:, :, 1] < Y_MIN) | (fake_np[:, :, 1] > Y_MAX)
                    )
                    invalid_mask = (out_of_bounds_mask.sum(axis=1) > (fake_np.shape[1] // 3))  # shape: [N]
                else:
                    invalid_mask = torch.zeros(scores.shape, dtype=torch.bool)

                scores_masked = scores.clone()
                scores_masked[invalid_mask] = float('nan')
                score_list_sim.append(scores_masked)
                if invalid_mask.any(): # if there is any invalid agent - dont add it to calc for fixed number of N
                    pass
                else:
                    valid_fake_chunks = np.concatenate((valid_fake_chunks, fake_traj[0].cpu().numpy()), axis=1) #N, T, 2

            agents_to_remove = []

            for i in reversed(range(len(agents_idx))):  # reversed to safely remove
                agent = agents_idx[i]
                mission_id = agent_missions[i]
                target = agents_targs[i, mission_id]
                agent_path = fake_traj[0, agent]  # shape: [T, 2]

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
            future_traj = np.concatenate((future_traj, fake_traj[0].cpu().numpy()), axis=1)
            traj_sample = fake_traj[:, :, -args.past_length:, :]
            iter_num += 1


        if np.isnan(future_traj).any():
            continue
        centroids = future_traj.mean(axis=0)  # shape (T, 2)
        all_centroids.append(centroids)
        all_traj.append(future_traj)

        print("real all_traj len", future_traj.shape[1])
        if args.dataset in ['sdd', 'eth']:
            valid_fake_centroids = valid_fake_chunks.mean(axis=0) # [20, 2], [8,2]...
            all_valid_fake_centroids.append(valid_fake_centroids)  # S, T-changing, 2
            all_valid_fake_traj.append(valid_fake_chunks)
            print("fake all_traj len", valid_fake_chunks.shape[1])

        scores_tensor = torch.stack(score_list_sim, dim=0)  # shape: [T, N]
        mask = ~torch.isnan(scores_tensor)  # shape: [T, N], True where valid
        scores_tensor = torch.nan_to_num(scores_tensor, nan=0.0)
        #Sum and count
        sum_scores = (scores_tensor * mask).sum(dim=0)  # [N]
        count_valid = mask.sum(dim=0).clamp(min=1)  # [N], avoid division by 0
        sim_scores = sum_scores / count_valid

        score_list_all.append(sim_scores.cpu().numpy())  # collect into main list
        controlled_all.append(agents_idx.copy())

        sim_missions = len(controlled) * args.mission_num # C*missions for each agent -> total misssoin
        total_controlled += len(controlled)
        sim_achieved = len(mission_log) #total missions achieved
        total_missions += sim_missions # missions in all simulations
        missions_achieved += sim_achieved # missions achieved in all simulations
        if sim_achieved == sim_missions:
            full_success_count += 1 # in this sim all misssions achieved

        sim_success_rate = sim_achieved/sim_missions
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
            "avg_missions_per_agent": sim_achieved/ len(controlled),
            "mission_log": mission_log,
            "sim_success_rate": sim_success_rate,
        })

        if sim_id == 0:
            vis_predictions_missions(future_traj, mission_log, target_status, args, missions.cpu(), agents_idx_plot, stats_path)

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
        gen_gt_type = 'sampler'
    )

    df_per_sim = pd.DataFrame(sim_logs)

    avg_per_agent_all = df_per_sim["avg_missions_per_agent"].mean()
    std_per_agent_all = df_per_sim["avg_missions_per_agent"].std()
    avg_sim_success_rate = df_per_sim["sim_success_rate"].mean()
    std_sim_success_rate = df_per_sim["sim_success_rate"].std()

    n = len(df_per_sim)

    # 95% confidence interval
    confidence = 0.95
    z = stats.norm.ppf(1 - (1 - confidence) / 2)  # ≈ 1.96
    margin_of_error_sim = z * std_sim_success_rate / np.sqrt(n)
    margin_of_error_agent = z * std_per_agent_all / np.sqrt(n)

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

    if args.dataset == "nba":
        df_centroids.to_csv(f"{stats_path}/centroids_agents.csv")
    df_per_sim.to_csv(f"{stats_path}/per_sim_stats.csv")
    df_summary.to_csv(f"{stats_path}/overall_summary.csv")



def load_dataset(type, test_loader, args, model, S):
    DATASET_PATH = f"datasets/{args.dataset}/data/trajectory_dataset_sampler_GAN_{type}_{args.dataset}_{args.training_type}_{args.scene}_{args.sdd_scene}.pt"

    if os.path.exists(DATASET_PATH):
        print("Loading existing dataset...")
        traj_dataset = torch.load(DATASET_PATH, weights_only=False)
    else:
        print("Creating new dataset...")
        if args.dataset in ['nba', 'fish', 'syn']:
            traj_dataset = create_traj(test_loader, model, args, S)
        else:
            traj_dataset = create_traj_flex_N(test_loader, model, args, S)
        torch.save(traj_dataset, DATASET_PATH)

    print("traj_dataset", len(traj_dataset))
    if type == "train":
        new_batch_size = args.batch_size
    else:
        new_batch_size = args.batch_size * 32

    if args.dataset in ['nba', 'fish', 'syn']:
        loader = DataLoader(
            traj_dataset,
            batch_size=new_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=seq_collate_sampler_GAN)
    else:
        loader = DataLoader(traj_dataset, batch_sampler=GroupedBatchSampler(traj_dataset.grouped_seq_indices,
                                                                            batch_size=new_batch_size,
                                                                            shuffle=True,
                                                                            drop_last=False),
                            collate_fn=seq_collate_sampler_GAN)
    return loader


if __name__ == '__main__':
    args = parse_args()

    """ setup """
    names = [x for x in args.model_names.split(',')]

    if args.dataset == 'nba':
        args.model_save_dir = 'G1/saved_models/nba'
        args.agent_num = 11
        args.edge_num = 12
        args.mission_num = 12
        args.length =120
        d_set = NBADataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=args.training_type)

        constant = ConstantNBA

    elif args.dataset == 'fish':
        args.model_save_dir = 'G1/saved_models/fish'
        args.agent_num = 8
        args.edge_num = 16
        args.mission_num = 12
        args.length =120
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
        args.length =60
        args.model_save_dir = 'G1/saved_models/sdd'
        d_set = TrajectoryDatasetSDD(data_dir="./datasets/sdd", obs_len=args.past_length,
                                     pred_len=args.future_length, skip=1,
                                     min_ped=1, delim='space', save_path="./datasets/sdd/SDD.pt",
                                     mode=args.training_type)

        constant = ConstantSDD0
    elif args.dataset == 'eth':
        args.past_length = 8
        args.future_length = 12
        args.agent_num = 8
        args.model_save_dir = 'G1/saved_models/eth'
        d_set = TrajectoryDatasetETH(data_dir="./datasets/eth", obs_len=args.past_length,
                                     pred_len=args.future_length, skip=1, min_ped=1,
                                     delim='space', test_scene=args.scene, save_path=args.scene,
                                     mode=args.training_type)
        constant = return_the_eth_scene(args.scene)


    if args.dataset == 'nba' or args.dataset == 'fish' or args.dataset == 'syn':
        loader = DataLoader(
            d_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=seq_collate,
            pin_memory=True)


    for name in names:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        """ model """
        saved_path = os.path.join(args.model_save_dir, str(name) + '.p')
        print('load model from:', saved_path)
        checkpoint = torch.load(saved_path, map_location='cpu', weights_only=False)
        training_args = checkpoint['model_cfg']

        #for the sampler
        model = GroupNet(training_args, args.device)
        model.set_device(args.device)
        model.eval()
        model.load_state_dict(checkpoint['model_dict'], strict=True)
        model.args.sample_k = 20

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

        if args.mode == 'train':
            if args.dataset in ['sdd', 'eth']:
                loader = DataLoader(d_set,
                                          batch_sampler=GroupedBatchSampler(d_set.grouped_seq_indices_train,
                                                                            batch_size=args.batch_size, shuffle=False,
                                                                            drop_last=False), collate_fn=seq_collate)

            # mission aware model
            SM_path = f"SM/saved_model/{args.dataset}/{args.saved_models_SM}.pth"
            print('load model from:', SM_path)
            SM = SamplerMission(args, args.device, args.past_length - 1, args.future_length - 1, 32, 64,
                                args.past_length - 1
                                , args.future_length - 1,
                                32, 128, 1, 2, 8, edge_dim=16, bias=True).to(args.device)

            SM.load_state_dict(torch.load(SM_path, map_location=args.device, weights_only=False))
            SM.to(args.device)

            # classifier
            D = TrajectoryClassifier(args.device).to(args.device)
            saved_path = rf"Classifier/saved_model/Sampler/{args.dataset}/{args.scene}_{args.sdd_scene}_{args.training_type}_/{args.saved_models_DIS}.pth"
            D.load_state_dict(torch.load(saved_path, map_location=args.device, weights_only=False))
            D.to(args.device)


            extra_train_loader = load_dataset('train', loader, args, model, S)
            writer = SummaryWriter(log_dir=f"runs/GAN_training_sampler/{args.timestamp}_{args.dataset}_{args.scene}_{args.sdd_scene}_{args.training_type}_{args.seed}_{args.classifier_method}")

            train(constant, writer, extra_train_loader, args, SM, D)

        else:
            SEED = args.seed
            g = torch.Generator()
            g.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


            D = TrajectoryClassifier(args.device).to(args.device)
            saved_path = f"GANS/saved_model/{args.saved_models_GAN_DIS_SM}.pth"
            D.load_state_dict(torch.load(saved_path, map_location=args.device, weights_only=False))
            D.to(args.device)


            SM_path = f"GANS/saved_model/{args.saved_models_GAN_SM}.pth"
            SM.load_state_dict(torch.load(SM_path, weights_only=False)).to(args.device)


            if args.dataset in ['nba', 'syn', 'fish']:
                traj_dataset = torch.utils.data.Subset(d_set, range(args.sim_num))
            else:
                traj_dataset = Subset(d_set, constant.n_agents, args.sim_num, args.sdd_scene)
            data = prepare_target(args, args.device, traj_dataset,constant, "test")


            wrapped_test = MissionTestDataset(traj_dataset, data)
            testing_loader = DataLoader(
                wrapped_test,
                batch_size=args.batch_size,  # simulate one scenario at a time
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

            mission_test(testing_loader, args, model, S, SM, D ,constant)
            print("starting non mission test")
            non_mission_test(testing_loader, args,  model, S, SM, D , constant)



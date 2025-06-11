import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import scipy.stats as stats


import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from config import parse_args
from model.models_GAN import TrajectoryClassifier
from data.dataloader_fish import FISHDataset
from data.dataloader_syn import SYNDataset
from data.dataloader_SDD import TrajectoryDatasetSDD
from data.dataloader_ETH import TrajectoryDatasetETH, GroupedBatchSampler, seq_collate
from torch.utils.data import DataLoader
from model.GroupNet_nba_mission import GroupNetM
from model.GroupNet_nba import GroupNet
from loss.loss_groupnet_mission import LossCompute
from data.dataloader_GAN import TrajectoryDatasetMission, TrajectoryDatasetMissionUnorder, seq_collate_mission
from data.dataloader_nba import NBADataset
import time
from torch.utils.tensorboard import SummaryWriter
import math
from utilis import *


def create_traj(train_loader, model, args, mission_data =None):
    model.eval()

    if args.dataset in ['nba', 'fish', 'syn']:
        iter = 0
        past_traj = torch.empty((0, args.agent_num,  args.past_length, 2))  # total samples, 8 agents, 5 time steps, 2 coords
        old_future_traj = torch.empty((0, args.agent_num, args.future_length, 2))
        future_traj = torch.empty((0, args.agent_num, args.future_length, 2))

        for data in train_loader:
            with torch.no_grad():
                prediction, H = model.inference(data)  # 20, BN, T, 2 - > (B, N, 10, 2, 20)


            future_traj = torch.cat((future_traj, prediction.squeeze(0).view(data['past_traj'].shape[0], data['past_traj'].shape[1], args.future_length, 2).detach().cpu()), dim=0)
            past_traj = torch.cat((past_traj, data['past_traj'].detach().cpu()), dim=0)
            old_future_traj = torch.cat((old_future_traj, data['future_traj'].detach().cpu()), dim=0)

            iter += 1
            if iter % 100 == 0:
                print("iteration:", iter, "in data creation")

        traj_dataset = TrajectoryDatasetMission(past_traj, future_traj, old_future_traj)

    else:
        past_traj_list = []
        old_future_traj_list = []
        future_traj_list = []
        seq_start_end = []  # New seq start-end
        grouped_seq_indices = defaultdict(list)  # New grouped indices

        agent_counter = 0
        scene_counter = 0

        for data in train_loader: #B, N, T, 2 #NOT TO SHUFFLE
            with torch.no_grad():
                B, N, T, _ = data['past_traj'].shape
                prediction, H = model.inference(data)

            past_traj_list.append(data['past_traj'].view(B*N,T, 2).detach().cpu())
            old_future_traj_list.append(data['future_traj'].view(B*N,args.future_length, 2).detach().cpu())
            future_traj_list.append(prediction.squeeze(0).view(B*N,args.future_length, 2).detach().cpu())

            for _ in range(B):
                start = agent_counter
                end = agent_counter + N
                seq_start_end.append((start, end))
                agent_counter = end

                grouped_seq_indices[N].append(scene_counter)
                scene_counter += 1

        past_traj_all = torch.cat(past_traj_list, dim=0)  # (N, obs_len, 2)
        old_future_traj_all = torch.cat(old_future_traj_list, dim=0)  # (N, pred_len, 2)
        future_traj_all = torch.cat(future_traj_list, dim=0)  # (N, pred_len, 2)

        if mission_data:
            traj_dataset = TrajectoryDatasetMissionUnorder(
                past_traj_all,
                future_traj_all,
                old_future_traj_all,
                seq_start_end,
                grouped_seq_indices
            )

        else:
            traj_dataset = TrajectoryDatasetMissionUnorder(
                past_traj_all,
                future_traj_all,
                old_future_traj_all,
                seq_start_end,
                grouped_seq_indices, mission_data
            )

    return traj_dataset


def plot_losses(args, train_losses_g, train_losses_d, train_scores_real, train_scores_fake,train_scores_unco,
                val_losses_g, val_losses_d, val_scores_real, val_scores_fake):
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
    plt.savefig(f"GANG/plots/GAN_Generator_Loss_GroupNet_{args.dataset}_{args.classifier_method}_{args.scene}_{args.sdd_scene}_{args.training_type}_how_far_{args.how_far}_{args.test_mlp}_{args.info}.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses_d, label='Train Discriminator Loss', color='red')
    # plt.plot(epochs, val_losses_d, label='Val Discriminator Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Discriminator Loss')
    plt.legend()
    plt.title('Discriminator Loss Progression')
    plt.savefig(f"GANG/plots/GAN_Discriminator_Loss_GroupNet_{args.dataset}_{args.classifier_method}_{args.scene}_{args.sdd_scene}_{args.training_type}_how_far_{args.how_far}_{args.test_mlp}_{args.info}.png")
    plt.close()

    color_map = get_agent_colors(args, num_agents)
    controlled_agents = args.agent if isinstance(args.agent, (list, tuple)) else [args.agent]
    cols = 3
    rows = math.ceil(num_agents / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True)

    axs = axs.flatten()  # Flatten even if its 2D

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

    for ax in axs[num_agents:]:  # Hide unused axes
        ax.axis("off")

    plt.xlabel("Epochs")
    plt.suptitle("Train Discriminator Scores Per Agent", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        f"GANG/plots/Train_Discriminator_Scores_Per_Agent_Faceted_GroupNet_{args.dataset}_{args.classifier_method}_{args.scene}_{args.sdd_scene}_{args.training_type}_how_far_{args.how_far}_{args.test_mlp}_{args.info}.png")
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
    #     f"GANG/plots/Val_Discriminator_Scores_Per_Agent_Faceted_{args.timestamp}_{args.dataset}_{args.classifier_method}.png")
    # plt.close()

def train(constant, writer, train_loader, args, GM, D, mission_val_data=None, val_loader=None):
    print(args.timestamp)
    lossfn = LossCompute(args, netD=D, netGM=GM)
    optimizer_GM = torch.optim.Adam(GM.parameters(), lr=2e-4)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)
    scheduler_GM= torch.optim.lr_scheduler.StepLR(optimizer_GM, step_size=args.lr_step, gamma=args.lr_gamma)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=args.lr_step, gamma=args.lr_gamma)


    train_losses_g = []
    train_losses_d = []
    val_losses_g = []
    val_losses_d = []
    train_scores_real = []
    train_scores_fake = []
    train_scores_unco = []
    val_scores_real = []
    val_scores_fake = []
    val_scores_unco= []

    n_critic = args.n_critic
    for i in range(args.epoch_continue, args.epoch):
        iter_num = 0
        time_epoch = time.time()
        GM.train()
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
        div_mission_loss_mean = 0
        discriminator_loss_mean=0
        loss_mission_mean = 0
        loss_kl_mean = 0
        loss_recover_mean = 0
        loss_pred_mean = 0
        num_batches = 0

        for data in train_loader:
            iter_num += 1
            current_N = data['past_traj'].shape[1]
            agents_tragets, agents_idx, error_tolerance, _ = prepare_targets_mission_net(constant, data, args.dataset, args.device, 100)

            for _ in range(n_critic):
                optimizer_D.zero_grad()

                total_loss_d, loss_real, loss_fake, scores_fake, scores_real ,scores_uncontrolled= lossfn.compute_discriminator_loss(data,  agents_tragets, agents_idx, error_tolerance, i)

                total_loss_d.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(),
                                               max_norm=1.0)

                optimizer_D.step()


            optimizer_GM.zero_grad()
            tuple_loss = lossfn.compute_generator_loss(data, agents_tragets, agents_idx, error_tolerance, i)

            total_loss_g,loss_pred,loss_recover,loss_kl,loss_mission,loss_diverse, loss_mission_div, discriminator_loss  = tuple_loss

            mission_loss_mean += loss_mission
            div_mission_loss_mean += loss_mission_div
            discriminator_loss_mean += discriminator_loss
            loss_mission_mean +=loss_mission
            loss_kl_mean += loss_kl
            loss_recover_mean += loss_recover
            loss_pred_mean += loss_pred


            total_loss_g.backward()
            torch.nn.utils.clip_grad_norm_(GM.parameters(), max_norm=1.0)

            optimizer_GM.step()
            train_loss_g += total_loss_g.item()
            train_loss_d += total_loss_d.item() #the last one from n_critic



            scores_real_mean = scores_real.mean(dim=0).detach().cpu().numpy()  # N
            scores_fake_mean = scores_fake.mean(dim=0).detach().cpu().numpy()  # N,
            if agents_idx.numel() != 0:
                scores_uncontrolled_mean  = scores_uncontrolled.mean(dim=0).detach().cpu().numpy() #N
                uncontrolled_idx = [i for i in range(current_N) if i not in agents_idx.tolist()]
                for k, a in enumerate(uncontrolled_idx):  # the order is not important when dataset is not nba\target
                    train_unco_score[a] += scores_uncontrolled_mean[k]
                    unco_agent_counts[a] += 1
                if iter_num % args.iternum_print == 0:
                    print("scores_uncontrolled", scores_uncontrolled_mean, "\n")

            # train_real_score += scores_real_mean

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
                print("total_loss_d, ", total_loss_d.item(), "loss_real, " , loss_real, " loss_fake, ", loss_fake)
                print("scores_fake, ",  scores_fake_mean)
                print("scores_real", scores_real_mean)

                print("############## generator loss ")
                print("total_loss_g ", total_loss_g.item(),  " loss_pred ,", loss_pred,  "loss_recover ", loss_recover , "loss_kl " ,
                      loss_kl, "loss_mission " ,loss_mission, "loss_diverse ", loss_diverse,
                      "discriminator_loss", discriminator_loss)
                print("loss_mission_div", loss_mission_div)
                print("##########################################   \n")

        scheduler_GM.step()
        scheduler_D.step()


        # training metrics
        train_losses_g.append(train_loss_g / num_batches)
        train_losses_d.append(train_loss_d / num_batches)
        train_scores_real.append(train_real_score / np.clip(real_agent_counts, 1, None))
        train_scores_fake.append(train_fake_score / np.clip(fake_agent_counts, 1, None))
        train_scores_unco.append(train_unco_score / np.clip(unco_agent_counts, 1, None))
        mission_loss_mean =mission_loss_mean/num_batches
        div_mission_loss_mean = div_mission_loss_mean/num_batches
        discriminator_loss_mean =discriminator_loss_mean/num_batches
        loss_kl_mean = loss_kl_mean/num_batches
        loss_recover_mean = loss_recover_mean/num_batches
        loss_pred_mean = loss_pred_mean/num_batches

        # scalar losses
        writer.add_scalar("Loss/Train_Generator", train_losses_g[-1], i)
        writer.add_scalar("Loss/Train_Discriminator", train_losses_d[-1], i)

        for agent_id in range(args.agent_num):
            writer.add_scalar(f"Score/Train_Real_Agent_{agent_id}", train_scores_real[-1][agent_id], i)
            writer.add_scalar(f"Score/Train_Fake_Agent_{agent_id}", train_scores_fake[-1][agent_id], i)
            writer.add_scalar(f"Score/Train_Uncontrolled_Agent_{agent_id}", train_scores_unco[-1][agent_id], i)

        # Validation Phase
        # GM.eval()
        # D.eval()
        # val_loss_g = 0
        # val_loss_d = 0
        # val_real_score = np.zeros(args.agent_num)
        # val_fake_score = np.zeros(args.agent_num)
        # num_batches_val = 0
        # val_unco_score = np.zeros(args.agent_num)
        # mission_loss_mean_val = 0
        # discriminator_loss_mean_val = 0
        # div_mission_loss_mean_val = 0


        # with torch.no_grad():
        #     for data in val_loader:
        #         if args.dataset in ['nba', 'fish', 'syn']:
        #             batch_size = data['past_traj'].shape[0]
        #             missions = mission_val_data["targets"][num_batches_val].to(args.device)
        #             missions = missions.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        #             controlled_agents = mission_val_data["controlled_agents"][num_batches_val].to(args.device)
        #         else:
        #             missions = data["targets"].to(args.device) #N, M, 2
        #             controlled_agents = data["controlled_agents"].to(args.device)
        #
        #         agents_idx = controlled_agents[controlled_agents != -1]
        #         agents_tragets = missions[:,agents_idx, 0, :] #C, 2 -> take always the first mission.
        #
        #
        #         iter_num += 1
        #
        #
        #         total_loss_d, loss_real, loss_fake, scores_fake, scores_real, scores_uncontrolled = lossfn.compute_discriminator_loss(
        #                 data, agents_tragets, agents_idx, error_tolerance, i) #error tolerance from trian
        #
        #         tuple_loss = lossfn.compute_generator_loss(data, agents_tragets, agents_idx, error_tolerance, i)
        #         total_loss_g, loss_pred, loss_recover, loss_kl, loss_mission, loss_diverse, loss_mission_div, discriminator_loss = tuple_loss
        #
        #
        #         mission_loss_mean_val += loss_mission
        #         div_mission_loss_mean_val += loss_mission_div
        #         discriminator_loss_mean_val += discriminator_loss
        #
        #         val_loss_d += total_loss_d.item()
        #         val_loss_g += total_loss_g.item()
        #
        #         scores_real_mean = scores_real.mean(dim=0).detach().cpu().numpy()  # N
        #         scores_fake_mean = scores_fake.mean(dim=0).detach().cpu().numpy()  # N,
        #         scores_uncontrolled_mean = scores_uncontrolled.mean(dim=0).detach().cpu().numpy()  # N
        #
        #         val_real_score += scores_real_mean
        #         for k, a in enumerate(agents_idx):
        #             val_fake_score[a] += scores_fake_mean[k]
        #             fake_agent_counts_val[a] += 1
        #         uncontrolled_idx = list(set(range(args.agent_num)) - set(agents_idx))
        #         for k, a in enumerate(uncontrolled_idx):
        #             val_unco_score[a] += scores_uncontrolled_mean[k]
        #             unco_agent_counts_val[a] += 1
        #         num_batches_val += 1
        #
        # # Store validation metrics
        # val_losses_g.append(val_loss_g /num_batches_val)
        # val_losses_d.append(val_loss_d / num_batches_val)
        # val_scores_real.append(val_real_score / num_batches_val)
        # val_scores_fake.append(val_fake_score /  np.clip(fake_agent_counts_val, 1, None))
        # val_scores_unco.append(val_unco_score / np.clip(unco_agent_counts_val, 1, None))
        # mission_loss_mean_val = mission_loss_mean_val / num_batches_val
        # div_mission_loss_mean_val = div_mission_loss_mean_val/num_batches_val
        # discriminator_loss_mean_val = discriminator_loss_mean_val / num_batches_val
        #
        # print("Validation mission: ", mission_loss_mean_val, "diverse mission validation: ", div_mission_loss_mean_val)
        #
        # writer.add_scalar("Loss/Val_Generator", val_losses_g[-1], i)
        # writer.add_scalar("Loss/Val_Discriminator", val_losses_d[-1], i)
        #
        # for agent_id in range(args.agent_num):
        #     writer.add_scalar(f"Score/Val_Real_Agent_{agent_id}", val_scores_real[-1][agent_id], i)
        #     writer.add_scalar(f"Score/Val_Fake_Agent_{agent_id}", val_scores_fake[-1][agent_id], i)
        #     writer.add_scalar(f"Score/Val_Uncontrolled_Agent_{agent_id}", val_scores_unco[-1][agent_id], i)
        time_end = time.time()

        if (i + 1) % args.save_every == 0:
            saveModel_real(GM, D, args, str(i + 1))

        print(
            f"Epoch [{i + 1}/{args.epoch}] - "
            f"Train Loss G: {train_losses_g[-1]:.4f}, D: {train_losses_d[-1]:.4f}, \n"
            
            f"######################## scores: \n"
            f"Train Real Score (targeted agents): {np.round(train_scores_real[-1], 4)},"
            f"Fake Score (targeted agents): {np.round(train_scores_fake[-1], 4)},"
            f"Uncontrolled Score (targeted agents): {np.round(train_scores_unco[-1], 4)},\n"
            
            f"######## missions \n"
            f"average mission_loss : {mission_loss_mean} , "
            f"average diverse mission loss : {div_mission_loss_mean} ,"
            f"discriminator_loss_mean: {discriminator_loss_mean} \n, "
            
            f"########################## Trajectory loss \n"
            f"loss_kl_mean : {loss_kl_mean} , "
            f"loss_recover_mean : {loss_recover_mean} ,"
            f"loss_pred_mean: {loss_pred_mean} \n, "
            
            f"Time: {time_end - time_epoch:.2f}s"
        )


    plot_losses(args, train_losses_g, train_losses_d, train_scores_real, train_scores_fake,train_scores_unco,
                val_losses_g, val_losses_d, val_scores_real, val_scores_fake)


def plot_score_list(score_list, args):
    score_array = np.array(score_list)  # shape: (steps, N)
    num_agents = score_array.shape[1]
    timesteps = 5 * 0.4 + np.arange(len(score_array)) * 10 * 0.4

    color_map = get_agent_colors(args, num_agents)
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
    plt.savefig(f"GANG/plots/faceted_score_list_{args.timestamp}_{args.dataset}_{args.classifier_method}.png")
    plt.close()


def non_mission_test(test_loader, args, G, D, GM, constant):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    stats_path = create_stats_folder(args, constant, 'groupnet-gan')

    G.eval()
    D.eval()
    GM.eval()

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
    all_valid_fake_traj= []
    X_MIN, X_MAX = constant.X_MIN, constant.X_MAX
    Y_MIN, Y_MAX = constant.Y_MIN, constant.Y_MAX

    for sim_id in range(len(testing_loader)):

        traj_sample, _, _, _ = test_loader.dataset[sim_id]  # traj_sample: [N, T, 2], missions: [N, M, 2], controlled: [max_controlled]
        score_list_sim = []

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
                fake_traj, _ = GM.inference_simulator(traj_sample, one_mission, torch.tensor(agents_idx, device=traj_sample.device), mission_tolerance)
                # fake_traj, _ = G.inference_simulator(traj_sample)


                scores = D(torch.cat([traj_sample, fake_traj], dim = 2)).squeeze(0)  # B=1, N -> N

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

        print("real all_traj len", future_traj.shape[1])
        if args.dataset in ['sdd', 'eth']:
            valid_fake_centroids = valid_fake_chunks.mean(axis=0) # [20, 2], [8,2]...
            all_valid_fake_centroids.append(valid_fake_centroids)  # S, T-changing, 2
            all_valid_fake_traj.append(valid_fake_chunks)
            print("fake all_traj len", valid_fake_chunks.shape[1])

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

    print(f"\n--- Final Statistics over {len(test_loader)} simulations ---")
    print(df_summary.T)

    if args.dataset == "nba":
        df_centroids.to_csv(
            f"{stats_path}/no_mission/centroids_agents.csv")
    df_per_sim.to_csv(
        f"{stats_path}/no_mission/per_sim_stats.csv")
    df_summary.to_csv(
        f"{stats_path}/no_mission/overall_summary.csv")

    return df_summary, df_per_sim, df_centroids







def mission_test(testing_loader, args, G, GM, D, constant):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    stats_path = create_stats_folder(args, constant, 'groupnet-gan')

    G.eval()
    D.eval()
    GM.eval()

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
    sims_success_rate= []

    all_valid_fake_centroids = []
    all_valid_fake_traj= []
    X_MIN, X_MAX = constant.X_MIN, constant.X_MAX
    Y_MIN, Y_MAX = constant.Y_MIN, constant.Y_MAX

    for sim_id in range(len(testing_loader)):#

        traj_sample, missions, controlled , _= testing_loader.dataset[sim_id]  # traj_sample: [N, T, 2], missions: [N, M, 2], controlled: [max_controlled]

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
        score_list_sim = []
        invalid_score_mask_sim = []  # stores one [N] mask per T step
        valid_fake_chunks = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]

        future_traj = traj_sample[0].detach().cpu().numpy()  # [N, T, 2]
        agent_mission_times = {a: [] for a in agents_idx}
        start_times = {a: args.past_length for a in agents_idx}

        while  (future_traj.shape[1] - args.past_length < max_steps):
            with torch.no_grad():
                agent_missions_ten = torch.tensor(agent_missions, device=traj_sample.device)  # shape: [C]
                one_mission = agents_targs.gather(1, agent_missions_ten.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)

                prediction_regular, _ = G.inference_simulator(traj_sample)
                predictions_covert, _ = GM.inference_simulator(traj_sample, one_mission, torch.tensor(agents_idx, device=traj_sample.device), mission_tolerance)

                fake_traj = prediction_regular.clone()
                fake_traj[:, agents_idx, :, :] = predictions_covert[: ,agents_idx, :, :]

                scores =D(torch.cat([traj_sample, fake_traj], dim = 2)).squeeze(0) # B=1, N -> N


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
        all_centroids.append(centroids) # S, T, 2
        all_traj.append(future_traj)

        # print("real all_traj len", future_traj.shape[1])
        if args.dataset in ['sdd', 'eth']:
            valid_fake_centroids = valid_fake_chunks.mean(axis=0) # [20, 2], [8,2]...
            all_valid_fake_centroids.append(valid_fake_centroids)  # S, T-changing, 2
            all_valid_fake_traj.append(valid_fake_chunks)
            # print("fake all_traj len", valid_fake_chunks.shape[1])

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
            "sim_success_rate":sim_success_rate
        })

        if sim_id == 0:
            vis_predictions_missions(constant, future_traj, mission_log, target_status, args, missions.cpu(), agents_idx_plot, stats_path)

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
        gen_gt_type = 'groupnet'
    )

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
        df_centroids.to_csv(
            f"{stats_path}/centroids_agents.csv")
    df_per_sim.to_csv(f"{stats_path}/per_sim_stats.csv")
    df_summary.to_csv(f"{stats_path}/overall_summary.csv")

    # return df_summary, df_per_sim, df_centroids



def load_dataset(mode, train_loader, args, model, val_loader =None, mission_data =None):

    if mode == 'train':

        DATASET_PATH = f"datasets/{args.dataset}/data/trajectory_dataset_for_GAN_mission_{mode}_{args.training_type}_{args.scene}_{args.sdd_scene}.pt"
        DATASET_PATH_VAL = f"datasets/{args.dataset}/data/trajectory_dataset_for_GAN_mission_val_{args.training_type}_{args.scene}_{args.sdd_scene}.pt"
        if os.path.exists(DATASET_PATH):
            print("Loading existing dataset...")
            traj_dataset = torch.load(DATASET_PATH, weights_only=False)
            traj_dataset_val = torch.load(DATASET_PATH_VAL, weights_only=False)
        else:
            print("Creating new datasets...")
            traj_dataset = create_traj(train_loader, model, args)
            torch.save(traj_dataset, DATASET_PATH)

            traj_dataset_val = create_traj(val_loader, model, args,   mission_data)
            torch.save(traj_dataset_val, DATASET_PATH_VAL)


        if args.dataset in ['nba', 'fish', 'syn']:
            new_train_loader = DataLoader(
                traj_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=seq_collate_mission,
                pin_memory=True)
            new_val_loader = DataLoader(
                traj_dataset_val,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=seq_collate_mission,
                pin_memory=True)
        else:
            new_train_loader = DataLoader(traj_dataset, batch_sampler=GroupedBatchSampler(traj_dataset.grouped_seq_indices,
                                                                                   batch_size=args.batch_size,
                                                                                   shuffle=True,
                                                                                   drop_last=False),
                                      collate_fn=seq_collate_mission)
            new_val_loader = DataLoader(traj_dataset_val, batch_sampler=GroupedBatchSampler(traj_dataset_val.grouped_seq_indices,
                                                                               batch_size=args.batch_size,
                                                                               shuffle=True,
                                                                               drop_last=False), collate_fn=seq_collate_mission)
        return new_train_loader, new_val_loader

    else: #test data
        DATASET_PATH = f"datasets/{args.dataset}/data/trajectory_dataset_for_GAN_mission_{mode}_{args.dataset}_{args.training_type}_{args.scene}_{args.sdd_scene}.pt"
        if os.path.exists(DATASET_PATH):
            print("Loading existing dataset...")
            traj_dataset = torch.load(DATASET_PATH, weights_only=False)
        else:
            print("Creating new datasets...")
            traj_dataset = create_traj(train_loader, model, args)
            torch.save(traj_dataset, DATASET_PATH)
        if args.dataset in ['nba', 'fish', 'syn']:
            new_train_loader = DataLoader(
                traj_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=seq_collate_mission,
                pin_memory=True)
        else:
            new_train_loader = DataLoader(traj_dataset, batch_sampler=GroupedBatchSampler(traj_dataset.grouped_seq_indices,
                                                                                   batch_size=args.batch_size,
                                                                                   shuffle=True,
                                                                                   drop_last=False),
                                      collate_fn=seq_collate_mission)

        return new_train_loader


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
        d_set = NBADataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=args.training_type)
        val_set = NBADataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part="val")
        constant = ConstantNBA

    elif args.dataset =='fish':
        args.model_save_dir = 'G1/saved_models/fish'
        args.agent_num = 8
        args.edge_num = 16
        args.mission_num =12
        args.length = 120
        d_set = FISHDataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=args.training_type)
        val_set = FISHDataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part="val")
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
        val_set = SYNDataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=val_part)

    elif args.dataset == 'sdd':
        args.past_length = 8
        args.future_length = 12
        args.agent_num = 8
        args.mission_num =8
        args.length = 60
        args.model_save_dir = 'G1/saved_models/sdd'
        d_set = TrajectoryDatasetSDD(data_dir="datasets/sdd", obs_len=args.past_length,
                                         pred_len=args.future_length, skip=1,
                                         min_ped=1, delim='space', save_path="datasets/sdd/SDD.pt",
                                         mode=args.training_type)
        val_set = TrajectoryDatasetSDD(data_dir="datasets/sdd", obs_len=args.past_length,
                                         pred_len=args.future_length, skip=1,
                                         min_ped=1, delim='space', save_path="datasets/sdd/SDD.pt",
                                         mode='val')
        constant = ConstantSDD0
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
        val_set = TrajectoryDatasetETH(data_dir="datasets/eth", obs_len=args.past_length,
                                         pred_len=args.future_length, skip=1, min_ped=1,
                                         delim='space', test_scene=args.scene, save_path=args.scene,
                                         mode='test')

    if args.dataset == 'nba' or args.dataset == 'fish' or args.dataset == 'syn':
        train_loader = DataLoader(
            d_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=seq_collate,
            pin_memory=True)
        val_loader = DataLoader(
            val_set,
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
        checkpoint = torch.load(saved_path, map_location=args.device, weights_only=False)
        training_args = checkpoint['model_cfg']

        #regular simulator
        G = GroupNet(training_args, args.device)
        G.set_device(args.device)
        G.eval()
        G.load_state_dict(checkpoint['model_dict'], strict=True)
        G.args.sample_k = 1


        if args.mode == 'train':
            if args.dataset == 'sdd':
                train_loader = DataLoader(d_set,
                                          batch_sampler=GroupedBatchSampler(d_set.grouped_seq_indices_train,
                                                                            batch_size=args.batch_size, shuffle=False,
                                                                            drop_last=False), collate_fn=seq_collate)
                val_loader = DataLoader(val_set, batch_sampler=GroupedBatchSampler(val_set.grouped_seq_indices_val,
                                                                                   batch_size=args.batch_size,
                                                                                   shuffle=False,
                                                                                   drop_last=False),  collate_fn=seq_collate)
            elif args.dataset == 'eth':
                train_loader = DataLoader(d_set,
                                          batch_sampler=GroupedBatchSampler(d_set.grouped_seq_indices_train,
                                                                            batch_size=args.batch_size,
                                                                            shuffle=False,
                                                                            drop_last=False), collate_fn=seq_collate)

            D = TrajectoryClassifier(args.device).to(args.device)
            GM = GroupNetM(args, args.device).to(args.device)

            if args.epoch_continue > 0:
                GM_path = f"GANG/saved_model/{args.dataset}/{args.saved_models_GM}.pth"
                D_path = f"GANG/saved_model/{args.dataset}/{args.saved_models_DIS}.pth"

                print(f'load model from: {D_path} and {GM_path}')
                Dstate_dict  = torch.load(D_path, map_location=args.device, weights_only=False)
                D.load_state_dict(Dstate_dict)
                D.to(args.device)

                GMstate_dict  = torch.load(GM_path, map_location=args.device, weights_only=False)
                GM.load_state_dict(GMstate_dict)
                GM.to(args.device)

            else:
                # mission aware model
                GM_path = f"GM/saved_models/{args.dataset}/{args.saved_models_GM}.p"
                checkpoint = torch.load(GM_path, map_location=args.device, weights_only=False)
                GM.set_device(args.device)
                GM.load_state_dict(checkpoint['model_dict'], strict=True)

                # D = DiscV2(args.agent_num, args.device, args.past_length - 1,32, 64,
                #         args.past_length - 1, 32, 128, 1, 2, 8, edge_dim=16, bias=True).to(args.device)

                saved_path = f"Classifier/saved_model/{args.dataset}/{args.scene}_{args.sdd_scene}_{args.training_type}_/{args.saved_models_DIS}.pth"
                D.load_state_dict(torch.load(saved_path, map_location=args.device, weights_only=False))
                D.to(args.device)

            mission_val_data = prepare_target_GAN(constant , args,args.device, val_set, "val")

            ###for getting the G predictions from the data.
            if args.dataset in ['nba', 'fish', 'syn']:
                new_train_loader, new_val_loader = load_dataset('train', train_loader, args, G, val_loader =val_loader)
            elif args.dataset == 'eth':
                new_train_loader = load_dataset('train_no_val', train_loader, args, G)
            else:
                new_train_loader, new_val_loader = load_dataset('train', train_loader, args, G, val_loader =val_loader, mission_data = mission_val_data)


            writer = SummaryWriter(log_dir=f"runs/GAN_training/{args.timestamp}_{args.dataset}_{args.scene}_{args.sdd_scene}_{args.training_type}_{args.seed}_{args.classifier_method}")

            if args.dataset in ['nba', 'fish', 'syn', 'sdd']:
                train(constant, writer, new_train_loader, args, GM, D, mission_val_data, new_val_loader)
            else:
                train(constant, writer, new_train_loader, args, GM, D, mission_val_data)


        else:
            SEED = args.seed
            g = torch.Generator()
            g.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # mission aware model
            GM = GroupNetM(args, args.device).to(args.device)
            GM_path = f"GANG/saved_model/{args.dataset}/{args.saved_models_GAN_GM}.pth"

            GM.load_state_dict(torch.load(GM_path, map_location=args.device, weights_only=False))
            GM.to(args.device)



            # D = DiscV2(args.agent_num, args.device, args.past_length - 1,32, 64,
            #         args.past_length - 1, 32, 128, 1, 2, 8, edge_dim=16, bias=True).to(args.device)
            D = TrajectoryClassifier(args.device).to(args.device)
            saved_path = f"GANG/saved_model/{args.dataset}/{args.saved_models_GAN_DIS}.pth"

            D.load_state_dict(torch.load(saved_path, map_location=args.device,  weights_only=False))
            D.to(args.device)

            if args.dataset in ['nba', 'syn', 'fish']:
                traj_dataset = torch.utils.data.Subset(d_set, range(args.sim_num))
            else:
                loader = DataLoader(d_set,
                                          batch_sampler=GroupedBatchSampler(d_set.grouped_seq_indices_test,
                                                                            batch_size=args.batch_size, shuffle=False,
                                                                            drop_last=False), collate_fn=seq_collate) #trainset is actually test (controled in args)

                traj_dataset = Subset(d_set, constant.n_agents, args.sim_num, args.sdd_scene)
            data = prepare_target(args, args.device, traj_dataset, constant, "test")


            wrapped_test = MissionTestDataset(traj_dataset, data)
            testing_loader = DataLoader(
                wrapped_test,
                batch_size=args.batch_size,  # simulate one scenario at a time
                shuffle=False,
                num_workers=0,
                pin_memory=True,generator=g,
            )
            GM.args.sample_k = 1
            mission_test(testing_loader, args, G, GM, D, constant)
            print("starting non mission test")
            non_mission_test(testing_loader, args, G, D, GM, constant)



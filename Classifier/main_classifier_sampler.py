import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from data.dataloader_fish import FISHDataset
from data.dataloader_nba import NBADataset, seq_collate
from torch.utils.data import DataLoader
from model.models_sampler import SamplerMission, Sampler, SamplerMLP

from model.GroupNet_nba import GroupNet
import torch.optim as optim
import torch.nn as nn
from Classifier.config_classifier import parse_args
from Classifier.model_classifier import TrajectoryClassifier
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utilis import *
from data.dataloader_GAN import TrajectoryDatasetClassifierSamplerFlex, TrajectoryDatasetClassifierSamplerNoneFlexN, seq_collate_classifier_sampler
from data.dataloader_SDD import TrajectoryDatasetSDD
from data.dataloader_ETH import TrajectoryDatasetETH, GroupedBatchSampler, seq_collate
from data.dataloader_syn import SYNDataset


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:

            predictions_real = batch['predictions_real'].to(args.device)
            predictions_controlled = batch['predictions_controlled'].to(args.device)
            controlled_idx = batch['controlled_idx'].to(args.device)

            controlled_idx = controlled_idx[0][controlled_idx[0] != -1]
            predictions_controlled = predictions_controlled[:, controlled_idx, :, :]

            outputs_real = model(predictions_real)  # (B, N)
            probs_real = torch.sigmoid(outputs_real)
            pred_real = (probs_real > 0.5).view(-1).cpu().numpy()
            label_real = np.ones_like(pred_real)

            if len(controlled_idx.tolist()) != 0:
                outputs_fake = model(predictions_controlled)
                probs_fake = torch.sigmoid(outputs_fake)
                pred_fake = (probs_fake > 0.5).view(-1).cpu().numpy()
                label_fake = np.zeros_like(pred_fake)
                y_true.extend(label_real.tolist() + label_fake.tolist())
                y_pred.extend(pred_real.tolist() + pred_fake.tolist())
            else:
                y_true.extend(label_real.tolist())
                y_pred.extend(pred_real.tolist())


    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall (Sensitivity): {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def plot_losses(args, train_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    # plt.plot(epochs, val_losses, label='Validation Loss', color='red')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f"Classifier/plots/loss_calssifier_{args.method}_{args.length}_{args.dataset}_{args.classifier_method}.png")


def train(args, D, optimizer, criterion, train_loader, num_epochs, device):
    train_losses = []
    D.train()

    for epoch in range(num_epochs):
        time_epoch = time.time()

        total_loss_list = 0
        num_batches = 0
        iter_num =0
        for batch in train_loader:
            past_traj = batch['past_traj'].to(args.device)
            future_traj = batch['future_traj'].to(args.device)
            future_mean_list = batch['future_mean_list'].to(args.device)
            future_first_list = batch['future_first_list'].to(args.device)
            fake_list = batch['fake_list'].to(args.device)
            predictions_real = batch['predictions_real'].to(args.device)
            predictions_controlled = batch['predictions_controlled'].to(args.device)
            controlled_idx = batch['controlled_idx'].to(args.device)

            true_outputs = D(torch.cat([past_traj,predictions_real], dim = 2)) #B, N
            true_future_traj = D(torch.cat([past_traj,future_traj], dim = 2))
            extra_fake_outputs = D(torch.cat([past_traj,fake_list], dim = 2))
            future_mean_fake = D(torch.cat([past_traj,future_mean_list], dim = 2))
            future_first_fake = D(torch.cat([past_traj,future_first_list], dim = 2))

            controlled_idx = controlled_idx[0][controlled_idx[0] != -1]

            if len(controlled_idx.tolist() ) != 0:
                predictions_controlled = predictions_controlled[:, controlled_idx, :, :]
                past_controlled = past_traj[:, controlled_idx, :, :]
                predictions_controlled_fake = D(torch.cat([past_controlled,predictions_controlled],dim = 2)) #B, C
                y_fake_controlled = torch.zeros_like(predictions_controlled_fake).to(args.device)
                loss_fake_controlled = criterion(predictions_controlled_fake, y_fake_controlled)

            y_real = torch.ones_like(true_outputs).to(args.device)
            y_real_original = torch.ones_like(true_future_traj).to(args.device)

            y_fake_extra = torch.zeros_like(extra_fake_outputs).to(args.device)
            y_fake_mean = torch.zeros_like(future_mean_fake).to(args.device)
            y_fake_first = torch.zeros_like(future_first_fake).to(args.device)



            loss_real = criterion(true_outputs, y_real)
            loss_real_origin = criterion(true_future_traj, y_real_original)

            loss_fake_extra = criterion(extra_fake_outputs, y_fake_extra)
            loss_fake_mean = criterion(future_mean_fake, y_fake_mean)
            loss_fake_first = criterion(future_first_fake, y_fake_first)

            if len(controlled_idx.tolist()) != 0:
                total_loss = loss_real + loss_real_origin + loss_fake_extra + loss_fake_mean + loss_fake_first + loss_fake_controlled
                if iter_num % args.iternum_print == 0:
                    print(f"loss_fake_controlled {loss_fake_controlled.item():.3f}")
                    print(f"scores fake {predictions_controlled_fake.mean()}")

            else:
                total_loss = loss_real + loss_real_origin + loss_fake_extra + loss_fake_mean + loss_fake_first

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_list += total_loss.item()
            num_batches += 1

            if iter_num % args.iternum_print == 0:
                print(f" total_loss: {total_loss:.3f}, loss_real: {loss_real.item():.3f}, loss_real_origin: {loss_real_origin.item():.3f},"
                      f"loss_fake_extra: {loss_fake_extra.item():.3f}, loss_fake_mean: {loss_fake_mean.item():.3f}, loss_fake_first: {loss_fake_first.item():.3f},  "
                      f"scores real: {true_outputs.mean()}")

            iter_num =+ 1
        time_end = time.time()

        avg_train_loss = total_loss_list / num_batches
        train_losses.append(avg_train_loss)

        if (epoch + 1) % args.save_every == 0:
            saveModel_DS( D,  args, str(epoch + 1))
        # Validation
        # avg_val_loss = validate(model, criterion, val_loader, device)
        # val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Time: {time_end-time_epoch:.2f}s ")

    # plot Losses
    plot_losses(args, train_losses)

def create_traj_flex_classifier(SM, test_loader,S, model, args, constant):
    init_sample = 0
    model.eval()
    SM.eval()
    S.eval()

    past_traj, future_traj, predictions_real, predictions_controlled = [], [], [], []
    group_net = []
    all_agents_idx = []
    ew_list, ef_list, dir_list, vel_list, vis_list, idx_list = [], [], [], [], [], []

    future_mean_list, future_first_list, fake_list = [], [], []
    methods = ['linear', 'random_walk', 'noise']

    for data in test_loader:
        with torch.no_grad():
            model.args.sample_k = 20
            prediction_20, _ = model.inference(data)  # 20, BN, T, 2
            B, N, T, _ = data['past_traj'].shape

            visability_mat_past, velocity_past, direction_past, edge_features_past, edge_weights_past = compute_features(
                args,data['past_traj'], "past")

            past_traj_ten = data['past_traj'].to(args.device)
            visability_mat_past = visability_mat_past.to(args.device)
            velocity_past = velocity_past.to(args.device)
            direction_past = direction_past.to(args.device)
            edge_features_past = edge_features_past.to(args.device)
            edge_weights_past = edge_weights_past.to(args.device)

            prediction_for_infer = prediction_20.view(20, B, N, args.future_length, 2).permute(1, 2, 3, 4,
                                                                                                            0)  # (B, N, T, 2, 20)
            prediction_new, indixes = S.inference(past_traj_ten, visability_mat_past, velocity_past, direction_past,
                                                  edge_features_past,
                                                  edge_weights_past, prediction_for_infer)  # B, N, 10, 2
            indices_expanded = indixes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.future_length, 2,1)
            selected = torch.gather(prediction_for_infer, dim=4, index=indices_expanded).squeeze(-1)

            y = data['future_traj'].reshape(B * N, args.future_length,2)  # 200, 10, 2
            y = y.unsqueeze(0).repeat(20, 1, 1, 1).to(args.device)
            error = torch.norm(y - prediction_20, dim=3).mean(dim=2)  # 20, BN,
            indices = torch.argmin(error, dim=0).view(B, N)  # BN


            agents_tragets, agents_idx, _, _ = prepare_targets_mission_net(constant, data, args.dataset, args.device, 100)  # high epoch
            prediction_controlled, indixes_controlled = SM.inference(None, agents_idx.to(args.device), agents_tragets.to(args.device), constant.buffer,
                                                                     past_traj_ten, visability_mat_past,
                                                                     velocity_past, direction_past, edge_features_past,
                                                                     edge_weights_past, prediction_for_infer)
            indices_expanded_c = indixes_controlled.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1,
                                                                                                     args.future_length,
                                                                                                     2, 1)
            selected_controlled = torch.gather(prediction_for_infer, dim=4, index=indices_expanded_c).squeeze(-1)




        agents_idx = agents_idx.tolist()
        padded_batch = agents_idx + [-1] * (N - len(agents_idx))
        padded_tensor = torch.tensor(padded_batch).unsqueeze(0).repeat(B, 1)

        future_mean = torch.mean(prediction_20, dim=0).view(B, N,args.future_length, 2)  # ( B ,N,10,2)
        future_first = prediction_20[0, :, :, :].view(B, N, args.future_length,2)  # ( B,N,10,2)

        fake_method = methods[init_sample % len(methods)]
        start_points = data['past_traj'][:, :, -1, :].to(args.device)
        end_points = selected[:, :, -1, :].to(args.device)  #the last predicted future points

        if fake_method == 'linear':
            fake_traj = torch.stack([
                torch.stack([
                    linear_interpolation(start_points[b, n], end_points[b, n], args.future_length)
                    for n in range(N)
                ], dim=0)
                for b in range(B)
            ], dim=0).to(args.device)
        elif fake_method == 'random_walk':
            fake_traj = torch.stack([
                torch.stack([
                    random_walk(start_points[b, n], args.future_length)
                    for n in range(N)
                ], dim=0)
                for b in range(B)
            ], dim=0).to(args.device)
        elif fake_method == 'noise':
            real_traj = selected.view(-1, args.future_length, 2).to(
                args.device)
            fake_traj = noise_addition(real_traj).view(B, N, args.future_length,
                                                       2).to( args.device)

        past_traj.extend(data['past_traj'].cpu() )
        future_traj.extend(data['future_traj'].cpu() )
        group_net.extend(prediction_for_infer.cpu() )
        ew_list.extend(edge_weights_past.cpu() )
        ef_list.extend(edge_features_past.cpu() )
        dir_list.extend(direction_past.cpu() )
        vel_list.extend(velocity_past.cpu() )
        vis_list.extend(visability_mat_past.cpu() )
        idx_list.extend(indices.cpu() )
        predictions_real.extend(selected.cpu() )
        predictions_controlled.extend(selected_controlled.cpu() )
        future_first_list.extend(future_first.cpu())
        future_mean_list.extend(future_mean.cpu())
        fake_list.extend(fake_traj.cpu())
        all_agents_idx.extend(padded_tensor.cpu())
        # print("all_agents_idx", all_agents_idx)

        init_sample += 1
        if init_sample % 100 == 0:
            print("iteration:", init_sample, "in data creation")

    seq_start_end = []
    grouped_seq_indices = defaultdict(list)
    agent_counter = 0
    for scene_i, traj in enumerate(past_traj):
        Ni = traj.shape[0]
        start = agent_counter
        end = start + Ni
        seq_start_end.append((start, end))
        grouped_seq_indices[Ni].append(scene_i)
        agent_counter = end

    # print("all_agents_idx", len(all_agents_idx) )
    device = args.device
    all_past = torch.cat(past_traj, dim=0).to(device)  # (sum Ni, T, 2)
    all_future = torch.cat(future_traj, dim=0).to(device)  # (sum Ni, T, 2)
    all_groupnet = torch.cat(group_net, dim=0).to(device)  # (sum Ni, T,2,20)
    all_future_mean_list = torch.cat(future_mean_list, dim=0).to(device)  # (sum Ni,)
    all_future_first_list = torch.cat(future_first_list, dim=0).to(device)
    all_predictions_real = torch.cat(predictions_real, dim=0).to(device)  # (sum Ni,T)
    all_predictions_controlled  = torch.cat(predictions_controlled, dim = 0).to(device)
    all_fake_list = torch.cat(fake_list, dim=0).to(device)
    all_all_agents_idx = torch.cat(all_agents_idx, dim=0).to(device) #N, 1
    all_edge_w = torch.cat(ew_list, dim=0).to(device)  # (sum Ni,)
    all_edge_f = torch.cat(ef_list, dim=0).to(device)
    all_dir = torch.cat(dir_list, dim=0).to(device)  # (sum Ni,T)
    all_vel = torch.cat(vel_list, dim=0).to(device)
    all_vis = vis_list  # (S, N, N)
    all_indices = torch.cat(idx_list, dim=0).to(device)  # (sum Ni,)



    traj_dataset = TrajectoryDatasetClassifierSamplerFlex(all_past, all_future,all_groupnet, all_edge_w, all_edge_f, all_dir, all_vel, all_vis , all_indices,
                                            all_future_mean_list, all_future_first_list,
                                          all_predictions_real, all_predictions_controlled, all_fake_list,all_all_agents_idx, grouped_seq_indices,seq_start_end)

    return traj_dataset


def create_traj(SM, test_loader, S, model, args, constant):
    init_sample = 0
    model.eval()
    SM.eval()
    S.eval()

    future_traj = torch.empty((0, args.agent_num, args.future_length, 2)).to(args.device)
    future_mean_list = torch.empty((0, args.agent_num, args.future_length, 2)).to(args.device)
    future_first_list = torch.empty((0, args.agent_num, args.future_length, 2)).to(args.device)
    fake_list = torch.empty((0, args.agent_num, args.future_length, 2)).to(args.device)
    predictions_controlled = torch.empty((0, args.agent_num, args.future_length, 2)).to(args.device)
    predictions_real = torch.empty((0, args.agent_num, args.future_length, 2)).to(args.device)


    past_traj = torch.empty((0, args.agent_num, args.past_length, 2)).to(
        args.device)  # total samples, 8 agents, 5 time steps, 2 coords
    group_net = torch.empty((0, args.agent_num, args.future_length, 2, 20)).to(
        args.device)  # (Total_samples, N, 10, 2, 20)

    edge_weights_past_list = torch.empty((0, args.agent_num)).to(args.device)
    edge_features_past_list = torch.empty((0, args.agent_num)).to(args.device)
    direction_past_list = torch.empty((0, args.agent_num, args.past_length-1)).to(args.device)
    velocity_past_list = torch.empty((0, args.agent_num, args.past_length -1)).to(args.device)
    visability_mat_past_list = torch.empty((0, args.agent_num, args.agent_num)).to(args.device)
    indexes_list = torch.empty((0, args.agent_num)).to(args.device)


    methods = ['linear', 'random_walk', 'noise']
    for data in test_loader:
        with torch.no_grad():
            model.args.sample_k = 20
            prediction_20, _ = model.inference(data) #20, BN, T, 2

            visability_mat_past, velocity_past, direction_past, edge_features_past, edge_weights_past = compute_features(args,
                data['past_traj'], "past")

            past_traj_ten = data['past_traj'].to(args.device)
            visability_mat_past = visability_mat_past.to(args.device)
            velocity_past = velocity_past.to(args.device)
            direction_past = direction_past.to(args.device)
            edge_features_past = edge_features_past.to(args.device)
            edge_weights_past = edge_weights_past.to(args.device)

            prediction_for_infer = prediction_20.view(20, data['future_traj'].shape[0], args.agent_num, args.future_length, 2).permute(1, 2, 3, 4,0)  # (B, N, T, 2, 20)
            prediction_new, indixes = S.inference( past_traj_ten, visability_mat_past, velocity_past, direction_past,
                                                  edge_features_past,
                                                  edge_weights_past, prediction_for_infer)  # B, N, 10, 2
            indices_expanded = indixes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.future_length, 2,1)
            selected = torch.gather(prediction_for_infer, dim=4, index=indices_expanded).squeeze(-1)


            y = data['future_traj'].reshape(data['future_traj'].shape[0] * args.agent_num, args.future_length,2)  # 200, 10, 2
            y = y.unsqueeze(0).repeat(20, 1, 1, 1).to(args.device)
            error = torch.norm(y - prediction_20, dim=3).mean(dim=2)  # 20, BN,
            indices = torch.argmin(error, dim=0)  # BN

            indexes_list = torch.cat((indexes_list, indices.view(data['future_traj'].shape[0], args.agent_num)), dim=0)

            edge_weights_past_list = torch.cat((edge_weights_past_list, edge_weights_past.to(args.device)), dim=0)
            edge_features_past_list = torch.cat((edge_features_past_list, edge_features_past.to(args.device)), dim=0)
            direction_past_list = torch.cat((direction_past_list, direction_past.to(args.device)), dim=0)
            velocity_past_list = torch.cat((velocity_past_list, velocity_past.to(args.device)), dim=0)
            visability_mat_past_list = torch.cat((visability_mat_past_list, visability_mat_past.to(args.device)), dim=0)

            future_traj = torch.cat((future_traj, data['future_traj'].to(args.device)), dim=0)

            group_net = torch.cat((group_net, prediction_for_infer.to(args.device)), dim=0)
            past_traj = torch.cat((past_traj, data['past_traj'].to(args.device)), dim=0)
            predictions_real = torch.cat((predictions_real, selected), dim=0)


            agents_tragets, agents_idx, _, _ = prepare_targets_mission_net(constant, data, args.dataset,device, 100) #high epoch
            prediction_controlled, indixes_controlled = SM.inference(None, agents_idx.to(args.device), agents_tragets.to(args.device), constant.buffer, past_traj_ten, visability_mat_past,
                                                    velocity_past, direction_past,edge_features_past,edge_weights_past, prediction_for_infer)
            indices_expanded_c = indixes_controlled.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.future_length, 2,1)
            selected_controlled = torch.gather(prediction_for_infer, dim=4, index=indices_expanded_c).squeeze(-1)
            predictions_controlled = torch.cat((predictions_controlled, selected_controlled), dim=0)

        padded_row = torch.full((args.agent_num,), -1, dtype=torch.long, device=args.device)
        num_agents = len(agents_idx)
        padded_row[:num_agents] = agents_idx.to(args.device)
        padded_agents_idx = padded_row.unsqueeze(0).repeat(data["future_traj"].shape[0], 1)  # shape (B, N)

        if init_sample == 0:
            all_agents_idx = padded_agents_idx.clone()  # shape ( B, N)
        else:
            all_agents_idx = torch.cat((all_agents_idx, padded_agents_idx), dim=0)

        future_mean = torch.mean(prediction_20, dim=0).view(data["future_traj"].shape[0], args.agent_num,
                                                         args.future_length, 2)  # ( B ,N,10,2)
        future_first = prediction_20[0, :, :, :].view(data["future_traj"].shape[0], args.agent_num, args.future_length,
                                                   2)  # ( B,N,10,2)
        future_first_list = torch.cat((future_first_list, future_first), dim=0)
        future_mean_list = torch.cat((future_mean_list, future_mean), dim=0)

        fake_method = methods[init_sample % len(methods)]
        start_points = data['past_traj'][:, :, -1, :].to(args.device)
        end_points = selected[:, :, -1, :].to(args.device)

        if fake_method == 'linear':
            fake_traj = torch.stack([
                torch.stack([
                    linear_interpolation(start_points[b, n], end_points[b, n], args.future_length)
                    for n in range(args.agent_num)
                ], dim=0)
                for b in range(data['future_traj'].shape[0])
            ], dim=0).to(args.device)
        elif fake_method == 'random_walk':
            fake_traj = torch.stack([
                torch.stack([
                    random_walk(start_points[b, n], args.future_length)
                    for n in range(args.agent_num)
                ], dim=0)
                for b in range(data['future_traj'].shape[0])
            ], dim=0).to(args.device)
        elif fake_method == 'noise':
            real_traj = selected.view(-1, args.future_length, 2).to(args.device)
            fake_traj = noise_addition(real_traj).view(data['future_traj'].shape[0], args.agent_num, args.future_length, 2).to(
                args.device)

        init_sample += 1
        fake_list = torch.cat((fake_list, fake_traj), dim=0)
        if init_sample % 100 == 0:
            print("iteration:", init_sample, "in data creation")
    traj_dataset = TrajectoryDatasetClassifierSamplerNoneFlexN(past_traj, future_traj,group_net, edge_weights_past_list, edge_features_past_list,
                                         direction_past_list, velocity_past_list, visability_mat_past_list,indexes_list,
                                                               future_mean_list, future_first_list,predictions_real,
                                                       predictions_controlled, fake_list, all_agents_idx)
    return traj_dataset

def linear_interpolation(start, end, steps):
    vector = end - start
    step_vector = vector / (steps - 1)
    trajectory = torch.stack([start + i * step_vector for i in range(steps)])
    return trajectory

def random_walk(start, steps, scale=0.1):
    trajectory = [start]
    current = start
    for _ in range(1, steps):
        step = torch.randn_like(current) * scale
        current = current + step
        trajectory.append(current)
    return torch.stack(trajectory)

def noise_addition(real_traj, noise_level=0.5):
    noise = torch.randn_like(real_traj) * noise_level
    return real_traj + noise


def load_dataset(type, loader, args, model, S,SM, train_path, constant):

    if os.path.exists(train_path):
        print(f"Loading existing dataset from {train_path}...")
        data = torch.load(train_path, weights_only=False)
    else:
        print(f"No Dataset was found, creating {train_path}...")
        if args.dataset in ['nba', 'fish', 'syn']:
            data = create_traj(SM, loader,S, model, args, constant)
        else:
            data = create_traj_flex_classifier(SM, loader, S, model, args, constant)

        torch.save(data, train_path)
        print(f"Dataset saved at {train_path}")

    print("traj_dataset", len(data))
    if type == "train":
        new_batch_size = args.batch_size
    else:
        new_batch_size = args.batch_size*32

    if args.dataset in ['nba', 'fish', 'syn']:
        loader = DataLoader(
            data,
            batch_size=new_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=seq_collate_classifier_sampler)
    else:
        loader = DataLoader(data, batch_sampler=GroupedBatchSampler(data.grouped_seq_indices, batch_size=new_batch_size,
                                                                    shuffle=False, drop_last=False),
                            collate_fn=seq_collate_classifier_sampler)

    return loader



if __name__ == '__main__':
    args = parse_args()

    """ setup """
    names = [x for x in args.model_names.split(',')]
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    load_device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    args.device = device

    for name in names:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


        if args.dataset == 'nba':
            args.model_save_dir = 'G1/saved_models/nba'
            args.agent_num = 11
            d_set = NBADataset(
                obs_len=args.past_length,
                pred_len=args.future_length,
                dataset_part=args.training_type)

            constant = ConstantNBA
        elif args.dataset == 'fish':
            args.model_save_dir = 'G1/saved_models/fish'
            args.agent_num = 8
            d_set = FISHDataset(
                obs_len=args.past_length,
                pred_len=args.future_length,
                dataset_part=args.training_type)
            constant = ConstantFish
        elif args.dataset == 'syn':
            args.model_save_dir = 'G1/saved_models/syn'
            args.agent_num = 6
            d_set = SYNDataset(
                obs_len=args.past_length,
                pred_len=args.future_length,
                dataset_part=args.training_type)
            if args.training_type == 'train_rigid':
                constant = ConstantSYNR
            else:
                constant = ConstantSYNS
        elif args.dataset == 'sdd':
            args.model_save_dir = 'G1/saved_models/sdd'
            args.past_length = 8
            args.future_length = 12
            constant = ConstantSDD0
            d_set = TrajectoryDatasetSDD(data_dir="datasets/sdd", obs_len=args.past_length,
                                         pred_len=args.future_length, skip=1,
                                         min_ped=1, delim='space', save_path="datasets/sdd/SDD.pt",
                                         mode=args.training_type)
        elif args.dataset == 'eth':
            args.model_save_dir = 'G1/saved_models/eth'
            args.past_length = 8
            args.future_length = 12
            constant = return_the_eth_scene(args.scene)
            d_set = TrajectoryDatasetETH(data_dir="datasets/eth", obs_len=args.past_length,
                                             pred_len=args.future_length, skip=1, min_ped=1,
                                             delim='space', test_scene=args.scene, save_path=args.scene,
                                             mode=args.training_type)

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

        #non mission aware
        S_path = f"Sampler/saved_model/{args.dataset}/{args.saved_models_SAM}.pth"
        S.load_state_dict(torch.load(S_path, map_location=args.device, weights_only=False))
        S.to(args.device)


        #mission aware
        SM_path = f"SM/saved_model/{args.dataset}/{args.saved_models_SM}.pth"
        print('load model from:', SM_path)
        SM = SamplerMission(args, args.device, args.past_length - 1, args.future_length - 1, 32, 64,
                            args.past_length - 1
                            , args.future_length - 1,
                            32, 128, 1, 2, 8, edge_dim=16, bias=True).to(args.device)

        SM.load_state_dict(torch.load(SM_path, map_location=args.device, weights_only=False))
        SM.to(args.device)


        #classifier
        D = TrajectoryClassifier(args.device).to(args.device)
        D.to(args.device)

        pos_weight = torch.tensor([3.0], device=args.device)  # adjust based on actual imbalance
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(D.parameters(), lr=0.001)

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
                                                                             shuffle=False,
                                                                             drop_last=False), collate_fn=seq_collate)


            dataset_train_path = f"Classifier/data/classifier_trajectories_sampler_{args.dataset}_{args.scene}_{args.training_type}.pt"
            new_loader = load_dataset('train', loader, args, model, S, SM, dataset_train_path, constant)
            train(args ,D, optimizer, criterion, new_loader, args.epoch, args.device)

        else:

            SEED = args.seed
            g = torch.Generator()
            g.manual_seed(SEED)

            D_path = f"{args.model_dir}/Sampler/{args.dataset}/{args.saved_models_DIS}.pth"
            D.eval()
            D.load_state_dict(torch.load(D_path, map_location=args.device, weights_only=False))

            if args.dataset == 'nba' or args.dataset == 'fish' or args.dataset == 'syn':
                loader = DataLoader(
                    d_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=seq_collate,generator=g)
            else:
                loader = DataLoader(d_set, batch_sampler=GroupedBatchSampler(d_set.grouped_seq_indices_test,
                                                                             batch_size=args.batch_size,
                                                                             shuffle=False,
                                                                             drop_last=False), collate_fn=seq_collate,generator=g)

            dataset_train_path = f"Classifier/data/classifier_trajectories_sampler_{args.dataset}_{args.scene}_{args.training_type}.pt"
            new_loader = load_dataset('test', loader, args, model,S, SM, dataset_train_path, constant)

            test(D, new_loader, args.device)

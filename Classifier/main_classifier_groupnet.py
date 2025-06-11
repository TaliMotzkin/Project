import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from data.dataloader_fish import FISHDataset
from data.dataloader_nba import NBADataset, seq_collate
from torch.utils.data import DataLoader

from model.GroupNet_nba import GroupNet
import torch.optim as optim
import torch.nn as nn
from Classifier.config_classifier import parse_args
from Classifier.model_classifier import TrajectoryClassifier
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utilis import *
from data.dataloader_GAN import TrajectoryDatasetClassifierGroupnetFlex, TrajectoryDatasetClassifierGroupnet, seq_collate_classifier
from data.dataloader_SDD import TrajectoryDatasetSDD
from data.dataloader_ETH import TrajectoryDatasetETH, GroupedBatchSampler, seq_collate
from data.dataloader_syn import SYNDataset
from model.GroupNet_nba_mission import GroupNetM


def test(model, test_loader, device):
    model.eval()

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
    val_losses = []
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
            saveModel_D( D,  args, str(epoch + 1))
        # Validation
        # avg_val_loss = validate(model, criterion, val_loader, device)
        # val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Time: {time_end-time_epoch:.2f}s ")

    # plot Losses
    plot_losses(args, train_losses)

def create_traj_flex_classifier(GM, test_loader, model, args, constant):
    init_sample = 0
    model.eval()
    GM.eval()
    past_traj, future_traj, predictions_real, predictions_controlled = [], [], [], []
    group_net = []
    all_agents_idx = []

    future_mean_list, future_first_list, fake_list = [], [], []
    methods = ['linear', 'random_walk', 'noise']
    for data in test_loader:
        with torch.no_grad():
            model.args.sample_k = 20
            prediction_20, _ = model.inference(data)  # 20, BN, T, 2
            model.args.sample_k = 1
            prediction_real, _ = model.inference(data)  # 1, BN, T, 2

            agents_tragets, agents_idx, _, _ = prepare_targets_mission_net(constant, data, args.dataset, args.device, 100)  # high epoch
            GM.args.sample_k = 1
            prediction_controlled, _ = GM.inference(data, agents_tragets, agents_idx, constant.buffer)  # 1, BN, T, 2


        B , N, T, _= data["future_traj"].shape

        agents_idx = agents_idx.tolist()
        padded_batch = agents_idx + [-1] * (N - len(agents_idx))
        # print("padded_batch", padded_batch)
        padded_tensor = torch.tensor(padded_batch).unsqueeze(0).repeat(B, 1)
        # print("padded_tensor", padded_tensor)

        future_mean = torch.mean(prediction_20, dim=0).view(B, N,args.future_length, 2)  # ( B ,N,10,2)
        future_first = prediction_20[0, :, :, :].view(B, N, args.future_length,2)  # ( B,N,10,2)
        prediction_real = prediction_real.squeeze(0).view(B, N, args.future_length,2).to(args.device)
        prediction_controlled = prediction_controlled.squeeze(0).view(B, N, args.future_length, 2).to(args.device)
        pred_per_scene = prediction_20.permute(1, 2, 3, 0).view(B, N,args.future_length, 2, 20)

        fake_method = methods[init_sample % len(methods)]
        start_points = data['past_traj'][:, :, -1, :].to(args.device)
        end_points = prediction_real[:, :, -1, :].to(args.device)  #the last predicted future points

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
            real_traj = prediction_real.view(-1, args.future_length, 2).to(
                args.device)
            fake_traj = noise_addition(real_traj).view(B, N, args.future_length,
                                                       2).to( args.device)


        future_first_list.extend(future_first.cpu())
        future_mean_list.extend(future_mean.cpu())
        future_traj.extend(data['future_traj'].cpu())
        past_traj.extend(data['past_traj'].cpu())
        predictions_real.extend(prediction_real.cpu())
        predictions_controlled.extend(prediction_controlled.cpu())
        group_net.extend(pred_per_scene.cpu())
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

    # print("all_all_agents_idx", all_all_agents_idx.shape)
    # print("predictions_real", len(predictions_real))
    # print("all_predictions_real", all_predictions_real.shape)


    traj_dataset = TrajectoryDatasetClassifierGroupnetFlex(all_groupnet, all_past, all_future, all_future_mean_list, all_future_first_list,
                                          all_predictions_real, all_predictions_controlled, all_fake_list,grouped_seq_indices,seq_start_end,all_all_agents_idx)

    return traj_dataset


def create_traj(GM, test_loader, model, args, constant):
    init_sample = 0
    model.eval()
    GM.eval()

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


    methods = ['linear', 'random_walk', 'noise']
    for data in test_loader:
        with torch.no_grad():
            model.args.sample_k = 20
            prediction_20, _ = model.inference(data) #20, BN, T, 2
            model.args.sample_k = 1
            prediction_real, _ = model.inference(data) #1, BN, T, 2

            agents_tragets, agents_idx, _, _ = prepare_targets_mission_net(constant, data, args.dataset,device, 100) #high epoch
            GM.args.sample_k = 1
            prediction_controlled, _ = GM.inference(data, agents_tragets, agents_idx,constant.buffer)  # 1, BN, T, 2

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

        prediction_real = prediction_real.squeeze(0).view(data["future_traj"].shape[0], args.agent_num, args.future_length, 2).to(args.device)
        future_traj = torch.cat((future_traj, data['future_traj'].to(args.device)), dim=0)
        predictions_real = torch.cat((predictions_real,prediction_real), dim=0)

        prediction_controlled = prediction_controlled.squeeze(0).view(data["future_traj"].shape[0], args.agent_num, args.future_length, 2).to(args.device)
        predictions_controlled = torch.cat((predictions_controlled,prediction_controlled), dim=0)


        prediction = prediction_20.permute(1, 2, 3, 0).view(data['future_traj'].shape[0], args.agent_num, args.future_length, 2, 20)
        group_net = torch.cat((group_net, prediction.to(args.device)), dim=0)
        past_traj = torch.cat((past_traj, data['past_traj'].to(args.device)), dim=0)


        fake_method = methods[init_sample % len(methods)]
        start_points = data['past_traj'][:, :, -1, :].to(
            args.device)
        end_points = prediction_real[:, :, -1, :].to(args.device)

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
            real_traj = prediction_real.view(-1, args.future_length, 2).to(args.device)
            fake_traj = noise_addition(real_traj).view(data['future_traj'].shape[0], args.agent_num, args.future_length, 2).to(
                args.device)

        init_sample += 1
        fake_list = torch.cat((fake_list, fake_traj), dim=0)
        if init_sample % 100 == 0:
            print("iteration:", init_sample, "in data creation")
    traj_dataset = TrajectoryDatasetClassifierGroupnet(group_net, past_traj, future_traj, future_mean_list, future_first_list,predictions_real,
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


def load_dataset(type, loader, args, model, GM, train_path, constant):

    if os.path.exists(train_path):
        print(f"Loading existing dataset from {train_path}...")
        data = torch.load(train_path, weights_only=False)
    else:
        print(f"No Dataset was found, creating {train_path}...")
        if args.dataset in ['nba', 'fish', 'syn']:
            data = create_traj(GM, loader, model, args, constant)
        else:
            data = create_traj_flex_classifier(GM, loader, model, args, constant)

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
            collate_fn=seq_collate_classifier)
    else:
        loader = DataLoader(data, batch_sampler=GroupedBatchSampler(data.grouped_seq_indices, batch_size=new_batch_size,
                                                                    shuffle=False, drop_last=False),
                            collate_fn=seq_collate_classifier)

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
            if args.training_type == 'G1/train_rigid':
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

        model = GroupNet(training_args, args.device)
        model.set_device(args.device)
        model.eval()
        model.load_state_dict(checkpoint['model_dict'], strict=True)
        model.args.sample_k = args.sample_k

        GM_path = f"GM/saved_models/{args.dataset}/{args.saved_models_GM}.p"
        print('load model from:', GM_path)
        checkpointGM = torch.load(GM_path, map_location='cpu', weights_only=False)
        training_argsGM = checkpointGM['model_cfg']
        GM = GroupNetM(training_argsGM, device)
        GM.set_device(device)
        GM.eval()
        GM.load_state_dict(checkpointGM['model_dict'], strict=True)
        GM.args.sample_k = args.sample_k

        D = TrajectoryClassifier(args.device).to(args.device)
        D.to(args.device)
        pos_weight = torch.tensor([3.0], device=args.device)  # adjusted based on actual imbalance
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


            dataset_train_path = f"Classifier/data/classifier_trajectories_{args.dataset}_{args.scene}_{args.training_type}.pt"
            new_loader = load_dataset('train', loader, args, model, GM, dataset_train_path, constant)
            train(args ,D, optimizer, criterion, new_loader, args.epoch, args.device)

        else:

            SEED = args.seed
            g = torch.Generator()
            g.manual_seed(SEED)

            D_path = f"{args.model_dir}{args.dataset}/{args.saved_models_DIS}.pth"
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

            dataset_train_path = f"Classifier/data/classifier_trajectories_{args.dataset}_{args.scene}_{args.training_type}.pt"
            new_loader = load_dataset('test', loader, args, model, GM, dataset_train_path, constant)

            test(D, new_loader, args.device)

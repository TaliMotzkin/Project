import os
import sys
import argparse
import time
import numpy as np
import torch
import random
from torch import optim
from torch.optim import lr_scheduler

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from torch.utils.data import DataLoader
from data.dataloader_fish import FISHDataset, seq_collate
from data.dataloader_nba import NBADataset, seq_collate
from data.dataloader_syn import SYNDataset, seq_collate
from data.dataloader_SDD import TrajectoryDatasetSDD, GroupedBatchSampler, seq_collate
from data.dataloader_ETH import TrajectoryDatasetETH, GroupedBatchSampler, seq_collate
from model.GroupNet_nba import GroupNet
import matplotlib.pyplot as plt

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', default='nba')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--past_length', type=int, default=5)
parser.add_argument('--future_length', type=int, default=10)
parser.add_argument('--traj_scale', type=int, default=1)
parser.add_argument('--learn_prior', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--sample_k', type=int, default=20)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--decay_step', type=int, default=10)
parser.add_argument('--decay_gamma', type=float, default=0.5)
parser.add_argument('--iternum_print', type=int, default=50)

parser.add_argument('--ztype', default='gaussian')
parser.add_argument('--zdim', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--hyper_scales', nargs='+', type=int, default=[5,11])

parser.add_argument('--num_decompose', type=int, default=2)
parser.add_argument('--min_clip', type=float, default=2.0)

parser.add_argument('--model_save_dir', default='G1/saved_models/nba')
parser.add_argument('--model_save_epoch', type=int, default=10)

parser.add_argument('--epoch_continue', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--agent_num', type=int, default=11)
parser.add_argument('--training_type', type=str, default="train_1")
parser.add_argument('--scene', type=str, default="")
parser.add_argument('--info', type=str, default="")

args = parser.parse_args()

if args.dataset == 'nba':
    args.model_save_dir = 'G1/saved_models/nba'
    args.hyper_scales = [5,11]
    args.agent_num = 11
elif args.dataset == 'fish':
    args.model_save_dir = 'G1/saved_models/fish'
    args.hyper_scales = [3, 5]
    args.agent_num = 8
elif args.dataset == 'syn':
    args.model_save_dir = 'G1/saved_models/syn'
    args.hyper_scales = [3, 5]
    args.agent_num = 6
elif args.dataset == 'sdd':
    args.model_save_dir = 'G1/saved_models/sdd'
    args.hyper_scales = [4, 8]
    args.past_length =8
    args.future_length= 12
elif args.dataset == 'eth':
    args.model_save_dir = 'G1/saved_models/eth'
    args.past_length =8
    args.future_length= 12
    args.hyper_scales = [4, 8]

""" setup """
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.set_default_dtype(torch.float32)
device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
print('device:', device)
print(args)
args.device  = device


def validate(validation_loader):
    model.eval()
    total_val_loss = 0
    iter = 0
    with torch.no_grad():
        for data in validation_loader:
            total_loss, loss_pred, loss_recover, loss_kl, loss_diverse, diverse_pred_traj = model(data)
            total_val_loss += total_loss.item()
            iter += 1

    avg_val_loss = total_val_loss / iter
    print(f'total avg validation Loss: {avg_val_loss:.3f}')
    print("other val losses: loss pred", loss_pred, "loss recover ", loss_recover, "loss kl ", loss_kl, "loss diverse ",
          loss_diverse)
    return avg_val_loss


def train(train_loader, epoch, batch_size, device):
    model.train()
    total_iter_num = len(train_loader)
    iter_num = 0
    avg_loss = 0
    all_loss = 0
    for data in train_loader:
        total_loss, loss_pred, loss_recover, loss_kl, loss_diverse, predictions = model(
            data)

        """ optimize """
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        B_N, S, T, C = predictions.shape
        N = B_N// batch_size

        if iter_num % args.iternum_print == 0:
            print(
                'Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Total loss: {:03f}| Loss_pred: {:03f}| Loss_recover: {:03f}| Loss_kl: {:03f}| Loss_diverse: {:03f} |'
                .format(epoch, args.num_epochs, iter_num, total_iter_num, total_loss.item(), loss_pred, loss_recover,
                        loss_kl, loss_diverse))
            # print("loss_dist", loss_dist)
        iter_num += 1
        all_loss += total_loss.item()

    # validate_loss = validate(validation_loader)
    scheduler.step()
    model.step_annealer()
    avg_loss = all_loss / iter_num

    return avg_loss





def ploting_losses(train_loss, valid_loss, file_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss[1:], label='Training Loss', color='blue')
    # plt.plot(valid_loss[1:], label='Validation Loss', color='red')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path)
        plt.close()
    else:
        plt.show()

""" model & optimizer """
model = GroupNet(args, device)
print("params model", model.parameters())
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)

""" dataloader """

if args.dataset == 'nba':
    train_set = NBADataset(
        obs_len=args.past_length,
        pred_len=args.future_length,
        dataset_part=args.training_type)
elif args.dataset == 'fish':
    train_set = FISHDataset(
        obs_len=args.past_length,
        pred_len=args.future_length,
        dataset_part=args.training_type)
elif args.dataset == 'syn':
    train_set = SYNDataset(
        obs_len=args.past_length,
        pred_len=args.future_length,
        dataset_part=args.training_type)
elif args.dataset == 'sdd':
    train_set = TrajectoryDatasetSDD(data_dir="datasets/SDD/raw", obs_len= args.past_length, pred_len=args.future_length, skip=1,
                                     min_ped=1, delim='space', save_path="datasets/sdd/SDD.pt", mode=args.training_type)
elif args.dataset == 'eth':
    train_set = TrajectoryDatasetETH(data_dir="datasets/eth/raw", obs_len=args.past_length, pred_len=args.future_length, skip=1, min_ped=0,
                                     delim='space', test_scene=args.scene, save_path=args.scene, mode = args.training_type)

if args.dataset == 'nba' or args.dataset == 'fish' or args.dataset == 'syn':
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate,
        pin_memory=True)
else:
    train_loader = DataLoader(train_set, batch_sampler=GroupedBatchSampler(train_set.grouped_seq_indices_train, batch_size=args.batch_size,shuffle=True,
        drop_last=False), collate_fn=seq_collate)


""" Loading if needed """
if args.epoch_continue > 0:
    checkpoint_path = os.path.join(args.model_save_dir, str(args.epoch_continue) + ''+ '.p')
    print('load model from: {checkpoint_path}')
    model_load = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_load['model_dict'])
    # model.set_device(device)
    if 'optimizer' in model_load:
        optimizer.load_state_dict(model_load['optimizer'])
    if 'scheduler' in model_load:
        scheduler.load_state_dict(model_load['scheduler'])

""" start training """
model.set_device(device)
args.device = device
avg_validate_losses = []
avg_train_losses = []

for epoch in range(args.epoch_continue, args.num_epochs):
    start_time = time.time()
    avg_loss = train(train_loader, epoch, args.batch_size, device)
    print("avg loss", avg_loss)
    end_time = time.time()
    print("time elapsed", end_time - start_time)
    avg_train_losses.append(avg_loss)

    """ save model """
    if (epoch + 1) % args.model_save_epoch == 0:
        model_saved = {'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(), 'epoch': int(epoch) + 1, 'model_cfg': args}
        saved_path = os.path.join(args.model_save_dir, str(int(epoch) + 1)+ '_'+ args.dataset +args.scene + '_' + args.training_type + '_'+ str(args.seed) + "_"+ args.info +'.p')
        torch.save(model_saved, saved_path)

ploting_losses(avg_train_losses, avg_validate_losses, f'G1/plots/training_loss_groupnet_{args.dataset}_{args.training_type}_{args.seed}.png')

import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import sys
import datetime
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from data.dataloader_nba import NBADataset

sys.path.append(os.getcwd())
from data.dataloader_fish import FISHDataset, seq_collate
from data.dataloader_syn import  SYNDataset, seq_collate

from data.dataloader_SDD import TrajectoryDatasetSDD, GroupedBatchSampler, seq_collate
from data.dataloader_ETH import TrajectoryDatasetETH, GroupedBatchSampler, seq_collate
from utilis import *




def mission_test(dataset_test, args,constant):

    stats_path = create_stats_folder(args, constant, "GT")

    if args.dataset in ['sdd', 'eth']:
        dataset_sub = Subset(dataset_test, args.testing_num_agents, 1000000, args.sdd_scene)
        if args.dataset == 'sdd':
            field = 1900
            dataset = dataset_sub.get_all_trajectories_sdd(args)
        else:
            field = 25
            dataset = dataset_sub.get_all_trajectories_eth(args)
    else:
        dataset = dataset_test.for_GT()

    # sim_length = int(args.length /0.4)
    # number_of_timeataps = args.sim_num * sim_length
    past_future_length = (args.future_length + args.past_length)
    # number_of_samples = int(number_of_timeataps / past_future_length)

    # current_dataset = dataset[:number_of_samples, :, :, :] # 2000, 11, 15, 2
    current_dataset = dataset
    current_dataset_length = current_dataset.shape[0]
    N = current_dataset.shape[2]
    reshaped_dataset = current_dataset.reshape(current_dataset_length*past_future_length, N, 2) #S*20, N, 2

    centroids = reshaped_dataset.mean(axis=1) #2000*15, 2
    # all_traj = [np.asarray(sample) for sample in current_dataset]
    # all_centroids = [traj.mean(axis=1) for traj in all_traj]  #shape (2000, T, 2)

    print("starting analysis of usage")


    if args.dataset == 'nba':
        field = 29.9
    elif args.dataset == "target":
        field = 90
    elif args.dataset == "syn":
        field = 50
    df_centroids = analyze_usage_GT(
        all_trajectories = current_dataset, #B, T, N, 2
        all_centroids=centroids,  # originaly list of np.ndarray, shape [B*T, 2]
        field_length= field,
        num_blocks=10,
        timestep_duration=0.4,
        args=args,
        folder = stats_path,
        constant = constant
    )

    if args.dataset=='nba':
        df_centroids.to_csv(f"{stats_path}/centroids_agents.csv")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model_names', default=None)
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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--agent_num', type=int, default=11)
    parser.add_argument('--sim_num', type=int, default=100)
    parser.add_argument('--mission_num', type=int, default=5)
    parser.add_argument('--testing',action='store_true', default=False)
    parser.add_argument('--mission_buffer',type=float, default=1)
    parser.add_argument('--agent',  nargs='+', type=int, default=[0 , 2])
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument('--info', type=str, default="")
    parser.add_argument('--max_covert', type=int, default=5)
    parser.add_argument('--scene', type=str, default='')
    parser.add_argument('--learn_prior', action='store_true', default=False)
    parser.add_argument('--testing_num_agents', type=int, default=8)
    parser.add_argument('--sdd_scene', type=int, default=None)

    args = parser.parse_args()
    """ setup """

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)
    args.device = device

    if args.dataset == 'nba':
        args.model_save_dir = 'saved_models/nba'
        args.agent_num = 11
        constant = ConstantNBA
        test_dset = NBADataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=args.training_type)

    elif args.dataset == "target":
        args.model_save_dir = 'saved_models/fish_overlap'
        args.agent_num = 8
        constant = ConstantFish
        test_dset = FISHDataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=args.training_type)


    elif args.dataset == 'syn':
        args.model_save_dir = 'saved_models/syn'
        args.agent_num = 6
        if args.training_type == 'test_rigid':
            constant = ConstantSYNR
        else:
            constant = ConstantSYNS
        test_dset = SYNDataset(
            obs_len=args.past_length,
            pred_len=args.future_length,
            dataset_part=args.training_type)

    elif args.dataset == 'sdd':
        args.past_length = 8
        args.future_length = 12
        args.model_save_dir = 'saved_models/sdd'
        constant = return_the_sdd_scene(args.sdd_scene)

        test_dset = TrajectoryDatasetSDD(data_dir="datasets/SDD/raw", obs_len=args.past_length,
                                         pred_len=args.future_length, skip=1,
                                         min_ped=1, delim='space', save_path="datasets/sdd/SDD.pt",
                                         mode=args.training_type)

    elif args.dataset == 'eth':
        args.past_length = 8
        args.future_length = 12
        args.model_save_dir = 'saved_models/eth'
        constant = return_the_eth_scene(args.scene)
        test_dset = TrajectoryDatasetETH(data_dir="datasets/eth/raw", obs_len=args.past_length,
                                         pred_len=args.future_length, skip=1, min_ped=1,
                                         delim='space', test_scene=args.scene, save_path=args.scene,
                                         mode=args.training_type)


    mission_test(test_dset, args,constant)
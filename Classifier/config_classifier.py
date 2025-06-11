import argparse
import datetime
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="NPCGAN")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--noise_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--out_dir", dest="out_dir", type=str, default="./out/")
    parser.add_argument("--model_dir", dest="model_dir", type=str, default="Classifier/saved_model/")
    parser.add_argument("--dataset", dest="dataset", type=str, default="nba")
    parser.add_argument("--G_dict", dest="G_dict", type=str, default="")
    parser.add_argument("--C_dict", dest="C_dict", type=str, default="")
    parser.add_argument("--D_dict", dest="D_dict", type=str, default="")
    parser.add_argument("--traj_len", dest="traj_len", type=int, default=8)
    parser.add_argument("--delim", dest="delim", default="\t")
    parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
    parser.add_argument("--l2_weight", type=float, default=1)
    parser.add_argument("--lr_step", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.9)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--model_names', default="100_nba_train_1_1_")
    parser.add_argument('--model_save_dir', default='G1/saved_models/nba')
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--traj_scale', type=int, default=1)
    parser.add_argument('--sample_k', type=int, default=20)
    parser.add_argument('--past_length', type=int, default=5)
    parser.add_argument('--future_length', type=int, default=10)
    parser.add_argument('--iternum_print', type=int, default=500)


    parser.add_argument('--agent', default=0)
    parser.add_argument('--target', default=[[80,80], [0,80], [0,0], [80,0], [40,40]])
    parser.add_argument('--method', default="mean")
    parser.add_argument('--length',  default=2500)
    parser.add_argument('--heat_map_path', default='Classifier/heatmap')
    parser.add_argument('--test_target', default=[[60, 60], [10, 50], [10, 10], [86, 0], [0, 40]])
    parser.add_argument("--agent_num", type=int, default=11)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--classifier_method", type=str, default='sampler')
    parser.add_argument('--training_type', type=str, default="train")
    parser.add_argument('--mission_buffer', type=float, default=2.0)
    parser.add_argument("--saved_models_GAN", default=None)
    parser.add_argument("--saved_models_GM", default="mission_aware30_None_nba_train_1_1_0.2")
    parser.add_argument("--saved_models_DIS", default=None)
    parser.add_argument("--saved_models_SAM", default="G__None_train_1_300_0_False_")
    parser.add_argument("--saved_models_SM", default="SM__None_test_False_10_0_0.2_")

    parser.add_argument('--scene', type=str, default="")
    parser.add_argument('--sdd_scene', type=int, default=None)
    parser.add_argument('--info', type=str, default="")
    parser.add_argument('--learn_prior', action='store_true', default=False)
    parser.add_argument('--test_mlp', action='store_true', default=False)



    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if "timestamp" not in args:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return args


if __name__ == "__main__":
    print(parse_args())

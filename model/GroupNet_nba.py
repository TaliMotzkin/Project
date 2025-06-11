from random import sample
from tkinter import TRUE
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
from model.utils_model import initialize_weights
from .MS_HGNN_batch import MS_HGNN_oridinary, MS_HGNN_hyper, MLP
import math


class DecomposeBlock(nn.Module):
    '''
    Balance between reconstruction task and prediction task.
    '''

    def __init__(self, past_len, future_len, input_dim):  # 10,5,288
        super(DecomposeBlock, self).__init__()
        # * HYPER PARAMETERS
        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        dim_embedding_key = 96
        self.past_len = past_len
        self.future_len = future_len

        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.encoder_past = nn.GRU(channel_out, dim_embedding_key, 1, batch_first=True)

        self.decoder_y = MLP(dim_embedding_key + input_dim, future_len * 2, hidden_size=(512, 256))
        self.decoder_x = MLP(dim_embedding_key + input_dim, past_len * 2, hidden_size=(512, 256))

        self.relu = nn.ReLU()

        # kaiming initialization
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)

    def forward(self, x_true, x_hat, f):
        '''
        >>> Input:
            x_true: N, T_p, 2
            x_hat: N, T_p, 2
            f: N, D

        >>> Output:
            x_hat_after: N, T_p, 2
            y_hat: n, T_f, 2
        '''
        x_ = x_true - x_hat
        # print("x_ in decompose ", x_.shape) #640 ,5, 2,
        x_ = torch.transpose(x_, 1, 2)
        # print("x_ T in decompose ", x_.shape)#640,2,5

        past_embed = self.relu(self.conv_past(x_))
        # print("past_embed in decompose ", past_embed.shape) #640, 32,5 - depends on chacel out

        past_embed = torch.transpose(past_embed, 1, 2)
        # print("past_embed T in decompose ", past_embed.shape)  # back to 640, 5, 32

        _, state_past = self.encoder_past(past_embed)
        state_past = state_past.squeeze(0)

        input_feat = torch.cat((f, state_past), dim=1)

        x_hat_after = self.decoder_x(input_feat).contiguous().view(-1, self.past_len, 2)
        y_hat = self.decoder_y(input_feat).contiguous().view(-1, self.future_len, 2)
        # print("y_hat in decompose ", y_hat.shape) #([640, 10, 2])
        # print("x_hat_after T in decompose ", x_hat_after.shape) #([640, 5, 2])
        return x_hat_after, y_hat


class Normal:
    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)  # split to two chunks the last dim #640, 32
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)
        self.fixed_eps = torch.randn_like(self.mu)

    def rsample(self):

        return self.mu + self.fixed_eps * self.sigma  # e performs the reparameterization trick, where the sample from the distribution is obtained by scaling the random nois

    def sample(self):
        return self.rsample()

    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            term1 = (self.mu - p.mu) / (p.sigma + 1e-8)
            term2 = self.sigma / (p.sigma + 1e-8)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return kl

    def mode(self):
        return self.mu


class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        initialize_weights(self.affine_layers.modules())

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


""" Positional Encoding """


class PositionalAgentEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_t_len=200, concat=True):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)  # 200X64
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # vector 200,1-0,1,2,3...
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))  # normalize
        pe[:, 0::2] = torch.sin(position * div_term)  # odd and even calcualtion of sin and cps
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # 200X64

    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]  # get by size of T
        pe = pe[None].repeat(num_a, 1, 1)  # for all agents in all batches B*N, T, model
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset):
        ae = self.ae[a_offset: num_a + a_offset, :]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, t_offset=0):
        num_t = x.shape[1]  # time steps #x-> BN, T, 4,| 64  #numa = B*N
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset)  # (N,T,D)
        if self.concat:
            feat = [x, pos_enc]
            x = torch.cat(feat, dim=-1)  # add pos encoder to the original x
            x = self.fc(x)
        else:
            x += pos_enc
        return self.dropout(x)  # (N,T,D)


class PastEncoder(nn.Module):
    def __init__(self, args, in_dim=4):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim  # 64
        self.scale_number = len(args.hyper_scales)

        self.input_fc = nn.Linear(in_dim, self.model_dim)
        self.input_fc2 = nn.Linear(self.model_dim * args.past_length, self.model_dim)
        self.input_fc3 = nn.Linear(self.model_dim + 3, self.model_dim)

        self.interaction = MS_HGNN_oridinary(
            dataset=self.args.dataset,
            device=self.args.device,
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=1
        )

        if len(args.hyper_scales) > 0:
            self.interaction_hyper = MS_HGNN_hyper(
                dataset=self.args.dataset,
                device = self.args.device,
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[0],  # 5
                actor_number=self.args.agent_num
            )
        if len(args.hyper_scales) > 1:
            self.interaction_hyper2 = MS_HGNN_hyper(
                dataset=self.args.dataset,
                device = self.args.device,
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[1], #11
                actor_number=self.args.agent_num
            )

        if len(args.hyper_scales) > 2:
            self.interaction_hyper3 = MS_HGNN_hyper(
                dataset=self.args.dataset,
                device=self.args.device,
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[2],
                actor_number=self.args.agent_num
            )

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, 0.1, concat=True)

    def add_category(self, x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N, 3).type_as(x)
        if self.args.dataset == 'nba':
            category[0:5, 0] = 1
            category[5:10, 1] = 1
            category[10, 2] = 1
        else:
            category[:, 0] = 1
            category[:, 1] = 1
            category[:, 2] = 1
        #todo add category of SDD?
        category = category.repeat(B, 1, 1)
        x = torch.cat((x, category), dim=-1)
        return x

    def forward(self, inputs, batch_size, agent_num):
        length = inputs.shape[1]
        # print("past encoder: ", inputs.shape)
        tf_in = self.input_fc(inputs).view(batch_size * agent_num, length, self.model_dim)  # BN, T,  64
        # print("past encoder tf_in: ", tf_in.shape)

        tf_in_pos = self.pos_encoder(tf_in, num_a=batch_size * agent_num)  # NB, T, D
        tf_in_pos = tf_in_pos.view(batch_size, agent_num, length, self.model_dim)
        # print("past encoder tf_in_pos: ", tf_in_pos.shape) # 32, 20, 5, 64

        ftraj_input = self.input_fc2(
            tf_in_pos.contiguous().view(batch_size, agent_num, length * self.model_dim))  # B, N, 64*T
        # print("past encoder ftraj_input1: ", ftraj_input.shape) #32., 20, 64
        ftraj_input = self.input_fc3(
            self.add_category(ftraj_input))  # adding meaning of ball and differnt players, now all set to 1

        # print("past encoder ftraj_input: ", ftraj_input.shape) #32., 20, 64 ->!!!!!!! B, N, Features

        query_input = F.normalize(ftraj_input, p=2, dim=2)  # use L2 norm, dim 2
        feat_corr = torch.matmul(query_input,
                                 query_input.permute(0, 2, 1))  # [B, N, N]

        # print("past encoder feat_corr: ", feat_corr.shape)

        ftraj_inter, _ = self.interaction(ftraj_input)  # ([32, 20, 64]

        if len(self.args.hyper_scales) > 0:
            ftraj_inter_hyper, _ , H1 = self.interaction_hyper(ftraj_input, feat_corr)  # ([32, 20, 64]
        if len(self.args.hyper_scales) > 1:
            ftraj_inter_hyper2, _, H2 = self.interaction_hyper2(ftraj_input, feat_corr)  # ([32, 20, 64]
            new_H = torch.cat((H1, H2),dim= 1 )
        if len(self.args.hyper_scales) > 2:
            ftraj_inter_hyper3, _ , H3 = self.interaction_hyper3(ftraj_input, feat_corr)
            new_H = torch.cat((new_H, H3),dim= 1 )

        if len(self.args.hyper_scales) == 0:
            final_feature = torch.cat((ftraj_input, ftraj_inter), dim=-1)  # ([32, 20, 64]
        if len(self.args.hyper_scales) == 1:
            final_feature = torch.cat((ftraj_input, ftraj_inter, ftraj_inter_hyper), dim=-1)  # ([32, 20, 64]
        elif len(self.args.hyper_scales) == 2:
            final_feature = torch.cat((ftraj_input, ftraj_inter, ftraj_inter_hyper, ftraj_inter_hyper2), dim=-1)
        elif len(self.args.hyper_scales) == 3:
            final_feature = torch.cat(
                (ftraj_input, ftraj_inter, ftraj_inter_hyper, ftraj_inter_hyper2, ftraj_inter_hyper3), dim=-1)

        output_feature = final_feature.view(batch_size * agent_num, -1)  # 32*20, 64*4
        # print("after concutinationg all in encoder past ", output_feature.shape)

        # print("H", H1.shape, H2.shape, H1[10], H2[110:130])
        return output_feature, new_H


class FutureEncoder(nn.Module):
    def __init__(self, args, in_dim=4):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim

        self.input_fc = nn.Linear(in_dim, self.model_dim)
        scale_num = 2 + len(self.args.hyper_scales)
        self.input_fc2 = nn.Linear(self.model_dim * self.args.future_length, self.model_dim)
        self.input_fc3 = nn.Linear(self.model_dim + 3, self.model_dim)

        self.interaction = MS_HGNN_oridinary(
            dataset=self.args.dataset,
            device=self.args.device,
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=1,
            vis=False
        )

        if len(args.hyper_scales) > 0:
            self.interaction_hyper = MS_HGNN_hyper(
                dataset=self.args.dataset,
                device=self.args.device,
                embedding_dim=16,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[0],
                vis=False,
                actor_number= self.args.agent_num
            )
        if len(args.hyper_scales) > 1:
            self.interaction_hyper2 = MS_HGNN_hyper(
                dataset=self.args.dataset,
                device=self.args.device,
                embedding_dim=16,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[1],
                vis=False,
                actor_number=self.args.agent_num
            )
        if len(args.hyper_scales) > 2:
            self.interaction_hyper3 = MS_HGNN_hyper(
                dataset=self.args.dataset,
                device=self.args.device,
                embedding_dim=16,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[2],
                vis=False,
                actor_number=self.args.agent_num
            )

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, 0.1, concat=True)

        self.out_mlp = MLP2(scale_num * 2 * self.model_dim, [128], 'relu')  # 4*2*64 = 512
        self.qz_layer = nn.Linear(self.out_mlp.out_dim, 2 * self.args.zdim)
        initialize_weights(self.qz_layer.modules())

    def add_category(self, x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N, 3).type_as(x)
        if self.args.dataset == 'nba':
            category[0:5, 0] = 1
            category[5:10, 1] = 1
            category[10, 2] = 1
        else:
            category[:, 0] = 1
            category[:, 1] = 1
            category[:, 2] = 1
        category = category.repeat(B, 1, 1)
        x = torch.cat((x, category), dim=-1)
        return x

    def forward(self, inputs, batch_size, agent_num, past_feature):
        length = inputs.shape[1]
        tf_in = self.input_fc(inputs).view(batch_size * agent_num, length, self.model_dim)

        tf_in_pos = self.pos_encoder(tf_in, num_a=batch_size * agent_num)
        tf_in_pos = tf_in_pos.view(batch_size, agent_num, length, self.model_dim)
        # print("future encoder tf_in_pos ", tf_in_pos.shape) #([32, 20, 10, 64])

        ftraj_input = self.input_fc2(tf_in_pos.contiguous().view(batch_size, agent_num, -1))
        # print("future encoder ftraj_input ", ftraj_input.shape)#([32, 20, 64])

        ftraj_input = self.input_fc3(self.add_category(ftraj_input))
        query_input = F.normalize(ftraj_input, p=2, dim=2)
        feat_corr = torch.matmul(query_input, query_input.permute(0, 2, 1))
        ftraj_inter, _ = self.interaction(ftraj_input)
        if len(self.args.hyper_scales) > 0:
            ftraj_inter_hyper, _ , H1= self.interaction_hyper(ftraj_input, feat_corr)
        if len(self.args.hyper_scales) > 1:
            ftraj_inter_hyper2, _ , H2= self.interaction_hyper2(ftraj_input, feat_corr)
            new_H = torch.cat((H1, H2), dim=1)
        if len(self.args.hyper_scales) > 2:
            ftraj_inter_hyper3, _ , H3= self.interaction_hyper3(ftraj_input, feat_corr)
            new_H = torch.cat((new_H, H3), dim=1)

        if len(self.args.hyper_scales) == 0:
            final_feature = torch.cat((ftraj_input, ftraj_inter), dim=-1)
        if len(self.args.hyper_scales) == 1:
            final_feature = torch.cat((ftraj_input, ftraj_inter, ftraj_inter_hyper), dim=-1)
        elif len(self.args.hyper_scales) == 2:
            final_feature = torch.cat((ftraj_input, ftraj_inter, ftraj_inter_hyper, ftraj_inter_hyper2), dim=-1)
        elif len(self.args.hyper_scales) == 3:
            final_feature = torch.cat(
                (ftraj_input, ftraj_inter, ftraj_inter_hyper, ftraj_inter_hyper2, ftraj_inter_hyper3), dim=-1)

        final_feature = final_feature.view(batch_size * agent_num, -1)
        # print("final_feature future encoder ", final_feature.shape) #torch.Size([640, 256]) like in the past

        h = torch.cat((past_feature, final_feature), dim=-1)

        # print("h in future encoder ", h.shape)#640, 256*2
        h = self.out_mlp(h)
        # print("h in future encoder after mlp2 ", h.shape)#640, 128

        q_z_params = self.qz_layer(h)

        # print("q_z_params in future encoder after some linear ", q_z_params.shape)#640, 64
        return q_z_params, new_H


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.decode_way = 'RES'
        scale_num = 2 + len(self.args.hyper_scales)

        self.num_decompose = args.num_decompose
        input_dim = scale_num * self.model_dim + self.args.zdim  # =4*64 + 32 =288 -> 256 of past and 32 of future
        self.past_length = self.args.past_length
        self.future_length = self.args.future_length

        self.decompose = nn.ModuleList(
            [DecomposeBlock(self.args.past_length, self.args.future_length, input_dim) for _ in
             range(self.num_decompose)])

    # sending past features:32*20, 64*4,
    # qz_sampled mu and sigam shapes: 640, 32, batch 32,  agents 20, past traj: 640,5,2, current location, 640,1,2
    def forward(self, past_feature, z, batch_size_curr, agent_num_perscene, past_traj, cur_location, sample_num,
                mode='train'):
        agent_num = batch_size_curr * agent_num_perscene  # 640
        past_traj_repeat = past_traj.repeat_interleave(sample_num,
                                                       dim=0)  # After repeat_interleave(sample_num=4, dim=0): Shape becomes (640 * 4, 5, 2) = (2560, 5, 2)

        past_feature = past_feature.view(-1, sample_num, past_feature.shape[
            -1])  # If past_feature is (2560, 64) (from 640 agents × 4 samples):After reshaping: (640, 4, 64).

        # print("past_traj_repeat decoder ", past_traj_repeat.shape) #640, 5, 2
        # print("past_feature decoder ", past_feature.shape) #640,1, 256,

        z_in = z.view(-1, sample_num, z.shape[-1])

        # print("z_in decoder ", z_in.shape) #640,20,32

        hidden = torch.cat((past_feature, z_in), dim=-1)
        # print("hidden decoder", hidden.shape) #640, 1, 256+32=288
        hidden = hidden.view(agent_num * sample_num, -1)
        # print("hidden decoder after view, ", hidden.shape) #640, 288
        x_true = past_traj_repeat.clone()  # torch.transpose(pre_motion_scene_norm, 0, 1)

        x_hat = torch.zeros_like(x_true)
        batch_size = x_true.size(0)

        # print("x_true", x_true.shape,  " batch_size ", batch_size) #640,5,2  batch =640..
        prediction = torch.zeros((batch_size, self.future_length, 2)).to(self.args.device)
        reconstruction = torch.zeros((batch_size, self.past_length, 2)).to(self.args.device)
        # prediction = torch.zeros((batch_size, self.future_length, 2)).cuda()
        # reconstruction = torch.zeros((batch_size, self.past_length, 2)).cuda()

        for i in range(self.num_decompose):  # numdecompose = 2
            x_hat, y_hat = self.decompose[i](x_true, x_hat, hidden)  # recurrent - here it uses GRUs
            prediction += y_hat
            reconstruction += x_hat
        norm_seq = prediction.view(agent_num * sample_num, self.future_length, 2)
        recover_pre_seq = reconstruction.view(agent_num * sample_num, self.past_length, 2)

        # print("norm_seq decoder ", norm_seq.shape) #if samples are =20 -> 640*20 = 12800
        # print("recover_pre_seq decoder ", recover_pre_seq.shape)
        # norm_seq = norm_seq.permute(2,0,1,3).view(self.future_length, agent_num * sample_num,2)

        cur_location_repeat = cur_location.repeat_interleave(sample_num, dim=0)
        out_seq = norm_seq + cur_location_repeat
        if mode == 'inference':
            out_seq = out_seq.view(-1, sample_num, *out_seq.shape[1:])  # (agent_num*B,sample_num,self.past_length,2)
        return out_seq, recover_pre_seq



class GroupNet(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.device = device
        self.args = args
        # print("args", self.args)

        # models
        scale_num = 2 + len(self.args.hyper_scales)  # [5,11]=4
        self.past_encoder = PastEncoder(args)
        self.pz_layer = nn.Linear(scale_num * self.args.hidden_dim, 2 * self.args.zdim)  # 4*64, 2*32
        if args.learn_prior:
            initialize_weights(self.pz_layer.modules())
        self.future_encoder = FutureEncoder(args)
        self.decoder = Decoder(args)
        self.param_annealers = nn.ModuleList()
        self.criterion = nn.CrossEntropyLoss()

    def set_device(self, device):
        self.device = device
        self.to(device)

    def calculate_loss_pred(self, pred, target, batch_size):
        loss = (target - pred).pow(2).sum()
        loss /= batch_size
        loss /= pred.shape[1]
        return loss

    def calculate_loss_kl(self, qz_distribution, pz_distribution, batch_size, agent_num, min_clip):

        loss = qz_distribution.kl(pz_distribution).sum()
        loss /= (batch_size * agent_num)
        loss_clamp = loss.clamp_min_(min_clip)
        return loss_clamp

    def calculate_loss_recover(self, pred, target, batch_size):
        loss = (target - pred).pow(2).sum()
        loss /= batch_size
        loss /= pred.shape[1]
        return loss

    def calculate_loss_diverse(self, pred, target, batch_size):
        diff = target.unsqueeze(1) - pred  # future - 20 samples
        avg_dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
        loss = avg_dist.min(dim=1)[0]
        loss = loss.mean()
        return loss

    def calculate_softmax_loss(self, pred, target, model_output, kl_diverse, cross_entropy, noisy, epsilon=1e-1):
        # pred shape: [B*N, S, T, 2]
        # target shape: [B*N, T, 2]
        target_expanded = target.unsqueeze(1).expand_as(pred)  # [B*N, S, T, 2]
        # print(target_expanded)
        diff = pred - target_expanded  # Difference
        # print("diff", diff)
        dist_squared = diff.pow(2).sum(dim=-1).sum(dim=-1)  # Sum squared differences across T and 2 dimensions
        # print("dist_squared", dist_squared)
        # scores = -dist_squared
        # probabilities = F.softmax(scores, dim=1)

        probabilities = model_output
        _, closest_idx = dist_squared.min(dim=1)
        true_indices = torch.zeros_like(probabilities).scatter_(1, closest_idx.unsqueeze(1), 1)
        loss = self.criterion(probabilities, true_indices)

        if kl_diverse:
            soft_targets = F.softmax(-dist_squared, dim=1)
            epsilon = 1e-9
            predicted_probs = torch.clamp(probabilities, min=epsilon)
            soft_targets = torch.clamp(soft_targets, min=epsilon)

            # print("soft_targets", soft_targets[0:1, :])
            # print("predicted_probs", predicted_probs[0:1, :])

            kl_div = soft_targets * torch.log(soft_targets / predicted_probs)
            loss = torch.sum(kl_div, dim=1).mean()
        if cross_entropy:
            # print("probabilities", probabilities)
            _, closest_idx = dist_squared.min(dim=1)
            # print(closest_idx, "closest_idx")
            # true_indices = torch.zeros_like(probabilities).scatter_(1, closest_idx.unsqueeze(1), 1)
            # print("true_indices", true_indices)
            nll_loss = nn.NLLLoss()
            epsilon = 1e-9
            predicted_probs = torch.clamp(probabilities, min=epsilon)
            log_probabilities = predicted_probs.log()
            loss = nll_loss(log_probabilities, closest_idx)
            # print("loss", loss)
        if noisy:
            _, closest_idx = dist_squared.min(dim=1)
            # print(closest_idx.shape, "closest_idx")
            true_indices = torch.zeros_like(probabilities).scatter_(1, closest_idx.unsqueeze(1), 1)
            # print("true_indices", true_indices.shape)
            smoothed_targets = (1 - epsilon) * true_indices + epsilon / target_expanded.shape[1]
            # print("smoothed_targets", smoothed_targets.shape)
            log_probabilities = probabilities.log()
            # print("log_probabilities", log_probabilities.shape)
            loss = -torch.sum(smoothed_targets * log_probabilities, dim=1).mean()

        return loss

    def forward(self, data):
        device = self.device
        batch_size = data['past_traj'].shape[0]
        agent_num = data['past_traj'].shape[1]

        past_traj = data['past_traj'].view(batch_size * agent_num, self.args.past_length, 2).to(device).contiguous()
        future_traj = data['future_traj'].view(batch_size * agent_num, self.args.future_length, 2).to(
            device).contiguous()

        # print("past_traj ", past_traj.shape) #640,5,2
        past_vel = past_traj[:, 1:] - past_traj[:, :-1, :]  # calcuates velocity and then padding to match the seq length
        past_vel = torch.cat([past_vel[:, [0]], past_vel], dim=1)

        future_vel = future_traj - torch.cat([past_traj[:, [-1]], future_traj[:, :-1, :]], dim=1)
        cur_location = past_traj[:, [-1]]  # last agents locations

        # print("cur_location ", cur_location.shape) #640,1,2
        inputs = torch.cat((past_traj, past_vel), dim=-1)  # [x,y, xv,yv]
        # print("velocity and xy size", inputs.shape) #32*20, T=5, VXY =4
        inputs_for_posterior = torch.cat((future_traj, future_vel), dim=-1)  # 32*20, T=10, VXY =4

        # print("inputs_for_posterior, ", inputs_for_posterior.shape)
        past_feature, _ = self.past_encoder(inputs, batch_size, agent_num)  # 32*20, 64*4
        qz_param, _ = self.future_encoder(inputs_for_posterior, batch_size, agent_num, past_feature)  # 32*20, 64

        ### q dist ### of future and past
        if self.args.ztype == 'gaussian':
            qz_distribution = Normal(params=qz_param)
        else:
            ValueError('Unknown hidden distribution!')
        qz_sampled = qz_distribution.rsample()
        # print("qz_sampled mu and sigma ,", qz_sampled.shape)
        ### p dist ### only of past
        if self.args.learn_prior:
            pz_param = self.pz_layer(past_feature)  # send to linear the past embeding
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=pz_param)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(mu=torch.zeros(past_feature.shape[0], self.args.zdim).to(past_traj.device),
                                         logvar=torch.zeros(past_feature.shape[0], self.args.zdim).to(past_traj.device))
            else:
                ValueError('Unknown hidden distribution!')

        ### use q ###
        # z = qz_sampled
        pred_traj, recover_traj = self.decoder(past_feature, qz_sampled, batch_size, agent_num, past_traj, cur_location,
                                               sample_num=1)  # sending past features:32*20, 64*4,
        # qz_sampled mu and sigam shapes: 640, 32, batch 32,  agents 20, past traj: 640,5,2, current location, 640,1,2

        # print("pred_traj main ", pred_traj.shape) #640,10,2

        loss_pred = self.calculate_loss_pred(pred_traj, future_traj, batch_size)  # future loss

        loss_recover = self.calculate_loss_recover(recover_traj, past_traj, batch_size)  # past loss
        loss_kl = self.calculate_loss_kl(qz_distribution, pz_distribution, batch_size, agent_num,
                                         self.args.min_clip)  # calcualte KL divergance

        ### p dist for best 20 loss ###
        sample_num = 20
        if self.args.learn_prior:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            p_z_params = self.pz_layer(past_feature_repeat)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=p_z_params)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(
                    mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device),
                    logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device))
            else:
                ValueError('Unknown hidden distribution!')

        pz_sampled = pz_distribution.rsample()
        # z = pz_sampled

        diverse_pred_traj, _ = self.decoder(past_feature_repeat, pz_sampled, batch_size, agent_num, past_traj,
                                            cur_location, sample_num=20, mode='inference')

        loss_diverse = self.calculate_loss_diverse(diverse_pred_traj, future_traj, batch_size)  # future - 32*20, T, 2

        total_loss = loss_pred + loss_recover + loss_kl + loss_diverse

        return total_loss, loss_pred.item(), loss_recover.item(), loss_kl.item(), loss_diverse.item(), diverse_pred_traj

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def inference(self, data):
        device = self.device
        batch_size = data['past_traj'].shape[0]
        agent_num = data['past_traj'].shape[1]

        past_traj = data['past_traj'].view(batch_size * agent_num, self.args.past_length, 2).to(device).contiguous()

        past_vel = past_traj[:, 1:] - past_traj[:, :-1, :]
        past_vel = torch.cat([past_vel[:, [0]], past_vel], dim=1)

        cur_location = past_traj[:, [-1]]

        inputs = torch.cat((past_traj, past_vel), dim=-1)

        past_feature, H = self.past_encoder(inputs, batch_size, agent_num)

        sample_num = self.args.sample_k
        if self.args.learn_prior:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            p_z_params = self.pz_layer(past_feature_repeat)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=p_z_params)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(
                    mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device),
                    logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device))
            else:
                ValueError('Unknown hidden distribution!')

        pz_sampled = pz_distribution.rsample()
        z = pz_sampled

        diverse_pred_traj, _ = self.decoder(past_feature_repeat, z, batch_size, agent_num, past_traj, cur_location,
                                            sample_num=self.args.sample_k, mode='inference')  # Z in the decodng

        # outputs_softmax = self.final_model(diverse_pred_traj)
        diverse_pred_traj = diverse_pred_traj.permute(1, 0, 2, 3) #S, BN, T, 2
        # return diverse_pred_traj, outputs_softmax, H

        return diverse_pred_traj, H

    def inference_simulator(self, data):
        device = self.device
        batch_size = data.shape[0]
        agent_num = data.shape[1]

        past_traj = data.reshape(batch_size * agent_num, data.shape[2], 2).to(device).contiguous()

        past_vel = past_traj[:, 1:] - past_traj[:, :-1, :]
        past_vel = torch.cat([past_vel[:, [0]], past_vel], dim=1)

        cur_location = past_traj[:, [-1]]

        inputs = torch.cat((past_traj, past_vel), dim=-1)

        past_feature, H = self.past_encoder(inputs, batch_size, agent_num)

        sample_num = self.args.sample_k
        if self.args.learn_prior:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            p_z_params = self.pz_layer(past_feature_repeat)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=p_z_params)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(
                    mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device),
                    logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device))
            else:
                ValueError('Unknown hidden distribution!')

        pz_sampled = pz_distribution.rsample()
        z = pz_sampled

        diverse_pred_traj, _ = self.decoder(past_feature_repeat, z, batch_size, agent_num, past_traj, cur_location,
                                            sample_num=self.args.sample_k, mode='inference')  # Z in the decodng
        diverse_pred_traj = diverse_pred_traj.permute(1, 0, 2, 3)
        return diverse_pred_traj, H

    def inference_future(self, past, future):

        device = self.device
        batch_size = past.shape[0]
        agent_num = past.shape[1]

        past_traj = past.reshape(batch_size * agent_num, past.shape[2], 2).to(device).contiguous()
        future_traj = future.reshape(batch_size * agent_num, future.shape[2], 2).to(device).contiguous()

        # print("past_traj ", past_traj.shape) #640,5,2
        past_vel = past_traj[:, 1:] - past_traj[:, :-1,:]  # calcuates velocity and then padding to match the seq length
        past_vel = torch.cat([past_vel[:, [0]], past_vel], dim=1)

        future_vel = future_traj - torch.cat([past_traj[:, [-1]], future_traj[:, :-1, :]], dim=1)


        inputs = torch.cat((past_traj, past_vel), dim=-1)  # [x,y, xv,yv]
        # print("velocity and xy size", inputs.shape) #32*20, T=5, VXY =4
        inputs_for_posterior = torch.cat((future_traj, future_vel), dim=-1)  # 32*20, T=10, VXY =4

        # print("inputs_for_posterior, ", inputs_for_posterior.shape)
        past_feature, _ = self.past_encoder(inputs, batch_size, agent_num)  # 32*20, 64*4
        _ , H = self.future_encoder(inputs_for_posterior, batch_size, agent_num, past_feature)

        return  H

    def inference_simulator2(self, data, sample_size):
        device = self.device
        batch_size = data.shape[0]
        agent_num = data.shape[1]

        past_traj = data.reshape(batch_size * agent_num, data.shape[2], 2).to(device).contiguous()

        past_vel = past_traj[:, 1:] - past_traj[:, :-1, :]
        past_vel = torch.cat([past_vel[:, [0]], past_vel], dim=1)

        cur_location = past_traj[:, [-1]]

        inputs = torch.cat((past_traj, past_vel), dim=-1)

        past_feature, H = self.past_encoder(inputs, batch_size, agent_num)

        sample_num = sample_size
        if self.args.learn_prior:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            p_z_params = self.pz_layer(past_feature_repeat)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=p_z_params)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(
                    mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device),
                    logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device))
            else:
                ValueError('Unknown hidden distribution!')

        pz_sampled = pz_distribution.rsample()
        z = pz_sampled

        diverse_pred_traj, _ = self.decoder(past_feature_repeat, z, batch_size, agent_num, past_traj, cur_location,
                                            sample_num=sample_size, mode='inference')  # Z in the decodng
        diverse_pred_traj = diverse_pred_traj.permute(1, 0, 2, 3)
        return diverse_pred_traj, H







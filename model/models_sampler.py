
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
from torch.nn import functional as F

class TrajectoryClassifier(nn.Module):
    def __init__(self, device, input_dim=2, hidden_dim=128, num_layers=2):
        """
        Classifies whether a given trajectory sequence belongs to a controlled (real) or random (fake) movement.

        input_dim: 2 (xy coordinates)
        hidden_dim: LSTM hidden size
        num_layers: number of LSTM layers
        """
        super(TrajectoryClassifier, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (Batch, N, Seq, 2) - Trajectories for all agents
        """
        batch_size, num_agents, seq_length, _ = x.shape
        x = x.view(batch_size * num_agents, seq_length, -1)
        # with torch.backends.cudnn.flags(enabled=False):
        _, (hidden, _) = self.lstm(x)  #
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  #
        hidden = hidden.view(batch_size, num_agents, -1)
        x = hidden
        x = self.fc(x)
        x = x.squeeze(-1)
        return x


class SamplerMLP(nn.Module):
    def __init__(self ,args, device, in_channels_speed_past, in_channels_speed_future, out_channels_speed, hidden_size_speed,
                 in_channels_dir_past, in_channels_dir_future ,out_channels_dir ,hidden_size_dir ,heads ,depth, SWM_out
                 ,edge_dim, bias):
        super(SamplerMLP, self).__init__()
        self.args = args
        self.mlp_past = nn.Sequential(
            nn.Linear(2*self.args.past_length, hidden_size_speed, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size_speed, self.args.future_length*2),
        )
        self.groupnet = nn.Sequential(
            nn.Linear(self.args.future_length*4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def forward(self ,  past, visability_mat_past, speed_past, directions_past, edge_features_past
                ,edge_weights_past ,prediction):

        B, N, T_past, C =past.shape

        eq_in_past = self.mlp_past(past.view(B, N, T_past * C))  # b, n, 5, 2 --> ,b, n, 20

        prediction = prediction.permute(0, 1, 4, 2, 3).reshape(B, N, 20, -1)  # B N 20 20
        # B, N, 20, 10, 2
        x_query = eq_in_past.unsqueeze(2).expand(-1, -1, 20, -1)  # B N 20 20
        query_option_pair = torch.cat([x_query, prediction], dim=-1)  # concatenate query and each option
        scores = self.groupnet(query_option_pair).squeeze(-1)  # B, N, 20


        return eq_in_past.view(B, N, self.args.future_length, 2), scores,

    def inference(self, past, visability_mat_past, speed_past, directions_past, edge_features_past, edge_weights_past,
                  prediction):
        B, N, T_past, C = past.shape

        eq_in_past = self.mlp_past(past.reshape(B, N, T_past * C))  # ,b, n, 20

        prediction = prediction.permute(0, 1, 4, 2, 3).reshape(B, N, 20, -1)  # B N 20 20
        # B, N, 20, 10, 2
        x_query = eq_in_past.unsqueeze(2).expand(-1, -1, 20, -1)  # B N 20 20
        query_option_pair = torch.cat([x_query, prediction], dim=-1)  # concatenate query and each option
        scores = self.groupnet(query_option_pair).squeeze(-1)  # B, N, 20
        pred_index = scores.argmax(dim=-1)  # the indexes

        return eq_in_past.view(B, N,  self.args.future_length, 2), pred_index


class SamplerMLPALL(nn.Module):
    def __init__(self, args, device, in_channels_speed_past, in_channels_speed_future, out_channels_speed,
                 hidden_size_speed,
                 in_channels_dir_past, in_channels_dir_future, out_channels_dir, hidden_size_dir, heads, depth, SWM_out
                 , edge_dim, bias):
        super(SamplerMLPALL, self).__init__()
        self.args = args
        self.mlp_past = nn.Sequential(
            nn.Linear(args.past_length*2, hidden_size_speed, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size_speed, args.future_length*2),
        )

        self.mlp_speed = nn.Sequential(
            nn.Linear(args.past_length -1 , hidden_size_speed, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size_speed, args.future_length*2),
        )

        self.mlp_dir = nn.Sequential(
            nn.Linear(args.past_length-1, hidden_size_speed, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size_speed, args.future_length*2),
        )

        self.groupnet = nn.Sequential(
            nn.Linear(self.args.future_length*4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


        self.final = nn.Sequential(
            nn.Linear(6*args.future_length, 64),
            nn.ReLU(),
            nn.Linear(64, 2*args.future_length)
        )

    def forward(self, past, visability_mat_past, speed_past, directions_past, edge_features_past
                , edge_weights_past, prediction):
        B, N, T_past, C = past.shape

        eq_in_past = self.mlp_past(past.view(B, N, T_past * C))  # b, n, 5, 2 --> ,b, n, 20
        eq_in_speed = self.mlp_speed(speed_past.view(B, N, (T_past-1)))  # b, n, 5, 2 --> ,b, n, 20
        eq_in_dir = self.mlp_dir(directions_past.view(B, N, (T_past-1)))  # b, n, 5, 2 --> ,b, n, 20
        eq_in = torch.cat((eq_in_past, eq_in_speed, eq_in_dir), dim=-1)
        eq_in = self.final(eq_in)
        prediction = prediction.permute(0, 1, 4, 2, 3).reshape(B, N, 20, -1)  # B N 20 20
        # B, N, 20, 10, 2
        x_query = eq_in.unsqueeze(2).expand(-1, -1, 20, -1)  # B N 20 20
        query_option_pair = torch.cat([x_query, prediction], dim=-1)  # concatenate query and each option
        scores = self.groupnet(query_option_pair).squeeze(-1)  # B, N, 20

        return eq_in.view(B, N, self.args.future_length, 2), scores

    def inference(self, past, visability_mat_past, speed_past, directions_past, edge_features_past, edge_weights_past,
                  prediction):
        B, N, T_past, C = past.shape

        eq_in_past = self.mlp_past(past.reshape(B, N, T_past * C))  # b, n, 5, 2 --> ,b, n, 20
        eq_in_speed = self.mlp_speed(speed_past.reshape(B, N, (T_past - 1) ))  # b, n, 5, 2 --> ,b, n, 20
        eq_in_dir = self.mlp_dir(directions_past.reshape(B, N, (T_past - 1) ))  # b, n, 5, 2 --> ,b, n, 20
        eq_in = torch.cat((eq_in_past, eq_in_speed, eq_in_dir), dim=-1)
        eq_in = self.final(eq_in)
        prediction = prediction.permute(0, 1, 4, 2, 3).reshape(B, N, 20, -1)  # B N 20 20
        # B, N, 20, 10, 2
        x_query = eq_in.unsqueeze(2).expand(-1, -1, 20, -1)  # B N 20 20
        query_option_pair = torch.cat([x_query, prediction], dim=-1)  # concatenate query and each option
        scores = self.groupnet(query_option_pair).squeeze(-1)  # B, N, 20

        pred_index = scores.argmax(dim=-1)  # the indexes

        return eq_in_past.view(B, N, self.args.future_length, 2), pred_index


class SamplerMLPATT(nn.Module):
    def __init__(self, args, device, in_channels_speed_past, in_channels_speed_future, out_channels_speed,
                 hidden_size_speed,
                 in_channels_dir_past, in_channels_dir_future, out_channels_dir, hidden_size_dir, heads, depth, SWM_out
                 , edge_dim, bias):
        super(SamplerMLPATT, self).__init__()
        self.args = args
        self.mlp_past = nn.Sequential(
            nn.Linear(args.past_length*2, hidden_size_speed, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size_speed, args.future_length*2),
        )

        self.mlp_speed = nn.Sequential(
            nn.Linear(args.past_length -1 , hidden_size_speed, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size_speed, args.future_length*2),
        )

        self.mlp_dir = nn.Sequential(
            nn.Linear(args.past_length-1, hidden_size_speed, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size_speed, args.future_length*2),
        )

        self.final = nn.Sequential(
            nn.Linear(6 * args.future_length, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * args.future_length)
        )

        self.pred_proj = nn.Linear(64, 1)
        self.pred_lstm = nn.LSTM(input_size=2,
                                 hidden_size=64,
                                 num_layers=1,
                                 batch_first=True)

    def forward(self, past, visability_mat_past, speed_past, directions_past, edge_features_past
                , edge_weights_past, prediction):
        B, N, T_past, C = past.shape

        eq_in_past = self.mlp_past(past.view(B, N, T_past * C))  # b, n, 5, 2 --> ,b, n, 20
        eq_in_speed = self.mlp_speed(speed_past.view(B, N, (T_past - 1) ))  # b, n, 5, 2 --> ,b, n, 20
        eq_in_dir = self.mlp_dir(directions_past.view(B, N, (T_past - 1)))  # b, n, 5, 2 --> ,b, n, 20
        eq_in = torch.cat((eq_in_past, eq_in_speed, eq_in_dir), dim=-1)
        eq_in = self.final(eq_in)

        pred_seq = prediction.permute(0, 1, 4, 2, 3).reshape(B*N*20, self.args.future_length, 2)  # B N 20 20
        lstm_out, (h_n, c_n) = self.pred_lstm(pred_seq)
        h_n = h_n.squeeze(0)                            # (B·N·20, lstm_hidden)
        pred_feat = self.pred_proj(h_n).squeeze(-1)        # B, N, 20
        pred_feat = pred_feat.view(B, N, 20)

        combined = eq_in + pred_feat

        return eq_in.view(B, N, self.args.future_length, 2), combined.view(B, N, -1)

    def inference(self, past, visability_mat_past, speed_past, directions_past, edge_features_past, edge_weights_past,
                  prediction):
        B, N, T_past, C = past.shape

        eq_in_past = self.mlp_past(past.reshape(B, N, T_past * C))  # b, n, 5, 2 --> ,b, n, 20
        eq_in_speed = self.mlp_speed(speed_past.reshape(B, N, (T_past - 1) ))  # b, n, 5, 2 --> ,b, n, 20
        eq_in_dir = self.mlp_dir(directions_past.reshape(B, N, (T_past - 1) ))  # b, n, 5, 2 --> ,b, n, 20
        eq_in = torch.cat((eq_in_past, eq_in_speed, eq_in_dir), dim=-1)
        eq_in = self.final(eq_in)


        pred_seq = prediction.permute(0, 1, 4, 2, 3).reshape(B*N*20, self.args.future_length, 2)  # B N 20 20
        lstm_out, (h_n, c_n) = self.pred_lstm(pred_seq)
        h_n = h_n.squeeze(0)                            # (B·N·20, lstm_hidden)
        pred_feat = self.pred_proj(h_n).squeeze(-1)  # B, N, 20
        pred_feat = pred_feat.view(B, N, 20)
        combined = eq_in + pred_feat

        pred_index = combined.argmax(dim=-1)

        return eq_in.view(B, N, self.args.future_length, 2), pred_index


class Sampler(nn.Module):
    def __init__(self, args, device, in_channels_speed_past, in_channels_speed_future, out_channels_speed,
                 hidden_size_speed,
                 in_channels_dir_past, in_channels_dir_future, out_channels_dir, hidden_size_dir, heads, depth, SWM_out,
                 edge_dim, bias):
        super(Sampler, self).__init__()
        self.MSG_past = MessagePassing("past", args, args.past_length, device, in_channels_speed_past, out_channels_speed,
                                       hidden_size_speed
                                       , in_channels_dir_past, out_channels_dir, hidden_size_dir, heads, depth, SWM_out,
                                       edge_dim, bias).to(device)

        self.args = args
        self.groupnet = nn.Sequential(
            nn.Linear(self.args.future_length*4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )



    def forward(self,  past, visability_mat_past, speed_past, directions_past, edge_features_past,
                edge_weights_past,prediction):
        B, N, T_past, _ = past.shape

        # print("past", past.shape)
        # print("visability_mat_past", visability_mat_past.shape)
        eq_in_past = self.MSG_past(past, visability_mat_past, speed_past, directions_past, edge_features_past,
                                   edge_weights_past)  # ,b, n, 20
        cur_location = past[:,:, [-1]] # last agents locations
        eq_in_past = eq_in_past.view(B, N, self.args.future_length, 2) + cur_location

        prediction = prediction.permute(0, 1, 4, 2, 3).reshape(B, N, 20, -1)  # B N 20 20
        # B, N, 20, 10, 2
        x_query = eq_in_past.reshape(B, N, self.args.future_length*2).unsqueeze(2).expand(-1, -1, 20, -1)  # B N 20 20
        query_option_pair = torch.cat([x_query, prediction], dim=-1)  # concatenate query and each option
        # print("query_option_pair", query_option_pair.shape)
        scores = self.groupnet(query_option_pair).squeeze(-1)  # B, N, 20



        return eq_in_past.view(B, N, self.args.future_length, 2), scores

        # return eq_in_past.view(B, N, 10, 2), scores, eq_in_future, eq_in_best

    def inference(self, past, visability_mat_past, speed_past, directions_past, edge_features_past, edge_weights_past,
                  prediction):
        B, N, T_past, _ = past.shape
        eq_in_past = self.MSG_past(past, visability_mat_past, speed_past, directions_past, edge_features_past,
                                   edge_weights_past)  # ,b, n, 20
        cur_location = past[:, :, [-1]]  # last agents locations
        eq_in_past = eq_in_past.view(B, N, self.args.future_length, 2) + cur_location

        prediction = prediction.permute(0, 1, 4, 2, 3).reshape(B, N, 20, -1)  # B N 20 20
        # B, N, 20, 10, 2
        x_query = eq_in_past.reshape(B, N, self.args.future_length*2).unsqueeze(2).expand(-1, -1, 20, -1)  # B N 20 20
        query_option_pair = torch.cat([x_query, prediction], dim=-1)  # concatenate query and each option
        scores = self.groupnet(query_option_pair).squeeze(-1)  # B, N, 20
        pred_index = scores.argmax(dim=-1)  # the indexes

        return eq_in_past.view(B, N, self.args.future_length, 2), pred_index

class SamplerGraph(nn.Module):
    def __init__(self, args, device, in_channels_speed_past, in_channels_speed_future, out_channels_speed,
                 hidden_size_speed,
                 in_channels_dir_past, in_channels_dir_future, out_channels_dir, hidden_size_dir, heads, depth, SWM_out,
                 edge_dim, bias):
        super(SamplerGraph, self).__init__()
        self.MSG_past = MessagePassingGraph("past", args, args.past_length, device, in_channels_speed_past, out_channels_speed,
                                       hidden_size_speed
                                       , in_channels_dir_past, out_channels_dir, hidden_size_dir, heads, depth, SWM_out,
                                       edge_dim, bias).to(device)

        self.args = args
        self.groupnet = nn.Sequential(
            nn.Linear(self.args.future_length*4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )



    def forward(self,  past, visability_mat_past, speed_past, directions_past, edge_features_past,
                edge_weights_past,prediction):
        B, N, T_past, _ = past.shape

        # print("past", past.shape)
        # print("visability_mat_past", visability_mat_past.shape)
        eq_in_past = self.MSG_past(past, visability_mat_past, speed_past, directions_past)  # ,b, n, 20

        cur_location = past[:,:, [-1]] # last agents locations
        eq_in_past = eq_in_past.view(B, N, self.args.future_length, 2) + cur_location

        prediction = prediction.permute(0, 1, 4, 2, 3).reshape(B, N, 20, -1)  # B N 20 20
        # B, N, 20, 10, 2
        x_query = eq_in_past.reshape(B, N, self.args.future_length*2).unsqueeze(2).expand(-1, -1, 20, -1)  # B N 20 20
        query_option_pair = torch.cat([x_query, prediction], dim=-1)  # concatenate query and each option
        # print("query_option_pair", query_option_pair.shape)
        scores = self.groupnet(query_option_pair).squeeze(-1)  # B, N, 20



        return eq_in_past.view(B, N, self.args.future_length, 2), scores

        # return eq_in_past.view(B, N, 10, 2), scores, eq_in_future, eq_in_best

    def inference(self, past, visability_mat_past, speed_past, directions_past, edge_features_past, edge_weights_past,
                  prediction):
        B, N, T_past, _ = past.shape
        eq_in_past = self.MSG_past(past, visability_mat_past, speed_past, directions_past)  # ,b, n, 20
        cur_location = past[:, :, [-1]]  # last agents locations
        eq_in_past = eq_in_past.view(B, N, self.args.future_length, 2) + cur_location

        prediction = prediction.permute(0, 1, 4, 2, 3).reshape(B, N, 20, -1)  # B N 20 20
        # B, N, 20, 10, 2
        x_query = eq_in_past.reshape(B, N, self.args.future_length*2).unsqueeze(2).expand(-1, -1, 20, -1)  # B N 20 20
        query_option_pair = torch.cat([x_query, prediction], dim=-1)  # concatenate query and each option
        scores = self.groupnet(query_option_pair).squeeze(-1)  # B, N, 20
        pred_index = scores.argmax(dim=-1)  # the indexes

        return eq_in_past.view(B, N, self.args.future_length, 2), pred_index




class SamplerMission(nn.Module):
    def __init__(self,args, device, in_channels_speed_past, in_channels_speed_future, out_channels_speed, hidden_size_speed,
                 in_channels_dir_past, in_channels_dir_future,out_channels_dir,hidden_size_dir,heads,depth, SWM_out,edge_dim, bias):
        super(SamplerMission, self).__init__()
        self.MSG_past = MessagePassing("past", args, args.past_length, device, in_channels_speed_past,
                                       out_channels_speed,
                                       hidden_size_speed
                                       , in_channels_dir_past, out_channels_dir, hidden_size_dir, heads, depth, SWM_out,
                                       edge_dim, bias).to(device)

        self.args = args
        self.groupnet = nn.Sequential(
            nn.Linear(self.args.future_length*4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.mission_encoder = nn.Sequential(nn.Linear(3, 8),nn.ReLU())
        self.past_encoder = nn.Sequential(nn.Linear(args.future_length*2 + 9 , 48), nn.ReLU(), nn.Linear(48, args.future_length*2),)

        # self.mission_decoder = MissionAwareDecoder(args.future_length*2, 8, 32, args.future_length)

    def forward(self,alpha, agents_idx,one_mission,error_tolerance, past, visability_mat_past, speed_past,
                directions_past, edge_features_past,edge_weights_past,prediction):

        B, N, T_past, _ =past.shape

        eq_in_past = self.MSG_past( past, visability_mat_past, speed_past, directions_past, edge_features_past,edge_weights_past) #,b, n, 20
        cur_location = past[:, :, [-1]]  # last agents locations
        eq_in_past = eq_in_past.view(B, N, self.args.future_length, 2) + cur_location

        if agents_idx.numel() != 0:
            mission_tensor = torch.zeros(B, N, 3, device=past.device)  # (B, N, 3)
            mission_tensor[:, agents_idx, :2] = one_mission  # inject mission coords
            mission_tensor[:, agents_idx, 2] = error_tolerance  # inject scalar toleranc
            mission_emb = self.mission_encoder(mission_tensor)  # (B, N, 8)
            mission_mask = torch.zeros(B, N, 1,   device=past.device)  # shape: (B, N, 1)
            mission_mask[:, agents_idx, :] = 1.0
            mission_emb = (mission_emb * mission_mask) #B, N, 8

        else:
            mission_emb = torch.zeros(B, N, 8, device=past.device)  # (B, N, 3)
            mission_mask = torch.zeros(B, N, 1, device=past.device)  # shape: (B, N, 1)

        futures = torch.cat([eq_in_past.view(B, N, -1),mission_mask ,mission_emb ], dim =-1)# (B, N, 29)
        eq_in_past_mission = self.past_encoder(futures)

        # eq_in_past_mission = torch.cat([eq_in_past.view(B, N, -1), mission_emb, mission_mask], dim=-1)  #flaging where these agents should be
        # eq_in_past_mission = self.past_encoder(eq_in_past_mission)

        prediction_q = prediction.permute(0, 1, 4, 2, 3).reshape(B, N, 20, -1) #B N 20 20
        #B, N, 20, 10, 2
        x_query = eq_in_past_mission.unsqueeze(2).expand(-1, -1, 20, -1)  # B N 20 20
        query_option_pair = torch.cat([x_query, prediction_q], dim=-1) # concatenate query and each option
        scores = self.groupnet(query_option_pair).squeeze(-1) # B, N, 20


        return  eq_in_past_mission.view(B, N, self.args.future_length,2), scores


    def inference(self, alpha, agents_idx,one_mission, error_tolerance, past, visability_mat_past, speed_past, directions_past, edge_features_past,edge_weights_past , prediction):
        B, N, T_past, _ = past.shape

        eq_in_past = self.MSG_past(past, visability_mat_past, speed_past, directions_past, edge_features_past,
                                   edge_weights_past)  # ,b, n, 20
        cur_location = past[:, :, [-1]]  # last agents locations
        eq_in_past = eq_in_past.view(B, N, self.args.future_length, 2) + cur_location

        if agents_idx.numel() != 0:
            mission_tensor = torch.zeros(B, N, 3, device=past.device)  # (B, N, 3)
            mission_tensor[:, agents_idx, :2] = one_mission  # inject mission coords
            mission_tensor[:, agents_idx, 2] = error_tolerance  # inject scalar toleranc
            mission_emb = self.mission_encoder(mission_tensor)  # (B, N, 8)
            mission_mask = torch.zeros(B, N, 1,   device=past.device)  # shape: (B, N, 1)
            mission_mask[:, agents_idx, :] = 1.0
            mission_emb = (mission_emb * mission_mask) #B, N, 8

        else:
            mission_emb = torch.zeros(B, N, 8, device=past.device)  # (B, N, 3)
            mission_mask = torch.zeros(B, N, 1, device=past.device)  # shape: (B, N, 1)

        futures = torch.cat([eq_in_past.view(B, N, -1), mission_mask, mission_emb], dim=-1)  # (B, N, 29)
        eq_in_past_mission = self.past_encoder(futures)

        prediction_q = prediction.permute(0, 1, 4, 2, 3).reshape(B, N, 20, -1)  # B N 20 20
        # B, N, 20, 10, 2
        x_query = eq_in_past_mission.unsqueeze(2).expand(-1, -1, 20, -1)  # B N 20 20
        query_option_pair = torch.cat([x_query, prediction_q], dim=-1)  # concatenate query and each option
        scores = self.groupnet(query_option_pair).squeeze(-1)  # B, N, 20

        pred_index = scores.argmax(dim=-1)  # the indexes

        return eq_in_past_mission.view(B, N, self.args.future_length,2), pred_index




class MessagePassingGraph(nn.Module):
    def __init__(self, mode, args, time, device, in_channels_speed, out_channels_speed, hidden_size_speed,
                 in_channels_dir, out_channels_dir, hidden_size_dir, heads, depth, SWM_out, edge_dim=16, bias=True):
        super(MessagePassingGraph, self).__init__()
        self.device = device
        self.args = args
        self.theta_vertex_speed = nn.Sequential(
            nn.Linear(in_channels_speed, hidden_size_speed, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size_speed, out_channels_speed))

        self.theta_vertex_direction = nn.Sequential(
            nn.Linear(in_channels_dir, hidden_size_dir, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size_dir, out_channels_dir),
        )

        self.attn_mlp_speed = nn.Sequential(
            nn.Linear(2 * out_channels_speed, hidden_size_speed, bias=bias),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_speed, 1)
        )


        self.attn_mlp_dir = nn.Sequential(
            nn.Linear(2 * out_channels_speed, hidden_size_speed, bias=bias),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_speed, 1)
        )


        self.transfom = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2, nhead=heads, dim_feedforward=hidden_size_speed, dropout=0, batch_first=True)
            , num_layers=depth)
        self.pos_encoder = PositionalEncoding1D(max_len=512, device=device)
        self.final_mlp = nn.Sequential(
            nn.Linear(SWM_out + out_channels_speed, args.future_length*2),
        )

        self.past_mlp = nn.Sequential(
            nn.Linear(2 * time, out_channels_speed),
            nn.ReLU()
        )
        self.out_channels_speed = out_channels_speed
        self.mode = mode


        self.SWM = SwarmLayer(out_channels_dir, SWM_out, 32, 4, n_dim=1, dropout=0.0, pooling='MEAN',
                              channel_first=True, cache=False).to(device)

    def forward(self, past, matrix, speed_past, directions_past):
        x_speed = self.theta_vertex_speed(speed_past)
        x_dir = self.theta_vertex_direction(directions_past)

        B, N, T_1 = speed_past.shape

        x_i_speed = x_speed.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        x_j_speed = x_speed.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        edge_feat = torch.cat([x_i_speed, x_j_speed], dim=-1)  # (B, N, N, 2D)

        attn_weights_speed = self.attn_mlp_speed(edge_feat).squeeze(-1)  # (B, N, N)
        attn_weights_speed = attn_weights_speed * matrix  # mask with adjacency
        attn_weights_speed = F.softmax(attn_weights_speed, dim=-1)  # softmax over neighbors

        out_speed = torch.matmul(attn_weights_speed, x_speed)

        x_i_dir = x_dir.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        x_j_dir = x_dir.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        edge_feat_dir = torch.cat([x_i_dir, x_j_dir], dim=-1)  # (B, N, N, 2D)

        attn_weights_dir = self.attn_mlp_dir(edge_feat_dir).squeeze(-1)  # (B, N, N)
        attn_weights_dir = attn_weights_dir * matrix  # mask with adjacency
        attn_weights_dir = F.softmax(attn_weights_dir, dim=-1)  # softmax over neighbors

        out_dir = torch.matmul(attn_weights_dir, x_dir)
        # print("out_dir", out_dir.shape)
        out_dir =  self.SWM(out_dir.permute(0, 2, 1)).permute(0, 2, 1)

        traj_hidden = self.past_mlp(past.reshape(B, N, self.args.past_length* 2))  # V reshape
        traj_pos = self.pos_encoder(traj_hidden)  # B N, 32

        for_transformer = traj_pos + out_speed


        if self.mode == "past":
            trans_inv_node = self.transfom(
                for_transformer.view(B * N, int(self.out_channels_speed / 2), 2))  # BN, 16, 2
        else:
            trans_inv_node = self.blstm(for_transformer.view(B * N, int(self.out_channels_speed / 2), 2))  # BN, 16, 32
            trans_inv_node = self.back_to_node(trans_inv_node)  # BN, 16, 2

        eq_in = torch.cat([trans_inv_node.reshape(B * N, -1), out_dir.reshape(B * N, -1)], dim=-1)  # BN , 40
        eq_in = self.final_mlp(eq_in)  # BN20

        return eq_in




class Node2Edge(nn.Module):
    def __init__(self, device, out_channels, hidden_size, bias=True, atten_neg_slope=0.1, drop_rate=0.2):
        super(Node2Edge, self).__init__()
        self.device = device

        self.atten_act = nn.Sequential(
            nn.Linear(2 * out_channels, hidden_size, bias=bias),
            nn.LeakyReLU(atten_neg_slope),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, matrix, node_edge_cat, edge_weights, x):
        attention_weight = self.atten_act(node_edge_cat)[:, :, :, 0]  # B, E, N

        # Agregation part
        H_weight = attention_weight * matrix  # element wise!
        H_weight = F.softmax(H_weight, dim=2)  # BEN
        edges_H = torch.matmul(H_weight, x)  # B,E, dim

        # update part
        e_weight = edge_weights.unsqueeze(-1)
        # print("e_weight", e_weight.shape)
        # print("edges_H", edges_H.shape)
        edges = e_weight * edges_H
        return edges  # B,E,dim


class Edge2Node(nn.Module):
    def __init__(self, device, out_channels, hidden_size, bias=True, atten_neg_slope=0.1, drop_rate=0.2):
        super(Edge2Node, self).__init__()
        self.device = device

        self.atten_act = nn.Sequential(
            nn.Linear(2 * out_channels, hidden_size, bias=bias),
            nn.LeakyReLU(atten_neg_slope),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, matrix, node_edge_cat, edge):
        attention_weight = self.atten_act(node_edge_cat)[:, :, :, 0]  # B, E, N

        # Agregation part
        H_weight = attention_weight * matrix  # element wise!
        H_weight = F.softmax(H_weight, dim=2)  # BEN
        nodes_H = torch.matmul(H_weight.permute(0, 2, 1), edge)  #
        # update part - no update...

        return nodes_H

class PositionalEncoding1D(nn.Module):
    def __init__(self, max_len=512, device='cpu'):
        super().__init__()
        self.register_buffer('pe', self._build_encoding(max_len, device))

    def _build_encoding(self, max_len, device):
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        div_term = 1 / (10000 ** (torch.arange(0, 1, 1).float() / 1))  # scalar version
        pe = torch.sin(position * div_term)  # (T, 1)
        return pe.squeeze(1)  # (T,)

    def forward(self, x):  # x: (B, N, T)
        T = x.size(2)
        pe = self.pe[:T]  # (T,)
        return x + pe.view(1, 1, T)  # broadcast to (B, N, T)

class MessagePassing(nn.Module):
    def __init__(self, mode, args, time, device, in_channels_speed, out_channels_speed, hidden_size_speed,
                 in_channels_dir, out_channels_dir, hidden_size_dir, heads, depth, SWM_out, edge_dim=16, bias=True):
        super(MessagePassing, self).__init__()
        self.device = device
        self.bias = bias
        self.args = args
        self.out_channels_speed = out_channels_speed
        self.node2edge_speed = Node2Edge(self.device, out_channels=out_channels_speed,
                                         hidden_size=hidden_size_speed).to(device)
        self.edge2node_speed = Edge2Node(self.device, out_channels=out_channels_speed,
                                         hidden_size=hidden_size_speed).to(device)

        self.node2edge_direction = Node2Edge(self.device, out_channels=out_channels_dir,
                                             hidden_size=hidden_size_dir).to(device)
        self.edge2node_direction = Edge2Node(self.device, out_channels=out_channels_dir,
                                             hidden_size=hidden_size_dir).to(device)

        self.theta_vertex_speed = nn.Sequential(
            nn.Linear(in_channels_speed, hidden_size_speed, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size_speed, out_channels_speed))

        self.theta_vertex_direction = nn.Sequential(
            nn.Linear(in_channels_dir, hidden_size_dir, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size_dir, out_channels_dir),
        )
        self.edge_encoder = nn.Embedding(4, edge_dim)  # 4 types of edges

        self.theta_hyperedge = nn.Sequential(
            nn.Linear(edge_dim, hidden_size_speed, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size_speed, out_channels_speed),
        )
        self.SWM = SwarmLayer(out_channels_dir, SWM_out, 32, 4, n_dim=1, dropout=0.0, pooling='MEAN',
                              channel_first=True, cache=False).to(device)
        self.transfom = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2, nhead=heads, dim_feedforward=hidden_size_speed, dropout=0, batch_first=True)
            , num_layers=depth)
        self.pos_encoder = PositionalEncoding1D(max_len=512, device=device)
        self.final_mlp = nn.Sequential(
            nn.Linear(SWM_out + out_channels_speed, args.future_length*2),
        )

        self.past_mlp = nn.Sequential(
            nn.Linear(2 * time, out_channels_speed),
            nn.ReLU()
        )
        self.mode = mode
        self.blstm = BiLSTM(2, int(out_channels_speed / 2), num_layers=1, dropout=0.0).to(device)
        self.back_to_node = nn.Sequential(
            nn.Linear(out_channels_speed, 2),
        )

        # self.weight = nn.Parameter(torch.randn(args.past_length - 1, int(out_channels_dir/2)))
        # self.weight_final = nn.Parameter(torch.randn(2* (args.future_length), args.future_length))
        # if bias:
        #     # optional 2D bias per output vector
        #     self.bias = nn.Parameter(torch.zeros(int(out_channels_dir/2), 2))
        # else:
        #     self.register_parameter('bias', None)

    def forward(self, trajectory, visability_mat, speed, directions, edge_features, edge_weights):
        """
        visability_mat : B, N, N
        speed : B, N, T-1
        directions : B, N, T-1
        Returns:
        edge_features: (B, N)
        edge_weights: (B, 12)
        past: B, N, 5, 2
        """
        B, N, T_past, _ = trajectory.shape
        x_speed = self.theta_vertex_speed(speed)
        y_encoder = self.edge_encoder(edge_features.long())
        y = self.theta_hyperedge(y_encoder)

        node_num = x_speed.shape[1]
        edge_num = edge_features.shape[1]

        x_rep_speed = (x_speed[:, :, None, :].transpose(2, 1)).repeat(1, edge_num, 1, 1)
        edge_rep = y[:, :, None, :].repeat(1, 1, node_num, 1)
        node_edge_cat_speed = torch.cat((x_rep_speed, edge_rep), dim=-1)

        inv_edges = self.node2edge_speed(visability_mat, node_edge_cat_speed, edge_weights, x_speed)  # B, E, out_speed
        inv_node = self.edge2node_speed(visability_mat, node_edge_cat_speed, inv_edges)  # B, N, out_speed

        ############################################################################

        x_dir = self.theta_vertex_direction(directions)
        # x_dir = directions
        # M, K = self.weight.shape
        # x_dir = torch.einsum('bnid,ij->bnjd', x_dir, self.weight)
        # if self.bias is not None:
        #     x_dir = x_dir + self.bias.view(1, 1, K, 2)
        # x_dir = x_dir.reshape(B, N, K*2)

        x_rep_dir = (x_dir[:, :, None, :].transpose(2, 1)).repeat(1, edge_num, 1, 1)
        node_edge_cat_dir = torch.cat((x_rep_dir, edge_rep), dim=-1)

        eqv_edges = self.node2edge_direction(visability_mat, node_edge_cat_dir, edge_weights, x_dir)  # B, E, out_dir
        eqv_nodes = self.edge2node_direction(visability_mat, node_edge_cat_dir, eqv_edges)  # B, N, out_dir

        # print("eqv_nodes", eqv_nodes.shape)
        eqv_nodes_final = self.SWM(eqv_nodes.permute(0, 2, 1)).permute(0, 2, 1)  # B, N, 8 #DID NOT USED AT THE END IN SIN_COS FOR EQV
        # print("eqv_nodes_final", eqv_nodes_final.shape)

        # traj_hidden = self.past_mlp(trajectory.view(B, N, T_past))
        # print("trajectory", trajectory.shape)
        traj_hidden = self.past_mlp(trajectory.reshape(B, N, T_past*2)) #V reshape
        # print("past_hidden", past_hidden.shape)
        traj_pos = self.pos_encoder(traj_hidden)  # B N, 32
        # acceleration = speed[:, :, 1:] - speed[:, :, :-1]
        for_transformer = traj_pos+inv_node
        # print("for_transformer",for_transformer.shape)
        # for_transformer = torch.cat([for_transformer,acceleration], dim=-1) #32+T-2

        if self.mode == "past":
            trans_inv_node = self.transfom(for_transformer.view(B*N, int(self.out_channels_speed/2), 2))# BN, 16, 2
        else:
            trans_inv_node = self.blstm(for_transformer.view(B*N, int(self.out_channels_speed/2), 2))  # BN, 16, 32
            # print("trans_inv_node", trans_inv_node.shape)
            trans_inv_node = self.back_to_node(trans_inv_node) #BN, 16, 2
            # print("trans_inv_node", trans_inv_node.shape)

        # print("trans_inv_node", trans_inv_node.shape)
        # print("eqv_nodes_final,", eqv_nodes_final.shape)

        eq_in = torch.cat([trans_inv_node.reshape(B*N, -1), eqv_nodes_final.reshape(B*N, -1)], dim=-1)  # BN , 40

        # eq_in = torch.einsum('bnid,ij->bnjd', eq_in.view(B, N, self.args.future_length*2, 2), self.weight_final)
        eq_in = self.final_mlp(eq_in)  # BN20

        # print("eq_in",eq_in[0], eq_in.shape)
        return eq_in


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,  # input/output will be (B, T, D)
            bidirectional=True
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        output, (h_n, c_n) = self.lstm(x)
        # output: (B, T, 2*hidden_dim) --- forward and backward are concatenated
        return output


class SwarmConvLSTMCell(nn.Module):
    # from https://github.com/zalandoresearch/SWARM
    def __init__(self, n_in, n_out, n_dim, pooling, cache=False):
        """
        Create a SwarmConvLSTMCell. We use 1-by-1 convolutions to carry on entities individually. The entities are aligned
        in a 1d or 2d spatial structure. Note that, unless pooling is 'CAUSAL', this setup is indeed permutation-equivariant.
        Populations of different sizes (different number of entities) can be grouped in one batch were missing entities
        will be padded and masked out.
        :param n_in: input dimension of the entities
        :param n_out: output dimension of the entities
        :param n_dim: dimension of the spatial arrangement of the entities (1 or 2)
        :param pooling: pooling method 'MEAN' or 'CAUSAL'
        :param cache: cache the result of self.Wih(x) in self.x_cache
        """
        assert isinstance(pooling, Pooling)
        assert pooling.n_dim == n_dim
        assert pooling.n_in == n_out

        super().__init__()

        self.n_in = n_in
        self.n_out = n_out

        if n_dim == 2:
            # output is 4 time n_out because it will be split into
            # input, output, and forget gates, and cell input
            self.Wih = nn.Conv2d(n_in, 4 * n_out, (1, 1), bias=True)
            self.Whh = nn.Conv2d(n_out, 4 * n_out, (1, 1), bias=False)
            self.Whp = nn.Conv2d(pooling.n_out, 4 * n_out, (1, 1), bias=False)
        elif n_dim == 1:
            self.Wih = nn.Conv1d(n_in, 4 * n_out, 1, bias=True)
            self.Whh = nn.Conv1d(n_out, 4 * n_out, 1, bias=False)
            self.Whp = nn.Conv1d(pooling.n_out, 4 * n_out, 1, bias=False)
        else:
            raise ValueError("dim {} not supported".format(n_dim))

        self.n_dim = n_dim

        self.pooling = pooling

        self.cache = cache
        self.x_cache = None

    def forward(self, x, mask=None, hc=None):
        """
        Forward process the SWARM cell
        :param x: input, size is (N,n_in,E1,E2,...)
        :param mask: {0,1}-mask, size is (N,E1,E2,...)
        :param hc: (hidden, cell) state of the previous iteration or None. If not None both their size is (N,n_out, E1,E2,...)
        :return: (hidden, cell) of this iteration
        """
        # x is (N,n_in,...)

        x_sz = x.size()
        N, C = x_sz[:2]
        # print("NC", N, C)
        assert C == self.n_in

        if hc is None:
            c = torch.zeros((N, self.n_out, *x_sz[2:]), dtype=x.dtype, device=x.device)
            # print("c", c.shape)
            tmp = self.Wih(x)  # (N,4*n_out, H,W)
            self.x_cache = tmp
        else:
            h, c = hc
            pool = self.Whp(self.pooling(h, mask))
            tmp = (self.x_cache if self.cache else self.Wih(x)) + self.Whh(h) + pool  # (N,4*n_out, H,W)

        tmp = tmp.view(N, 4, self.n_out, *x_sz[2:])

        ig = torch.sigmoid(tmp[:, 0])
        fg = torch.sigmoid(tmp[:, 1])
        og = torch.sigmoid(tmp[:, 2])
        d = torch.tanh(tmp[:, 3])

        c = c * fg + d * ig
        h = og * torch.tanh(c)

        return h, c


class SwarmLayer(nn.Module):
    #from https://github.com/zalandoresearch/SWARM
    def __init__(self,
                 n_in,
                 n_out,
                 n_hidden,
                 n_iter,
                 n_dim=2,
                 dropout=0.0,
                 pooling='MEAN',
                 channel_first=True,
                 cache=False):
        """
        Create a SwarmLayer that repeatedly executes a SwarmCell for a given number of iterations
        :param n_in: number of dimensions of input entities
        :param n_out: number of dimensions of output entities
        :param n_hidden: number of dimensions of entities in intermediate iterations
        :param n_iter: number of iterations
        :param n_dim: spatial entity layout (1 or 2)-d
        :param dropout: dropout rate (applied to h, not c, between iterations)
        :param pooling: to be used in the SWARM cell 'CAUSAL' or 'MEAN'
        :param channel_first: entity dimension is dimension 1, right after batch dimension (default), otherwise it is last
        :param cache: perform the computation of self.cell.Wih(x) only once and cache it over the rest of the iterations
        """
        super().__init__()

        self.n_iter = n_iter

        if pooling == 'MEAN':
            pooling = Mean(n_hidden, n_hidden, n_dim)
        elif pooling == 'CAUSAL':
            pooling = Causal(n_hidden, n_hidden, n_dim)
        elif pooling == 'POMA':
            pooling = PoolingMaps(n_hidden, 4, n_dim)
        elif isinstance(pooling, Pooling):
            pass
        else:
            raise ValueError

        self.cell = SwarmConvLSTMCell(n_in, n_hidden, n_dim=n_dim, pooling=pooling, cache=cache)

        self.n_dim = n_dim
        if n_dim == 2:
            # an output feed forward layer after. Because channel_first is default, is is implemented by a 1-by-1 conv.
            self.ffwd = nn.Conv2d(2 * n_hidden, n_out, (1, 1), bias=True)
        elif n_dim == 1:
            self.ffwd = nn.Conv1d(2 * n_hidden, n_out, 1, bias=True)
        else:
            raise ValueError("dim {} not supported".format(n_dim))

        if dropout > 0:
            self.drop = nn.Dropout2d(dropout)
        else:
            self.drop = None

        self.channel_first = channel_first

    def forward(self, x, mask=None):
        """
        forward process the SwarmLayer
        :param x: input
        :param mask: entity mask
        :return:
        """

        # 1. permute channels dimension to the end if not channels_first
        if not self.channel_first:
            if self.n_dim == 1:
                x = x.transpose(1, 2)
            elif self.n_dim == 2:
                x = x.transpose(1, 2).transpose(2, 3)

        # 2. iteratively execute SWARM cell
        hc = None
        for i in range(self.n_iter):

            hc = self.cell(x, mask, hc)

            # 2a. apply dropout on h if desired
            if self.drop is not None:
                h, c = hc
                h = self.drop(h)
                hc = (h, c)

        # 3. execute the output layer on the concatenation of h an c
        h, c = hc
        hc = torch.cat((h, c), dim=1)
        y = self.ffwd(hc)

        # 4. back-permute the channels dimension
        if not self.channel_first:
            if self.n_dim == 1:
                y = y.transpose(1, 2)
            elif self.n_dim == 2:
                y = y.transpose(2, 3).transpose(1, 2)
        return y


class Pooling(nn.Module):

    def __init__(self, n_in, n_out, n_dim):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_dim = n_dim

        assert self.n_dim==1 or self.n_dim==2


    def forward(self, x, mask):
        # x is (N, n_in, E) or (N, n_in, E1, E2)
        # mask is (N, E) or (N, E1, E2)

        raise NotImplementedError("Pooling is only an abstract bas class")

class Mean( Pooling):

    def __init__(self, n_in, n_out, n_dim):
        super().__init__(n_in, n_out, n_dim)

        assert n_in==n_out


    def forward(self, x, mask=None):

        x_sz = x.size()

        if self.n_dim==1:
            pooling_dim = 2
        else:
            pooling_dim = (2,3)

        if mask is None:
            # 2. compute mean over spatial dimensions
            pool = x.mean(dim=pooling_dim, keepdim=True).expand(x_sz)
        else:
            # 2. compute masked mean over spatial dimensions
            mask = mask.view((x_sz[0], 1, *x_sz[2:])).float()
            pool = (x * mask).sum(dim=pooling_dim, keepdim=True).expand(x_sz)
            pool = pool / mask.sum(dim=pooling_dim, keepdim=True).expand(x_sz)
            pool = pool.view(x_sz)

        return pool



class Causal( Pooling):

    def __init__(self, n_in, n_out, n_dim):
        super().__init__(n_in, n_out, n_dim)

        assert n_in == n_out

    def forward(self, x, mask=None):

        if mask is not None:
            raise NotImplementedError("Causal pooling is not yet implemented for masked input!")

        x_sz = x.size()

        # 1. flatten all spatial dimensions
        pool = x.view((x_sz[0], self.n_in, -1))
        # 2. compute cumulative means of non-successort entities
        pool = torch.cumsum(pool, dim=2) / (torch.arange(np.prod(x_sz[2:]), device=pool.device).float() + 1.0).view(1, 1, -1)
        # 3. reshape to the original spatial layout
        pool = pool.view(x_sz)

        return pool



class PoolingMaps(Pooling):

    def __init__(self, n_in, n_slices, n_dim):

        n_out = n_in-2*n_slices
        super().__init__(n_in, n_out, n_dim)
        self.n_slices = n_slices


    def forward(self, x, mask = None):

        # x is (N, n_in+2*n_slices, E)

        assert x.size(1) == self.n_in

        a = x[:, :self.n_in-2*self.n_slices]
        b = x[:, self.n_in-2*self.n_slices:-self.n_slices]
        c = x[:, -self.n_slices:]

        if mask is not None:
            b = b+torch.log(mask.unsqueeze(1).float())
        b = torch.softmax(b.view(b.size(0),b.size(1),-1), dim=2).view(b.size())

        tmp = a.unsqueeze(1) * b.unsqueeze(2) #(N, n_slices, n_in, E)
        #print(tmp.size())
        tmp  = tmp.sum(dim=3, keepdim=True) #(N, n_slices, n_in, 1)
        #print(tmp.size())
        tmp =  tmp * c.unsqueeze(2)
        #print(tmp.size())
        out = torch.sum(tmp, dim=1)

        return out


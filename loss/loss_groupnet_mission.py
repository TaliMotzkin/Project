import torch
import random

from scipy.interpolate import CubicSpline, interp1d
import numpy as np
import matplotlib.pyplot as plt

class LossCompute:
    def __init__(self, args ,netG=None, netD=None, netGM=None, ):
        self.netG = netG
        self.netD = netD
        self.netGM = netGM
        self.device = args.device
        self.bce = torch.nn.BCELoss()
        self.l2_weight = args.l2_weight
        self.args = args
        self.ramp_k = 0.3
        self.max_alpha = 10


    def compute_generator_loss(self,  data,  agents_tragets, agents_idx, error_tolerance, epoch):

        batch_size = data['past_traj'].shape[0]
        agent_num = data['past_traj'].shape[1]
        past_traj = data['past_traj'].view(batch_size * agent_num, self.args.past_length, 2).to(self.device).contiguous()
        future_traj = data['future_traj'].view(batch_size * agent_num, self.args.future_length, 2).to(self.device).contiguous() # are predictions from G!


        #mission aware agents
        pred_traj,recover_traj, qz_distribution,pz_distribution, diverse_pred_traj  = self.netGM(data,  agents_tragets, agents_idx, error_tolerance)


        #non mission aware agents (current ground truth [simulated] data)
        # mission aware loss - of mission and trajectory
        loss_pred = self.calculate_loss_pred(pred_traj, future_traj, batch_size)  # future loss

        diff = diverse_pred_traj - future_traj.unsqueeze(1)  # future - 20 samples
        avg_dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
        best_sample_indices = avg_dist.argmin(dim=1)  # BN
        sample_idx = torch.arange(batch_size * agent_num, device=diverse_pred_traj.device)
        best_trajectories = diverse_pred_traj[sample_idx, best_sample_indices].view(batch_size, agent_num,  self.args.future_length,
                                                                                    2)  # shape: (B,N, T, 2)
        loss_diverse = self.calculate_loss_diverse(
            best_trajectories.view(batch_size * agent_num, self.args.future_length, 2), future_traj,
            batch_size)  # future - 32*20, T, 2

        loss_recover = self.calculate_loss_recover(recover_traj, past_traj, batch_size)  # past loss
        loss_kl = self.calculate_loss_kl(qz_distribution, pz_distribution, batch_size, agent_num,
                                         self.args.min_clip)
        loss_mission = self.calcualte_mission_loss(pred_traj, future_traj, agents_tragets, agents_idx, epoch,
                                                   batch_size, agent_num, self.args, error_tolerance, loss_pred)

        loss_mission_div = self.calcualte_mission_loss(best_trajectories, future_traj, agents_tragets, agents_idx,
                                                       epoch, batch_size, agent_num, self.args, error_tolerance,
                                                       loss_diverse)

        # fake_traj = future_traj.clone()
        # fake_traj = fake_traj.view(batch_size, agent_num, self.args.future_length, 2)
        # fake_traj[:, agents_idx, :,:] = pred_traj.view(batch_size, agent_num, self.args.future_length, 2)[:, agents_idx, :, :]

        scores_fake = self.netD(torch.cat([past_traj.view(batch_size, agent_num, self.args.past_length, 2),
                                           pred_traj.view(batch_size, agent_num, self.args.future_length, 2) ], dim = 2)) # B, N
        # scores_fake = self.netD(agents_traj, past, visability_mat_past, velocity_past, direction_past,
        #                         edge_features_past, edge_weights_past, prediction)

        # GANG + KL losses
        discriminator_loss = self.gan_g_loss(scores_fake)


        total_loss = (
                loss_pred +
                loss_recover +
                loss_kl +
                loss_mission +
                loss_diverse +
                loss_mission_div+
                 discriminator_loss
        )

        return (total_loss,
                loss_pred.item(),
                loss_recover.item(),
                loss_kl.item(),
                loss_mission.item(),
                loss_diverse.item(),
                loss_mission_div.item(),
                discriminator_loss.item())


    def calculate_loss_pred(self, pred, target, batch_size):
        loss = (target - pred).pow(2).sum()
        loss /= batch_size
        loss /= pred.shape[1]
        return loss




    def compute_discriminator_loss(self, data,  agents_tragets, agents_idx, error_tolerance, epoch):

        batch_size = data['past_traj'].shape[0]
        agent_num = data['past_traj'].shape[1]
        past_traj = data['past_traj'].to(self.device).contiguous()
        future_traj = data['future_traj'].view(batch_size * agent_num, self.args.future_length, 2).to(self.device).contiguous() # are predictions from G!


        #mission aware agents
        pred_traj,_, _, _, _  = self.netGM(data,  agents_tragets, agents_idx, error_tolerance)

        # scores_fake = self.netD(fake_traj, past, visability_mat_past, velocity_past, direction_past,
        #                         edge_features_past, edge_weights_past, prediction)  # B, N

        scores_real = self.netD(torch.cat([past_traj, future_traj.view(batch_size ,agent_num, self.args.future_length, 2)], dim = 2))

        # fake_traj = future_traj.clone()
        pred_traj_detached = pred_traj.view(batch_size , agent_num, self.args.future_length, 2).detach()
        # fake_traj = fake_traj.view(batch_size, agent_num, self.args.future_length, 2)
        # fake_traj[:, agents_idx, :, :] = pred_traj_detached.view(batch_size, agent_num, self.args.future_length, 2)[:, agents_idx, :, :]

        scores_fake = self.netD(torch.cat([past_traj,pred_traj_detached], dim = 2))  # B, N

        loss_real, loss_fake = self.gan_d_loss(scores_fake, scores_real, agents_idx)  # BCEloss
        total_loss = loss_real + loss_fake

        if agents_idx.numel() != 0:
            scores_fake_final = scores_fake[:, agents_idx]
            # print("list(set(range(agent_num))",list(set(range(agent_num))))
            # print(" set(agents_idx))", agents_idx.tolist())
            # print("scores_fake", scores_fake_final[0])
            uncontrolled_idx = [i for i in range(agent_num) if i not in agents_idx.tolist()]
            # print("uncontrolled_idx", uncontrolled_idx)
            if len(uncontrolled_idx) > 0:
                scores_uncontrolled_final = scores_fake[:, uncontrolled_idx]
            else:
                scores_uncontrolled_final = torch.empty(scores_fake.shape[0], 0, device=scores_fake.device)
            return total_loss, loss_real.item(), loss_fake.item(), scores_fake_final, scores_real, scores_uncontrolled_final
        else:
            scores_fake_final = scores_fake
            scores_uncontrolled_final = torch.empty(scores_fake.shape[0], 0, device=scores_fake.device)
            return total_loss, loss_real.item(), loss_fake.item(), scores_fake_final, scores_real, scores_uncontrolled_final



    def gan_g_loss(self, scores_fake):
        y_fake = torch.ones_like(scores_fake) * random.uniform(1, 1.0)
        #Unlike the discriminator's gan_d_loss, the generator wants scores_fake to be close to 1.0
        return self.bce(scores_fake, y_fake)

    def gan_d_loss(self, scores_fake, scores_real, agent_idx):


        y_real = torch.ones_like(scores_real) * random.uniform(1, 1.0)
        loss_real = self.bce(scores_real, y_real)

        # y_fake = torch.zeros_like(scores_fake)
        # mask = torch.zeros_like(y_fake)
        # mask[:, agent_idx] = 1.0
        y_fake = torch.empty_like(scores_fake).uniform_(0.0, 0)
        # loss_fake_matrix = F.binary_cross_entropy(scores_fake, y_fake, reduction='none')
        # loss_fake = (loss_fake_matrix * mask).sum() / mask.sum().clamp(min=1)

        # y_fake = torch.ones_like(scores_fake)
        # y_fake[:, agent_idx] = 0.0
        #
        # weights = torch.ones_like(scores_fake)
        # weights[:, agent_idx] = 1.5*scores_fake.shape[1] / len(agent_idx)
        #
        # loss_fake = F.binary_cross_entropy(scores_fake, y_fake, weight=weights, reduction='mean')

        loss_fake = self.bce(scores_fake, y_fake)


        return loss_real, loss_fake

    def coose_diverse(self, batch_size, agent_num, diverse_pred_traj,agents_idx , agents_tragets, future_traj):


        diverse_pred_traj = diverse_pred_traj.view(batch_size, agent_num, 20, self.args.future_length, 2)
        p1 = diverse_pred_traj[:, agents_idx, :, :-1, :]  # (B, C, 20, T-1, 2)
        p2 = diverse_pred_traj[:, agents_idx, :, 1:, :]  # (B, C, 20, T-1, 2)
        m = agents_tragets.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1, 2)

        seg = p2 - p1
        seg_len = torch.norm(seg, dim=-1, keepdim=True).clamp(min=1e-6)
        proj = ((m - p1) * seg).sum(-1, keepdim=True) / (seg_len ** 2)
        proj_clamped = proj.clamp(0, 1)
        closest = p1 + proj_clamped * seg  # (B, C, 20, T-1, 2)
        dists = (closest - m).norm(dim=-1)  # (B, C, 20, T-1)
        min_dists = dists.min(dim=-1).values  # (B, C, 20)
        best_controlled = min_dists.argmin(dim=-1)  # (B, C)


        uncontrolled_idx = [i for i in range(agent_num) if i not in agents_idx]

        if uncontrolled_idx:
            diff = diverse_pred_traj[:, uncontrolled_idx, :, :, :] - future_traj.view(batch_size, agent_num,
                                                                                      self.args.future_length, 2)[:,
                                                                     uncontrolled_idx,
                                                                     :, :].unsqueeze(2)  # (B, N_U, 20, T, 2)
            avg_dist = diff.pow(2).sum(dim=-1).sum(dim=-1)  # (B, N_u, 20)
            best_uncontrolled = avg_dist.argmin(dim=-1)  # (B, N_u)
        best_indices = torch.zeros((batch_size, agent_num), dtype=torch.long, device=self.device)
        best_indices[:, agents_idx] = best_controlled
        if uncontrolled_idx:
            best_indices[:, uncontrolled_idx] = best_uncontrolled
        sample_idx = torch.arange(batch_size * agent_num, device=self.device)
        best_trajectories = diverse_pred_traj.view(batch_size * agent_num, 20, self.args.future_length, 2)[
            sample_idx, best_indices.view(-1)]
        best_trajectories = best_trajectories.view(batch_size, agent_num, self.args.future_length, 2)

        return best_trajectories

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

    def calcualte_within_reach(self, pred_traj, future_traj, controlled_targets, agents_idx, epoch, batch_size, agent_num, args, error_tolerance):
        B = batch_size
        N = agent_num
        T = args.future_length

        pred_traj_all = pred_traj.view(B, N, T, 2)
        future_traj_all = future_traj.view(B, N, T, 2)
        pred_controlled = pred_traj_all[:, agents_idx, :, :]  # B, C, T, 2

        # compute min distance and if the goal is achieved
        p1 = pred_controlled[:, :, :-1, :]  # (B, C, T-1, 2)
        p2 = pred_controlled[:, :, 1:, :]
        m = controlled_targets.unsqueeze(2)

        seg = p2 - p1
        seg_len = torch.norm(seg, dim=-1, keepdim=True).clamp(min=1e-6)
        proj = ((m - p1) * seg).sum(-1, keepdim=True) / (seg_len ** 2)
        proj_clamped = proj.clamp(0, 1)
        closest = p1 + proj_clamped * seg  # (B, C, T-1, 2)
        d = (closest - m).norm(dim=-1)  # (B, C, T-1)
        achieved = (d <= error_tolerance).any(dim=-1)  # (B, C)


        #compute if the target is within reahch or not
        gt_movement = torch.norm(
            future_traj_all[:, agents_idx, 1:, :] - future_traj_all[:, agents_idx, :-1, :],
            dim=-1
        ).sum(dim=-1)  # (B, C)
        start_to_target = torch.norm(pred_controlled[:, :, 0, :] - controlled_targets, dim=-1)  # (B, C)
        within_reach = (start_to_target <= 0.5*gt_movement)

        mask = within_reach & ~achieved

        return mask

    def calcualte_mission_loss(self, pred_traj, future_traj, controlled_targets, agents_idx, epoch, batch_size, agent_num, args, error_tolerance,loss_pred):

        B = batch_size
        N = agent_num

        if agents_idx.numel() == 0:
            loss_mission_div = torch.tensor(0, dtype=torch.float32, device=pred_traj.device)
            return loss_mission_div

        C = len(agents_idx)
        T = args.future_length

        pred_traj_all = pred_traj.view(B, N, T, 2)
        future_traj_all = future_traj.view(B, N, T, 2)
        pred_controlled = pred_traj_all[:, agents_idx, :, :]  # B, C, T, 2


        #compute min distance and if the goal is achieved
        p1 = pred_controlled[:, :, :-1, :] # (B, C, T-1, 2)
        p2 = pred_controlled[:, :,  1:, :]
        m = controlled_targets.unsqueeze(2)

        seg = p2 - p1
        seg_len = torch.norm(seg, dim=-1, keepdim=True).clamp(min=1e-6)
        proj = ((m - p1) * seg).sum(-1, keepdim=True) / (seg_len ** 2)
        proj_clamped = proj.clamp(0, 1)
        closest = p1 + proj_clamped * seg # (B, C, T-1, 2)
        d = (closest - m).norm( dim=-1)  # (B, C, T-1)
        achieved = (d <= error_tolerance).any(dim=-1)  # (B, C)
        min_distance = d.min(dim=-1).values  # (B, C)


        #compute if the target is within reahch or not
        gt_movement = torch.norm(
            future_traj_all[:, agents_idx, 1:, :] - future_traj_all[:, agents_idx, :-1, :],
            dim=-1
        ).sum(dim=-1)  # (B, C)
        start_to_target = torch.norm(pred_controlled[:, :, 0, :] - controlled_targets, dim=-1)  # (B, C)
        within_reach = (start_to_target <= args.how_far*gt_movement)
        # very_far = ~within_reach

        # reach_center = 0.5 * gt_movement  # midpoint of the sigmoid
        # reach_sharpness = 10.0  # higher -> steeper transition
        # d_thresh = reach_center - start_to_target
        # reach_score = torch.sigmoid(reach_sharpness * d_thresh)


        start_dist = torch.norm(pred_controlled[:, :, 0, :] - controlled_targets, dim=-1)
        end_dist = torch.norm(pred_controlled[:, :, -1, :] - controlled_targets, dim=-1)
        direction_delta = end_dist - start_dist
        direction_sign = direction_delta.sign()  # +1 bad, -1 good
        abs_delta = direction_delta.abs().clamp(min=0.003)
        positive_branch = abs_delta
        negative_branch = 0.1 / abs_delta

        direction_loss = (direction_sign < 0).float() * negative_branch + \
                         (direction_sign >= 0).float() * positive_branch

        # print("achieved", achieved[0])
        # print("very_far", very_far[0])
        # print("direction_sign", direction_sign[0])
        # print("direction_loss", direction_loss[0])
        # print("min_distance", min_distance[0])


        mission_loss = torch.zeros(B, C, device=pred_traj.device)
        mask_not_achieved = ~achieved
        mask_direction_bad = direction_sign > 0
        mask_direction_good = direction_sign <= 0


        mask_4 = mask_not_achieved & within_reach & mask_direction_good
        mission_loss[mask_4] =  min_distance[mask_4] + 0.2 * direction_loss[mask_4]


        mask_5 = mask_not_achieved  & within_reach& mask_direction_bad
        mission_loss[mask_5] = min_distance[mask_5] + 2.5 * direction_loss[mask_5]

        mission_weight = min(2.0, epoch / 5)
        mission_loss_final = (mission_weight * mission_loss).mean()

        if mission_loss_final > loss_pred and not (mask_4.any() or mask_5.any()):
            scaling = (loss_pred / mission_loss_final).detach() * 0.95  # keep it just under
            mission_loss_final = mission_loss_final * scaling

        # # mission statistics
        # missions_total = achieved.numel()
        # missions_achieved = achieved.sum().item()
        #
        # stats = {
        #     "missions_total": missions_total,
        #     "missions_achieved": missions_achieved,
        #     "mask_2_bad_direction_very_far": mask_2.sum().item(),
        #     "mask_3_good_direction_very_far": mask_3.sum().item(),
        #     "mask_4_good_direction_close": mask_4.sum().item(),
        #     "mask_5_bad_direction_close": mask_5.sum().item()
        # }
        # print(stats)

        return mission_loss_final

    def stretch_gt_to_mission(self, future_traj, mission, mask):

        Bc, T, _ = future_traj.shape
        new_gt = future_traj.clone()
        t_uniform = np.linspace(0.0, 1.0, T)

        gt_np = future_traj.cpu().numpy()         # (Bc, T, 2)
        segs = gt_np[:,1:,:] - gt_np[:,:-1,:]      # (Bc, T-1, 2)
        seg_lens = np.linalg.norm(segs, axis=-1)  # (Bc, T-1)
        total_len = seg_lens.sum(axis=1)          # (Bc,)

        # start-to-mission distance for each:
        start_np   = gt_np[:,0,:]                 # (Bc,2)
        mission_np = mission.cpu().numpy()        # (Bc,2)
        dist_sm    = np.linalg.norm(mission_np - start_np, axis=1)  # (Bc,)

        #timeline at which we *should* hit the mission:
        alpha = dist_sm / (total_len + 1e-6)
        alpha = np.clip(alpha, 0.0, 1.0)# keep in [0,1]
        eps = 1e-3
        idx = mask.nonzero(as_tuple=True)[0]
        for i in idx.tolist():

            alpha_i = np.clip(alpha[i], eps, 1.0 - eps)

            t_knots = np.array([0.0, alpha_i, 1.0])
            x_knots = np.array([gt_np[i,0,0], mission_np[i,0], gt_np[i,-1,0]])
            y_knots = np.array([gt_np[i,0,1], mission_np[i,1], gt_np[i,-1,1]])

            cs_x = CubicSpline(t_knots, x_knots)
            cs_y = CubicSpline(t_knots, y_knots)


            xs = cs_x(t_uniform)
            ys = cs_y(t_uniform)

            warped = torch.from_numpy(np.stack([xs, ys], axis=1)).to(future_traj.device)

            warped[-1] = future_traj[i, -1]
            new_gt[i] = warped

            if i == 0:
                plt.figure()
                plt.plot(gt_np[i, :, 0], gt_np[i, :, 1], label='Original GT')
                plt.plot(xs, ys, label='Warped GT')
                plt.scatter(mission_np[i, 0], mission_np[i, 1], marker='x', label='Mission')
                plt.title(f'Trajectory {i}: Original vs Warped')
                plt.xlabel('X coordinate')
                plt.ylabel('Y coordinate')
                plt.legend()
                plt.savefig("GANG/plots/plt.png")
                plt.close()

        return new_gt

    def stretch_and_reparam_by_arclength_torch(self, future_traj: torch.Tensor,
                                               mission: torch.Tensor,
                                               mask: torch.Tensor,
                                               fine_factor: int = 10) -> torch.Tensor:

        device = future_traj.device
        gt_np = future_traj.detach().cpu().numpy()  # (Bc, T, 2)
        mission_np = mission.detach().cpu().numpy()  # (Bc, 2)
        mask_np = mask.detach().cpu().numpy()  # (Bc,)

        Bc, T, _ = gt_np.shape
        new_gt_np = gt_np.copy()

        segs = gt_np[:, 1:, :] - gt_np[:, :-1, :]  # (Bc, T-1, 2)
        seg_lens = np.linalg.norm(segs, axis=-1)  # (Bc, T-1)
        total_len = seg_lens.sum(axis=1)  # (Bc,)

        start_np = gt_np[:, 0, :]  # (Bc,2)
        dist_sm = np.linalg.norm(mission_np - start_np, axis=1)  # (Bc,)
        alpha = dist_sm / (total_len + 1e-6)

        alpha = np.clip(alpha, 1e-3, 1.0 - 1e-3)  # (Bc,)

        t_uniform = np.linspace(0.0, 1.0, T)

        for i in np.where(mask_np)[0]:
            t_knots = np.array([0.0, alpha[i], 1.0])
            x_knots = np.array([gt_np[i, 0, 0], mission_np[i, 0], gt_np[i, -1, 0]])
            y_knots = np.array([gt_np[i, 0, 1], mission_np[i, 1], gt_np[i, -1, 1]])
            cs_x = CubicSpline(t_knots, x_knots)
            cs_y = CubicSpline(t_knots, y_knots)

            tf = np.linspace(0.0, 1.0, T * fine_factor)
            xs_f = cs_x(tf)
            ys_f = cs_y(tf)
            dx = np.diff(xs_f)
            dy = np.diff(ys_f)
            seg_lf = np.hypot(dx, dy)
            cumlen = np.concatenate([[0], np.cumsum(seg_lf)])
            L = cumlen[-1]

            interp_t = interp1d(cumlen, tf, kind="linear")
            target_s = np.linspace(0.0, L, T)
            t_samp = interp_t(target_s)

            xs = cs_x(t_samp)
            ys = cs_y(t_samp)
            xs[-1], ys[-1] = gt_np[i, -1, 0], gt_np[i, -1, 1]

            new_gt_np[i] = np.stack([xs, ys], axis=1)

            # if i == 0:
            #     plt.figure()
            #     plt.plot(gt_np[i, :, 0], gt_np[i, :, 1], label='Original GT')
            #     plt.plot(xs, ys, label='Warped GT')
            #     plt.scatter(mission_np[i, 0], mission_np[i, 1], marker='x', label='Mission')
            #     plt.title(f'Trajectory {i}: Original vs Warped')
            #     plt.xlabel('X coordinate')
            #     plt.ylabel('Y coordinate')
            #     plt.legend()
            #     plt.savefig("GANG/plots/plt_new_GT.png")
            #     plt.close()


        new_gt = torch.from_numpy(new_gt_np).to(device)
        return new_gt

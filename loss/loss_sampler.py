import torch
import random
from utilis import *
import torch.nn.functional as F


class LossCompute:
    def __init__(self, args, S=None , SM=None, D= None):

        self.device = args.device
        self.bce = torch.nn.BCELoss()
        self.l2_weight = args.l2_weight
        self.args = args
        self.S = S
        self.SM = SM
        self.max_alpha = 0.7
        self.D = D

    def gan_d_loss(self, scores_fake, scores_real, agent_idx):

        y_real = torch.ones_like(scores_real) * random.uniform(1, 1.0)
        loss_real = self.bce(scores_real, y_real)
        y_fake = torch.empty_like(scores_fake).uniform_(0.0, 0)
        loss_fake = self.bce(scores_fake, y_fake)

        return loss_real, loss_fake

    def gan_g_loss(self, scores_fake):
        y_fake = torch.ones_like(scores_fake) * random.uniform(1, 1.0)
        #Unlike the discriminator's gan_d_loss, the generator wants scores_fake to be close to 1.0
        return self.bce(scores_fake, y_fake)

    def compute_discriminator_loss(self, agents_targs, agents_idx,error_tolerance,future,past, edge_weights_past, edge_features_past,
                                   direction_past, velocity_past, visability_mat_past,prediction):

        B, N, T, C =past.shape


        alpha = torch.rand(1).item() * self.max_alpha
        scores_real = self.D(torch.cat([past, future.view(B ,N, self.args.future_length, 2)], dim = 2))

        eq_in, scores = self.SM(alpha, agents_idx, agents_targs,error_tolerance,
                                                               past, visability_mat_past,velocity_past,
                                                               direction_past,  edge_features_past, edge_weights_past, prediction)


        if self.args.classifier_method == 'sampler_selected':
            scores = scores.argmax(dim=-1).detach()
            indices_expanded = scores.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1,  self.args.future_length, 2, 1)
            selected = torch.gather(prediction, dim=4, index=indices_expanded).squeeze(-1).detach()
            agents_traj = selected
        elif self.args.classifier_method == 'sampler_predicted':
            agents_traj = eq_in.detach()

        scores_fake = self.D(torch.cat([past, agents_traj.view(B ,N, self.args.future_length, 2)], dim = 2))


        loss_real, loss_fake = self.gan_d_loss(scores_fake, scores_real, agents_idx)  # BCEloss
        total_loss = loss_real + loss_fake

        if agents_idx.numel() != 0:
            scores_fake_final = scores_fake[:, agents_idx]
            uncontrolled_idx = [i for i in range(N) if i not in agents_idx.tolist()]
            if len(uncontrolled_idx) > 0:
                scores_uncontrolled_final = scores_fake[:, uncontrolled_idx]
            else:
                scores_uncontrolled_final = torch.empty(scores_fake.shape[0], 0, device=scores_fake.device)
            return total_loss, loss_real.item(), loss_fake.item(), scores_fake_final, scores_real, scores_uncontrolled_final
        else:
            scores_fake_final = scores_fake
            scores_uncontrolled_final = torch.empty(scores_fake.shape[0], 0, device=scores_fake.device)
            return total_loss, loss_real.item(), loss_fake.item(), scores_fake_final, scores_real, scores_uncontrolled_final

    def compute_generator_loss(self, epoch, agents_targs,agents_idx,error_tolerance,future,past, edge_weights_past, edge_features_past,
                               direction_past, velocity_past,visability_mat_past,prediction, indices):

        alpha = torch.rand(1).item() * self.max_alpha
        B, N, _, _ =past.shape

        eq_in, scores = self.SM(alpha, agents_idx, agents_targs,error_tolerance,
                                                                past, visability_mat_past,velocity_past,
                                                               direction_past,  edge_features_past, edge_weights_past, prediction)


        prediction_for_indexes = prediction.permute(4, 0,1,2, 3).view(20, B,N, self.args.future_length, 2)#(B, N, 10, 2, 20) -> 20, B, N, T, 2

        if agents_idx.numel() != 0:
            prediction_for_indexes = prediction_for_indexes[:, :, agents_idx, :, :]

            best_k = self.calculate_closest_index(prediction_for_indexes, agents_targs) #B, C
            scores_controlled = scores[:, agents_idx, :]
            not_controlled = [i for i in range(N) if i not in set(agents_idx)]
            scores_uncontrolled = scores[:, not_controlled, :]

            loss_score_c = F.cross_entropy(scores_controlled.view(-1, 20), best_k.view(-1).long())
            loss_score_u = F.cross_entropy(scores_uncontrolled.view(-1, 20), indices[:,not_controlled].view(-1).long())
            loss_score = loss_score_c + loss_score_u
        else:
            loss_score = F.cross_entropy(scores.view(-1, 20), indices.view(-1).long())

        if self.args.classifier_method == 'sampler_selected':
            scores_max = scores.argmax(dim=-1)
            indices_expanded = scores_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.args.future_length, 2, 1)
            selected = torch.gather(prediction, dim=4, index=indices_expanded).squeeze(-1)
            agents_traj = selected

        elif self.args.classifier_method == 'sampler_predicted':
            agents_traj = eq_in

        pred_eqin_loss = self.calculate_loss_pred(eq_in, future, B)
        mission_loss = self.calcualte_mission_loss(agents_traj, future, agents_targs, agents_idx, epoch, B,
                               N, error_tolerance, pred_eqin_loss)

        scores_fake = self.D(torch.cat([past, agents_traj.view(B ,N, self.args.future_length, 2)], dim = 2))
        discriminator_loss = self.gan_g_loss(scores_fake)


        total_loss = (
                pred_eqin_loss +
                10*mission_loss +
                loss_score +
                discriminator_loss )

        return (total_loss,
                pred_eqin_loss.item(),
                loss_score.item(),
                discriminator_loss.item(),
                10*mission_loss.item())

    def compute_sampler_loss(self, epoch, mid_epoch, future, past, edge_weights_past,
                                         edge_features_past, direction_past, velocity_past, visability_mat_past,
                                         prediction, indices):

        eq_in, scores = self.S( past, visability_mat_past, velocity_past,
                                                         direction_past, edge_features_past, edge_weights_past,
                                                         prediction)
        B, N, T, C = future.shape

        pred_loss = self.calculate_loss_pred(eq_in.view(B*N, T, 2), future.view(B*N, T, 2), B)
        loss_score = F.cross_entropy(scores.view(-1, 20), indices.view(-1).long())

        # probs = F.softmax(logits, dim=1)
        #
        # entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        #
        # loss = ce_loss - entropy_weight * entropy

        # alpha = self.max_alpha / (1 + math.exp(-self.ramp_k * (epoch - mid_epoch)))
        # alpha = min(self.max_alpha, alpha)
        alpha = 1
        total_loss = pred_loss + loss_score

        return total_loss, pred_loss, loss_score

    def compute_mission_loss(self, epoch, agents_targs,agents_idx,error_tolerance,future,past, edge_weights_past, edge_features_past, direction_past, velocity_past,
                               visability_mat_past,prediction, indices):


        alpha = torch.rand(1).item() * self.max_alpha
        B, N, _, _ =past.shape

        eq_in, scores = self.SM(alpha, agents_idx, agents_targs,error_tolerance,
                                                                past, visability_mat_past,velocity_past,
                                                               direction_past,  edge_features_past, edge_weights_past, prediction)


        prediction_for_indexes = prediction.permute(4, 0,1,2, 3).view(20, B,N, self.args.future_length, 2)#(B, N, 10, 2, 20) -> 20, B, N, T, 2

        if agents_idx.numel() != 0:
            prediction_for_indexes = prediction_for_indexes[:, :, agents_idx, :, :]

            best_k = self.calculate_closest_index(prediction_for_indexes, agents_targs) #B, C
            scores_controlled = scores[:, agents_idx, :]
            not_controlled = [i for i in range(N) if i not in set(agents_idx)]
            scores_uncontrolled = scores[:, not_controlled, :]

            loss_score_c = F.cross_entropy(scores_controlled.view(-1, 20), best_k.view(-1).long())
            loss_score_u = F.cross_entropy(scores_uncontrolled.view(-1, 20), indices[:,not_controlled].view(-1).long())
            loss_score = loss_score_c + loss_score_u
        else:
            loss_score = F.cross_entropy(scores.view(-1, 20), indices.view(-1).long())


        if self.args.classifier_method == 'sampler_selected':
            scores_max = scores.argmax(dim=-1)
            indices_expanded = scores_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.args.future_length, 2, 1)
            selected = torch.gather(prediction, dim=4, index=indices_expanded).squeeze(-1)
            agents_traj = selected

        elif self.args.classifier_method == 'sampler_predicted':
            agents_traj = eq_in

        pred_eqin_loss = self.calculate_loss_pred(eq_in, future, B)

        mission_loss = self.calcualte_mission_loss(agents_traj, future, agents_targs, agents_idx, epoch, B,
                               N, error_tolerance, pred_eqin_loss)

        total_loss = (
                pred_eqin_loss +
                mission_loss +
                loss_score
                 )

        return (total_loss,
                pred_eqin_loss,
                loss_score,
                mission_loss)

    def calculate_loss_pred(self, pred, target, B):
        batch_size = B
        loss = (target - pred).pow(2).sum()
        loss /= batch_size
        loss /= pred.shape[1]
        return loss


    def calculate_closest_index(self, prediction_for_indexes, controlled_targets):
        p1 = prediction_for_indexes[:, :, :, :-1, :]  # (20, B, C, T-1, 2)
        p2 = prediction_for_indexes[:, :, :, 1:, :]  # (20, B, C, T-1, 2)

        m = controlled_targets.unsqueeze(0).unsqueeze(3)  # (1, B, C, 1, 2)

        seg = p2 - p1  # (20, B, C, T-1, 2)
        seg_len = seg.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (20, B, C, T-1, 1)
        proj = ((m - p1) * seg).sum(-1, keepdim=True) / (seg_len ** 2)  # (20, B, C, T-1, 1)
        proj_cl = proj.clamp(0, 1)  # (20, B, C, T-1, 1)

        #closest points and distances
        closest = p1 + proj_cl * seg  # (20, B, C, T-1, 2)
        d = (closest - m).norm(dim=-1)  # (20, B, C, T-1)

        #minimum distance over each trajectory
        min_d, _ = d.min(dim=-1)  # (20, B, C)
        #pick best option k per (b,c)
        best_k = min_d.argmin(dim=0)  # (B, C)
        return best_k


    def calcualte_mission_loss(self, pred_traj, future_traj, controlled_targets, agents_idx, epoch, batch_size, agent_num, error_tolerance,loss_pred):

        if agents_idx.numel() == 0:
            loss_mission_div = torch.tensor(0, dtype=torch.float32, device=pred_traj.device)
            return loss_mission_div

        else:
            B = batch_size
            N = agent_num
            C = len(agents_idx)
            T = self.args.future_length

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
            within_reach = (start_to_target <= self.args.how_far*gt_movement)
            # very_far = ~within_reach

            start_dist = torch.norm(pred_controlled[:, :, 0, :] - controlled_targets, dim=-1)
            end_dist = torch.norm(pred_controlled[:, :, -1, :] - controlled_targets, dim=-1)
            direction_delta = end_dist - start_dist
            direction_sign = direction_delta.sign()  # +1 bad, -1 good
            abs_delta = direction_delta.abs().clamp(min=0.003)
            positive_branch = abs_delta
            negative_branch = 0.1 / abs_delta

            direction_loss = (direction_sign < 0).float() * negative_branch + \
                             (direction_sign >= 0).float() * positive_branch

            mission_loss = torch.zeros(B, C, device=pred_traj.device)
            mask_not_achieved = ~achieved
            mask_direction_bad = direction_sign > 0
            mask_direction_good = direction_sign <= 0

            mask_4 = mask_not_achieved & within_reach & mask_direction_good
            mission_loss[mask_4] =  min_distance[mask_4] + 0.2 * direction_loss[mask_4]
            #  not achieved & within reach & opposite direction --> heavier combined loss
            mask_5 = mask_not_achieved & within_reach & mask_direction_bad
            mission_loss[mask_5] = min_distance[mask_5] + 2.5 * direction_loss[mask_5]

            # mask_4 = mask_not_achieved & ~within_reach & mask_direction_good
            # mission_loss[mask_4] =   0.1 * direction_loss[mask_4]
            # mask_5 = mask_not_achieved & ~within_reach & mask_direction_bad
            # mission_loss[mask_5] = 0.2 * direction_loss[mask_5]
            #
            # mask_6 = mask_not_achieved & within_reach & mask_direction_good
            # mission_loss[mask_6] =   min_distance[mask_6]
            #
            # mask_7 = mask_not_achieved & within_reach & mask_direction_bad
            # mission_loss[mask_7] =   min_distance[mask_7] + 0.1 * direction_loss[mask_7]


            mission_weight = min(2.0, epoch / 5)
            mission_loss_final = (mission_weight * mission_loss).mean()

            if mission_loss_final > loss_pred and not (mask_4.any() or mask_5.any()):
                scaling = (loss_pred / mission_loss_final).detach() * 0.95  # keep it just under
                mission_loss_final = mission_loss_final * scaling

            return mission_loss_final
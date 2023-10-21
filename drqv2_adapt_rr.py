import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.conv1 = nn.Conv2d(obs_shape[0], 32, 3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1)
        self.relu4 = nn.ReLU()

        self.apply(utils.weight_init)

    def forward_with_FAU(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.relu1(self.conv1(obs))
        act_1 = (h > 0).sum().item()
        total_1 = h.numel()
        h = self.relu2(self.conv2(h))
        act_2 = (h > 0).sum().item()
        total_2 = h.numel()
        h = self.relu3(self.conv3(h))
        act_3 = (h > 0).sum().item()
        total_3 = h.numel()
        h = self.relu4(self.conv4(h))
        act_4 = (h > 0).sum().item()
        total_4 = h.numel()
        h = h.view(h.shape[0], -1)
        rate = (act_1+act_2+act_3+act_4)/(total_1+total_2+total_3+total_4)
        return h, rate

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.relu1(self.conv1(obs))
        h = self.relu2(self.conv2(h))
        h = self.relu3(self.conv3(h))
        h = self.relu4(self.conv4(h))
        h = h.view(h.shape[0], -1)
        return h

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward_with_FAU(self, obs, std):
        h_trunk = self.trunk[0](obs)
        h_trunk = self.trunk[1](h_trunk)
        h = self.trunk[2](h_trunk)

        h_policy = self.policy[:2](h)
        act_1 = (h_policy > 0).sum().item()
        total_1 = h_policy.numel()

        h_policy = self.policy[2:4](h_policy)
        act_2 = (h_policy > 0).sum().item()
        total_2 = h_policy.numel()

        mu = self.policy[4](h_policy)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        rate = (act_1 + act_2) / (total_1 + total_2)

        dist = utils.TruncatedNormal(mu, std)
        return dist, rate  

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward_with_FAU(self, obs, action):
        h_trunk = self.trunk[0](obs)
        h_trunk = self.trunk[1](h_trunk)
        h = self.trunk[2](h_trunk)
        
        h_action = torch.cat([h, action], dim=-1)
        
        h_Q1 = self.Q1[:2](h_action)
        act_1 = (h_Q1 > 0).sum().item()
        total_1 = h_Q1.numel()
        
        h_Q1 = self.Q1[2:4](h_Q1)
        act_2 = (h_Q1 > 0).sum().item()
        total_2 = h_Q1.numel()
        
        q1 = self.Q1[4](h_Q1)
        
        h_Q2 = self.Q2[:2](h_action)
        act_3 = (h_Q2 > 0).sum().item()
        total_3 = h_Q2.numel()
        
        h_Q2 = self.Q2[2:4](h_Q2)
        act_4 = (h_Q2 > 0).sum().item()
        total_4 = h_Q2.numel()
        
        q2 = self.Q2[4](h_Q2)
        
        rate = (act_1 + act_2 + act_3 + act_4) / (total_1 + total_2 + total_3 + total_4)
        
        return q1, q2, rate

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
    

class DrQV2_adapt_rr_Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps, use_aug, batch_size,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.batch_size = batch_size

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.use_aug = use_aug
        ## for adapt Replay Ratio
        self.cal_init_Phi = False
        self.init_Phi = 0.0
        self.last_average_Phi = 0.0
        self.average_Phi = 0.0
        self.sum_Phi = 0.0
        self.sum_Phi_count = int(0)
        self.replay_ratio = 1

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2, FAU_critic = self.critic.forward_with_FAU(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        self.sum_Phi += FAU_critic
        self.sum_Phi_count += 1
        ###
        if self.cal_init_Phi == False:
            self.init_Phi = FAU_critic
            self.last_average_Phi = self.init_Phi
            print('init Phi:',self.init_Phi)
            self.cal_init_Phi = True

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['FAU_Critic'] = FAU_critic

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist, FAU_actor = self.actor.forward_with_FAU(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics['FAU_Actor'] = FAU_actor

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        if self.use_aug:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
        else:
            obs = obs.float()
            next_obs = next_obs.float() 

        # encode
        obs, FAU_encoder = self.encoder.forward_with_FAU(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
            metrics['FAU_Encoder'] = FAU_encoder

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

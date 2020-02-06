# import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import environments.environments

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
import quadpy
import itertools

from scipy.special import binom


class ReverseKLNetwork(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(ReverseKLNetwork, self).__init__(sess, config, [config.pi_lr, config.qf_vf_lr])

        self.config = config

        self.optim_type = config.optim_type

        self.use_true_q = False
        if config.use_true_q == "True":
            self.use_true_q = True

        self.rng = np.random.RandomState(config.random_seed)

        self.actor_l1_dim = config.actor_l1_dim
        self.actor_l2_dim = config.actor_l2_dim
        self.critic_l1_dim = config.critic_l1_dim
        self.critic_l2_dim = config.critic_l2_dim

        self.input_norm = input_norm
        self.entropy_scale = config.entropy_scale

        # create network
        self.pi_net = PolicyNetwork(self.state_dim, self.action_dim, config.actor_l1_dim, config.actor_l2_dim, self.action_max[0])
        self.q_net = SoftQNetwork(self.state_dim, self.action_dim, config.critic_l1_dim, config.critic_l2_dim)

        self.v_net = ValueNetwork(self.state_dim, config.critic_l1_dim, config.critic_l2_dim)
        self.target_v_net = ValueNetwork(self.state_dim, config.critic_l1_dim, config.critic_l2_dim)

        # copy to target_v_net
        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(param.data)

        self.device = torch.device("cpu")

        # optimizer
        self.pi_optimizer = optim.Adam(self.pi_net.parameters(), lr=self.learning_rate[0])
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate[1])
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=self.learning_rate[1])

        dtype = torch.float32

        if self.action_dim == 1:
            self.N = config.N_param  # 1024

            scheme = quadpy.line_segment.clenshaw_curtis(self.N)
            # cut off endpoints since they should be zero but numerically might give nans
            # self.intgrl_actions = torch.tensor(scheme.points[1:-1], dtype=dtype).unsqueeze(-1) * self.action_max
            self.intgrl_actions = (torch.tensor(scheme.points[1:-1], dtype=dtype).unsqueeze(-1) * self.action_max).to(
                torch.float32)
            self.intgrl_weights = torch.tensor(scheme.weights[1:-1], dtype=dtype)

            self.intgrl_actions_len = np.shape(self.intgrl_actions)[0]

        else:
            self.l = config.l_param  # 5

            n_points = [1]
            for i in range(1, self.l):
                n_points.append(2 ** i + 1)

            schemes = [quadpy.line_segment.clenshaw_curtis(n_points[i]) for i in range(1, self.l)]
            points = [np.array([0.])] + [scheme.points[1:-1] for scheme in schemes]
            weights = [np.array([2.])] + [scheme.weights[1:-1] for scheme in schemes]

            # precalculate actions and weights
            self.intgrl_actions = []
            self.intgrl_weights = []

            for k in itertools.product(range(self.l), repeat=self.action_dim):
                if (np.sum(k) + self.action_dim < self.l) or (
                        np.sum(k) + self.action_dim > self.l + self.action_dim - 1):
                    continue
                coeff = (-1) ** (self.l + self.action_dim - np.sum(k) - self.action_dim + 1) * binom(
                    self.action_dim - 1, np.sum(k) + self.action_dim - self.l)

                for j in itertools.product(*[range(len(points[ki])) for ki in k]):
                    self.intgrl_actions.append(
                        torch.tensor([points[k[i]][j[i]] for i in range(self.action_dim)], dtype=dtype))
                    self.intgrl_weights.append(
                        coeff * np.prod([weights[k[i]][j[i]].squeeze() for i in range(self.action_dim)]))
            self.intgrl_weights = torch.tensor(self.intgrl_weights, dtype=dtype)
            self.intgrl_actions = torch.stack(self.intgrl_actions) * self.action_max

            self.intgrl_actions_len = np.shape(self.intgrl_actions)[0]

    def sample_action(self, state_batch):

        state_batch = torch.FloatTensor(state_batch).to(self.device)

        action, log_prob, z, mean, log_std = self.pi_net.evaluate(state_batch)

        # print("sample action: {}".format(action.detach().numpy()))
        return action.detach().numpy()

    def predict_action(self, state_batch):

        state_batch = torch.FloatTensor(state_batch).to(self.device)

        # mean, log_std = self.pi_net(state_batch)
        _, _, _, mean, log_std = self.pi_net.evaluate(state_batch)
        # print("predict action: {}".format(mean.detach().numpy()))

        return mean.detach().numpy()

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        gamma_batch = torch.FloatTensor(gamma_batch).to(self.device)

        reward_batch.unsqueeze_(-1)
        gamma_batch.unsqueeze_(-1)
        q_val = self.q_net(state_batch, action_batch)
        v_val = self.v_net(state_batch)
        new_action, log_prob, z, mean, log_std = self.pi_net.evaluate(state_batch)

        # q_loss, v_loss
        target_next_v_val = self.target_v_net(next_state_batch)
        target_q_val = reward_batch + gamma_batch * target_next_v_val
        q_value_loss = nn.MSELoss()(q_val, target_q_val.detach())

        new_q_val = self.q_net(state_batch, new_action)
        target_v_val = new_q_val - self.entropy_scale * log_prob
        value_loss = nn.MSELoss()(v_val, target_v_val.detach())

        # pi_loss

        if self.optim_type == 'll':
            log_prob_target = new_q_val # - v_val

            # loglikelihood update
            policy_loss = (-log_prob * (log_prob_target - self.entropy_scale * log_prob).detach()).mean()

        elif self.optim_type == 'reparam':
            # policy_loss = - (expected_new_q_val - self.entropy_scale * log_prob).mean()
            pass

        elif self.optim_type == 'intg':
            stacked_state_batch = state_batch.unsqueeze(1).repeat(1, self.intgrl_actions_len, 1).reshape(-1, self.state_dim)

            tiled_intgrl_actions = self.intgrl_actions.unsqueeze(0).repeat(self.config.batch_size, 1, 1)
            stacked_intgrl_actions = tiled_intgrl_actions.reshape(-1, self.action_dim)  # (32 x 254, 1)

            intgrl_q_val = self.q_net(stacked_state_batch, stacked_intgrl_actions)
            # intgrl_v_val = v_val.unsqueeze(1).repeat(1, self.intgrl_actions_len, 1).reshape(-1, 1)

            # intgrl_logprob = self.pi_net.get_logprob(stacked_state_batch, stacked_intgrl_actions)
            intgrl_logprob = self.pi_net.get_logprob(state_batch, tiled_intgrl_actions)

            # integrands = - torch.exp(intgrl_logprob.squeeze()) * ((intgrl_q_val.squeeze() - intgrl_v_val.squeeze()).detach() - intgrl_logprob.squeeze())
            integrands = - torch.exp(intgrl_logprob.squeeze()) * ((intgrl_q_val.squeeze()).detach() - intgrl_logprob.squeeze())

            policy_loss = (integrands * self.intgrl_weights.repeat(self.config.batch_size)).reshape(self.config.batch_size, -1).sum(-1).mean(-1)

        elif self.optim_type == 'hard_intg':
            stacked_state_batch = state_batch.unsqueeze(1).repeat(1, self.intgrl_actions_len, 1).reshape(-1, self.state_dim)
            tiled_intgrl_actions = self.intgrl_actions.unsqueeze(0).repeat(self.config.batch_size, 1, 1)
            stacked_intgrl_actions = tiled_intgrl_actions.reshape(-1, self.action_dim)  # (32 x 254, 1)

            intgrl_logprob = self.pi_net.get_logprob(state_batch, tiled_intgrl_actions)
            intgrl_q_val = self.q_net(stacked_state_batch, stacked_intgrl_actions)
            # intgrl_v_val = v_val.unsqueeze(1).repeat(1, self.intgrl_actions_len, 1).reshape(-1, 1)

            # integrands = -torch.exp(intgrl_logprob.squeeze()) * (intgrl_q_val.squeeze() - intgrl_v_val.squeeze()).detach()
            integrands = -torch.exp(intgrl_logprob.squeeze()) * (intgrl_q_val.squeeze()).detach()
            policy_loss = (integrands * self.intgrl_weights.repeat(self.config.batch_size)).reshape(self.config.batch_size, -1).sum(-1).mean(-1)
        elif self.optim_type == 'hard_ll':

            pass


        self.q_optimizer.zero_grad()
        q_value_loss.backward()
        self.q_optimizer.step()

        self.v_optimizer.zero_grad()
        value_loss.backward()
        self.v_optimizer.step()

        self.pi_optimizer.zero_grad()
        policy_loss.backward()
        self.pi_optimizer.step()

    def update_target_network(self):
        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def getQFunction(self, state):
        return lambda action: (self.q_net(torch.FloatTensor(state).to(self.device).unsqueeze(-1),
                                         torch.FloatTensor([action]).to(self.device).unsqueeze(-1))).detach().numpy()

    def getPolicyFunction(self, state):

        # mean, logstd = self.pi_net(torch.FloatTensor(state).to(self.device).unsqueeze(-1))
        _, _, _, mean, log_std = self.pi_net.evaluate(torch.FloatTensor(state).to(self.device).unsqueeze(-1))
        mean = mean.detach().numpy()
        std = (log_std.exp()).detach().numpy()
        return lambda action: 1/(std * np.sqrt(2 * np.pi)) * np.exp(- (action - mean)**2 / (2 * std**2))

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, l1_dim, l2_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, l1_dim)
        self.linear2 = nn.Linear(l1_dim, l2_dim)
        self.linear3 = nn.Linear(l2_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.device = torch.device("cpu")

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, l1_dim, l2_dim, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, l1_dim)
        self.linear2 = nn.Linear(l1_dim, l2_dim)
        self.linear3 = nn.Linear(l2_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.device = torch.device("cpu")

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, l1_dim, l2_dim, action_scale, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, l1_dim)
        self.linear2 = nn.Linear(l1_dim, l2_dim)

        self.mean_linear = nn.Linear(l2_dim, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(l2_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = torch.device("cpu")

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = self.get_distribution(mean, std)

        z = normal.sample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z)

        if len(log_prob.shape) == 1:
            log_prob.unsqueeze_(-1)
        log_prob -= torch.log(1 - action.pow(2) + epsilon).sum(-1, keepdim=True)

        # scale to correct range
        action *= self.action_scale

        mean = torch.tanh(mean)
        mean *= self.action_scale

        return action, log_prob, z, mean, log_std


    def get_logprob(self, states, tiled_actions, epsilon=1e-6):

        # states; (32, 3)
        # tiled_actions: (32, 254, 1)

        normalized_actions = tiled_actions.permute(1, 0, 2) / self.action_scale  # (254, 32, 1)
        atanh_actions = self.atanh(normalized_actions)  # (254, 32, 1)

        mean, log_std = self.forward(states)
        std = log_std.exp()

        normal = self.get_distribution(mean, std)

        log_prob = normal.log_prob(atanh_actions)

        if len(log_prob.shape) == 2:
            log_prob.unsqueeze_(-1)

        log_prob -= torch.log(1 - normalized_actions.pow(2) + epsilon).sum(dim=-1,keepdim=True)
        stacked_log_prob = log_prob.permute(1, 0, 2).reshape(-1, 1)

        return stacked_log_prob

    def get_distribution(self, mean, std):
        if self.action_dim == 1:
            normal = Normal(mean, std)
        else:
            normal = MultivariateNormal(mean, torch.diag_embed(std))

        return normal



    def atanh(self, x):
        return (torch.log(1 + x) - torch.log(1 - x)) / 2

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]

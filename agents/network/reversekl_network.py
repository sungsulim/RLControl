# import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import environments.environments

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class ReverseKLNetwork(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(ReverseKLNetwork, self).__init__(sess, config, [config.pi_lr, config.qf_vf_lr])

        self.config = config

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
        expected_q_val = self.q_net(state_batch, action_batch)
        expected_v_val = self.v_net(state_batch)
        new_action, log_prob, z, mean, log_std = self.pi_net.evaluate(state_batch)

        target_value = self.target_v_net(next_state_batch)
        next_q_value = reward_batch + gamma_batch * target_value
        q_value_loss = nn.MSELoss()(expected_q_val, next_q_value.detach())

        expected_new_q_val = self.q_net(state_batch, new_action)
        next_value = expected_new_q_val - self.entropy_scale * log_prob
        value_loss = nn.MSELoss()(expected_v_val, next_value.detach())

        log_prob_target = expected_new_q_val - expected_v_val

        # loglikelihood update
        policy_loss = (-log_prob * (log_prob_target - self.entropy_scale * log_prob).detach()).mean()

        # reparam update
        # policy_loss = - (expected_new_q_val - self.entropy_scale * log_prob).mean()

        # mean_lambda = 1  # 1e-3
        # std_lambda = 1  # 1e-3
        # z_lambda = 0.0

        # mean_loss = mean_lambda * mean.pow(2).mean()
        # std_loss = std_lambda * log_std.pow(2).mean()
        # z_loss = z_lambda * z.pow(2).sum(1).mean()

        # policy_loss += mean_loss + std_loss + z_loss

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

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        # scale to correct range
        action *= self.action_scale

        mean = torch.tanh(mean)
        mean *= self.action_scale

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]
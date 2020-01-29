# import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import environments.environments

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import quadpy
import itertools

class ForwardKLNetwork(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(ForwardKLNetwork, self).__init__(sess, config, [config.pi_lr, config.qf_vf_lr])

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



        #############
        dtype = torch.float

        self.l = 3  # 5
        self.N = 256  # 1024

        self.scheme = quadpy.line_segment.clenshaw_curtis(self.N)
        if self.action_dim == 1:
            # cut off endpoints since they should be zero but numerically might give nans
            self.intgrl_actions = torch.tensor(self.scheme.points[1:-1], dtype=dtype).unsqueeze(-1)
            self.intgrl_weights = torch.tensor(self.scheme.weights[1:-1], dtype=dtype)
        else:
            raise NotImplementedError

            # n_points = [1]
            # for i in range(1, self.l):
            #     n_points.append(2 ** i + 1)
            # # print(n_points)
            # schemes = [quadpy.line_segment.clenshaw_curtis(n_points[i]) for i in range(1, self.l)]
            # points = [np.array([0.])] + [scheme.points[1:-1] for scheme in schemes]
            # # print(points)
            # weights = [np.array([2.])] + [scheme.weights[1:-1] for scheme in schemes]
            # # precalculate actions and weights
            # self.intgrl_actions = []
            # self.intgrl_weights = []
            # for k in itertools.product(range(self.l), repeat=self.env.action_dim):
            #     if (np.sum(k) + self.env.action_dim < self.l) or (
            #             np.sum(k) + self.env.action_dim > self.l + self.env.action_dim - 1):
            #         continue
            #     coeff = (-1) ** (self.l + self.env.action_dim - np.sum(k) - self.env.action_dim + 1) * binom(
            #         self.env.action_dim - 1, np.sum(k) + self.env.action_dim - self.l)
            #     for j in itertools.product(*[range(len(points[ki])) for ki in k]):
            #         self.intgrl_actions.append(
            #             torch.tensor([points[k[i]][j[i]] for i in range(self.env.action_dim)], dtype=dtype))
            #         self.intgrl_weights.append(
            #             coeff * np.prod([weights[k[i]][j[i]].squeeze() for i in range(self.env.action_dim)]))
            # self.intgrl_weights = torch.tensor(self.intgrl_weights, dtype=dtype)
            # self.intgrl_actions = torch.stack(self.intgrl_actions)

        # self.name = "CCAllActions_alr={}_clr={}_hidden={}_N={}".format(self.actor_lr, self.critic_lr, self.n_hidden,
        #                                                                len(self.intgrl_actions))

    def get_pdf(self, s, a):
        # s is a single state
        # a could be a batch of actions
        mean, std = self.policy(s)
        m = Normal(loc=mean, scale=std)
        return torch.exp(m.log_prob(a).sum(dim=-1))

    def get_tanh_pdf(self, s, a):
        # mean, std = self.policy(s)
        # assuming a is already tanh transformed
        pdf = self.get_pdf(s, self.atanh(a)) / ((1 - a.pow(2)).prod(dim = -1))
        return pdf

    def step(self, s, a, r, sp, done):
        # update the critic
        v_loss = torch.pow(r + (1. - done) * self.gamma * self.vcritic(sp).detach() - self.vcritic(s), 2)
        # print(s.shape)
        q_loss = torch.pow(r + (1. - done) * self.gamma * self.vcritic(sp).detach() - self.qcritic(torch.cat([s.squeeze(), torch.tensor(a, dtype = dtype)], dim = -1)), 2)
        pdf = self.get_tanh_pdf(s, self.actions)
        # q_arg = torch.cat([s.squeeze().repeat(len(self.actions), 1), self.actions.unsqueeze(-1)], dim = -1) # this type of broadcasting only makes sense with 1D actions
        # print(pdf.shape, self.actions.shape)
        q_arg = torch.cat([s.squeeze().repeat(self.actions.shape[0], 1), self.actions], dim = -1)
        q_value = self.qcritic(q_arg).squeeze()
        # print(q_value.shape, pdf.shape, q_arg.shape)
        integrands = -pdf.squeeze() * ((q_value.squeeze() - self.vcritic(s)).detach())
        # print(integrands.shape)
        policy_loss = (integrands.squeeze() * self.weights.squeeze()).sum()
        self.vcritic_optim.zero_grad()
        self.qcritic_optim.zero_grad()
        self.policy_optim.zero_grad()
        (policy_loss + v_loss + q_loss).backward()
        self.vcritic_optim.step()
        self.qcritic_optim.step()
        self.policy_optim.step()

    def atanh(self, x):
        return (torch.log(1 + x) - torch.log(1 - x)) / 2

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
        stacked_state_batch = state_batch.unsqueeze(0).repeat(self.intgrl_actions.shape[0], 1, 1).permute(1,0,2).reshape(-1, self.state_dim)  # (254 x 32, 3)
        stacked_intgrl_actions = self.intgrl_actions.unsqueeze(0).repeat(self.config.batch_size, 1, 1).reshape(-1, self.action_dim)  # (254 x 32, 1)

        intgrl_q_val = self.q_net(stacked_state_batch, stacked_intgrl_actions)


        log_prob_target = new_q_val - v_val

        # loglikelihood update
        policy_loss = (-log_prob * (log_prob_target - self.entropy_scale * log_prob).detach()).mean()

        # reparam update
        # policy_loss = - (expected_new_q_val - self.entropy_scale * log_prob).mean()

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

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon).sum(-1, keepdim=True)
        # log_prob = log_prob.sum(-1, keepdim=True)

        # scale to correct range
        action *= self.action_scale

        mean = torch.tanh(mean)
        mean *= self.action_scale

        return action, log_prob, z, mean, log_std

    def get_logprob(self, stacked_states, stacked_actions, epsilon=1e-6):

        # assuming actions is already tanh transformed
        # states and actions should have same batch size

        # states: (254 * 32, 3)
        # actions: (254 * 32, 1)

        mean, log_std = self.forward(stacked_states)
        std = log_std.exp()

        normal = Normal(mean, std)

        normalized_actions = stacked_actions / self.action_scale
        atanh_actions = self.atanh(normalized_actions)

        # pdf = torch.exp(normal.log_prob(atanh_actions)) / ((1 - normalized_actions.pow(2)).prod(dim=-1))
        log_prob = normal.log_prob(atanh_actions) - torch.log(1 - normalized_actions.pow(2) + epsilon).sum(dim=-1)


        return log_prob

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

import numpy as np
from random import randint
import random
import datetime
import tensorflow as tf

from utils.replaybuffer import ReplayBuffer

# Agent interface
# Takes an environment (just so we can get some details from the environment like the number of observables and actions)
class BaseAgent(object):
    def __init__(self, env, config):
        self.state_dim = env.state_dim
        self.state_min = env.state_min
        self.state_max = env.state_max

        self.action_dim = env.action_dim
        self.action_min = env.action_min
        self.action_max = env.action_max

        self.seed = None

        if config.exploration_policy == 'ou_noise':
            from utils.exploration_policy import OrnsteinUhlenbeckProcess
            self.exploration_policy = OrnsteinUhlenbeckProcess(self.action_dim, self.action_min, self.action_max,
                                                             theta = config.ou_theta,
                                                             mu = config.ou_mu,
                                                             sigma = config.ou_sigma)

        elif config.exploration_policy == 'epsilon_greedy':
            from utils.exploration_policy import EpsilonGreedy
            self.exploration_policy = EpsilonGreedy(self.action_min, self.action_max, config.annealing_steps,
                                                    config.min_epsilon, config.max_epsilon, 
                                                    is_continuous=True)

        elif config.exploration_policy == 'random_uniform':
            from utils.exploration_policy import RandomUniform
            self.exploration_policy = RandomUniform(self.action_min, self.action_max, is_continuous=True)

        elif config.exploration_policy == 'none':
            self.exploration_policy = None

        else:
            raise NotImplementedError

        self.replay_buffer = ReplayBuffer(config.buffer_size)    
        self.batch_size = config.batch_size
        self.warmup_steps = config.warmup_steps
        self.gamma = config.gamma 


    # This should follow the agent's policy and return an action that will be used by the environment
    def step(self, obs):
        return 0

    def getValue(self, s, a):
        raise NotImplementedError
    
    # This should update the agent's parameters using S, A, R, Sp tuple received from environment
    def update(self, obs, obs_n, r, a):
        raise NotImplementedError


    # Sets the random seed for the agent
    def setSeed(self, seed):
        raise NotImplementedError

    # Resets the agent between episodes. Should primarily be used to clear traces or other temporally linked parameters
    def reset(self):
        return None

    # This should return a dictionary with the meta-parameters used by the agent.
    # Currently this information is used by the logger to make orginization of experiment files easier to navigate
    def params(self):
        raise Exception('Uh oh. The agent should have a params() function that returns a dictionary with all params for instance: agent.params() => {gamma: 0.1, alpha: 0.5}')

    def start(self, obs):
        return self.getAction(obs)

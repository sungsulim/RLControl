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
        # to log useful stuff within agent
        self.write_log = config.write_log
        
        self.state_dim = env.state_dim
        self.state_min = env.state_min
        self.state_max = env.state_max

        self.action_dim = env.action_dim
        self.action_min = env.action_min
        self.action_max = env.action_max

        self.seed = None
        self.network = None

        if config.exploration_policy == 'ou_noise':
            from utils.exploration_policy import OrnsteinUhlenbeckProcess
            self.use_external_exploration = True
            self.exploration_policy = OrnsteinUhlenbeckProcess(self.action_dim, self.action_min, self.action_max,
                                                             theta = config.ou_theta,
                                                             mu = config.ou_mu,
                                                             sigma = config.ou_sigma)

        elif config.exploration_policy == 'epsilon_greedy':
            from utils.exploration_policy import EpsilonGreedy
            self.use_external_exploration = True
            self.exploration_policy = EpsilonGreedy(self.action_min, self.action_max, config.annealing_steps,
                                                    config.min_epsilon, config.max_epsilon, 
                                                    is_continuous=True)

        elif config.exploration_policy == 'random_uniform':
            from utils.exploration_policy import RandomUniform
            self.use_external_exploration = True
            self.exploration_policy = RandomUniform(self.action_min, self.action_max, is_continuous=True)

        elif config.exploration_policy == 'none':
            self.use_external_exploration = False
            self.exploration_policy = None

        else:
            raise NotImplementedError

        self.replay_buffer = ReplayBuffer(config.buffer_size)    
        self.batch_size = config.batch_size
        self.warmup_steps = config.warmup_steps
        self.gamma = config.gamma 

    def start(self, state):
        raise NotImplementedError

    def step(self, state):
        raise NotImplementedError

    def get_value(self, s, a):
        raise NotImplementedError
    
    def update(self, state, next_state, reward, action, is_terminal):
        raise NotImplementedError


    # Sets the random seed for the agent
    def set_seed(self, seed):
        raise NotImplementedError

    # Resets the agent between episodes. Should primarily be used to clear traces or other temporally linked parameters
    def reset(self):
        raise NotImplementedError

    # set writer (to log useful stuff in tensorboard)
    def set_writer(self, writer):
        self.network.writer = writer




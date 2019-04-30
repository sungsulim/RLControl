import tensorflow as tf

from utils.running_mean_std import RunningMeanStd


class BaseNetwork_Manager(object):
    def __init__(self, config):

        self.random_seed = config.random_seed

        # Env config
        self.state_dim = config.state_dim
        self.state_min = config.state_min
        self.state_max = config.state_max

        self.action_dim = config.action_dim
        self.action_min = config.action_min
        self.action_max = config.action_max

        # Log config
        self.write_log = config.write_log
        self.write_plot = config.write_plot
        self.writer = config.writer

        # record step for tf Summary
        self.train_global_steps = 0
        self.eval_global_steps = 0
        self.train_ep_count = 0
        self.eval_ep_count = 0

        self.use_external_exploration = None
        self.exploration_policy = None
        self.set_exploration(config)

        # type of normalization: 'none', 'batch', 'layer', 'input_norm'
        if config.norm_type != 'none':
            self.input_norm = RunningMeanStd(self.state_dim)
        else:
            self.input_norm = None

        self.graph = tf.Graph()

    def set_exploration(self, config):
        if config.exploration_policy == 'ou_noise':
            from utils.exploration_policy import OrnsteinUhlenbeckProcess
            self.use_external_exploration = True
            self.exploration_policy = OrnsteinUhlenbeckProcess(self.random_seed,
                                                               self.action_dim,
                                                               self.action_min,
                                                               self.action_max,
                                                               theta=config.ou_theta,
                                                               mu=config.ou_mu,
                                                               sigma=config.ou_sigma)

        elif config.exploration_policy == 'epsilon_greedy':
            from utils.exploration_policy import EpsilonGreedy
            self.use_external_exploration = True
            self.exploration_policy = EpsilonGreedy(self.random_seed, self.action_min, self.action_max,
                                                    config.annealing_steps, config.min_epsilon, config.max_epsilon,
                                                    is_continuous=True)

        elif config.exploration_policy == 'random_uniform':
            from utils.exploration_policy import RandomUniform
            self.use_external_exploration = True
            self.exploration_policy = RandomUniform(self.random_seed, self.action_min, self.action_max,
                                                    is_continuous=True)

        elif config.exploration_policy == 'none':
            self.use_external_exploration = False
            self.exploration_policy = None

        else:
            raise ValueError("Invalid Value for config.exploration_policy")

    def take_action(self, state, is_train, is_start):
        raise NotImplementedError

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):
        raise NotImplementedError

    def reset(self):

        self.train_ep_count = 0
        self.eval_ep_count = 0

        if self.exploration_policy:
            self.exploration_policy.reset()

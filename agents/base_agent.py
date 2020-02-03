from utils.replaybuffer import ReplayBuffer
import numpy as np


# Agent interface
# Takes an environment (just so we can get some details from the environment like the number of observables and actions)
class BaseAgent(object):
    def __init__(self, config, network_manager):

        self.norm_type = config.norm_type

        # Env config
        self.state_dim = config.state_dim
        self.state_min = config.state_min
        self.state_max = config.state_max

        self.action_dim = config.action_dim
        self.action_min = config.action_min
        self.action_max = config.action_max

        self.replay_buffer = ReplayBuffer(config.buffer_size, config.random_seed)
        self.batch_size = config.batch_size
        self.warmup_steps = config.warmup_steps
        self.gamma = config.gamma

        # to log useful stuff within agent
        self.write_log = config.write_log
        self.write_plot = config.write_plot

        self.network_manager = network_manager
        self.writer = config.writer
        self.config = config

    def start(self, state, is_train):
        return self.take_action(state, is_train, is_start=True)

    def step(self, state, is_train):
        return self.take_action(state, is_train, is_start=False)

    def take_action(self, state, is_train, is_start):
        # Warmup step not really used
        if self.replay_buffer.get_size() < self.warmup_steps:

            # use random seed
            # action = (np.random.random_sample(size=self.action_dim) - 0.5) * 2 * self.action_max[0]
            raise NotImplementedError
        else:
            action = self.network_manager.take_action(state, is_train, is_start)
        return action

    def get_value(self, s, a):
        raise NotImplementedError
    
    def update(self, state, next_state, reward, action, is_terminal, is_truncated):

        # if using replay buffer
        if not is_truncated:

            if not is_terminal:
                self.replay_buffer.add(state, action, reward, next_state, self.gamma)
            else:
                self.replay_buffer.add(state, action, reward, next_state, 0.0)
        if self.norm_type != 'none':
            self.network_manager.input_norm.update(state)
        self.learn()

        # if not using replay buffer
        # if not is_truncated:
        #
        #     state = np.expand_dims(state, 0)
        #     next_state = np.expand_dims(next_state, 0)
        #     action = np.expand_dims(action, 0)
        #     reward = np.expand_dims(reward, 0)
        #
        #     if not is_terminal:
        #         # self.replay_buffer.add(state, action, reward, next_state, self.gamma)
        #         self.network_manager.update_network(state, action, next_state, reward, [self.gamma])
        #     else:
        #         # self.replay_buffer.add(state, action, reward, next_state, 0.0)
        #         self.network_manager.update_network(state, action, next_state, reward, [0.0])
        #
        # if self.norm_type != 'none':
        #     self.network_manager.input_norm.update(state)

    def learn(self):
        if self.replay_buffer.get_size() > max(self.warmup_steps, self.batch_size):
            state, action, reward, next_state, gamma = self.replay_buffer.sample_batch(self.batch_size)
            self.network_manager.update_network(state, action, next_state, reward, gamma)
        else:
            return

    # Resets the agent between episodes. Should primarily be used to clear traces or other temporally linked parameters
    def reset(self):
        self.network_manager.reset()

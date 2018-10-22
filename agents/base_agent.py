from utils.replaybuffer import ReplayBuffer


# Agent interface
# Takes an environment (just so we can get some details from the environment like the number of observables and actions)
class BaseAgent(object):
    def __init__(self, config):

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

        self.network_manager = None
        self.writer = config.writer
        self.config = config

    def start(self, state, is_train):
        raise NotImplementedError

    def step(self, state, is_train):
        raise NotImplementedError

    def get_value(self, s, a):
        raise NotImplementedError
    
    def update(self, state, next_state, reward, action, is_terminal, is_truncated):
        raise NotImplementedError

    # Sets the random seed for the agent
    def set_seed(self, seed):
        raise NotImplementedError

    # Resets the agent between episodes. Should primarily be used to clear traces or other temporally linked parameters
    def reset(self):
        raise NotImplementedError





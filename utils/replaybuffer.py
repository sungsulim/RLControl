"""
Data structure for Replay Buffer

"""
from utils.custom_collections import RandomAccessQueue

from collections import deque, namedtuple
import numpy as np


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'transition_gamma'])


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed):

        # self.rng = np.random.RandomState(random_seed)
        self.buffer_size = int(buffer_size)
        # self.count = 0

        # Right side of deque contains newest experience
        self.buffer = RandomAccessQueue(maxlen=buffer_size, seed=random_seed)

    def add(self, state, action, reward, next_state, transition_gamma):
        experience = Transition(state, action, reward, next_state, transition_gamma)
        self.buffer.append(experience)

    def get_size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):

        assert len(self.buffer) >= batch_size
        batch = self.buffer.sample(batch_size)

        return map(np.array, zip(*batch))

    # this probably won't be used
    def clear(self):
        print('clear buffer')
        self.buffer = RandomAccessQueue(maxlen=self.buffer_size)


"""
Data structure for Replay Buffer

"""
from collections import deque, namedtuple
import random
import numpy as np

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'transition_gamma'])

class ReplayBuffer(object):

    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.count = 0
        # Right side of deque contains newest experience
        self.buffer = deque()
        

    def add(self, state, action, reward, next_state, transition_gamma):
        experience = Transition(state, action, reward, next_state, transition_gamma)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def get_size(self):
        return self.count

    def sample_batch(self, batch_size):
        assert(self.count >= batch_size)
        batch = random.sample(self.buffer, batch_size)

        return map(np.array, zip(*batch))

    def clear(self):
        self.buffer.clear()
        self.count = 0

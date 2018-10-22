"""
Data structure for Replay Buffer

"""
from collections import deque, namedtuple
import numpy as np

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'transition_gamma'])


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed):

        self.rng = np.random.RandomState(random_seed)
        self.buffer_size = int(buffer_size)
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

        idx = self.rng.choice(range(self.count), size=batch_size, replace=False)
        batch = [val for i, val in enumerate(self.buffer) if i in idx]
        return map(np.array, zip(*batch))

    def clear(self):
        self.buffer.clear()
        self.count = 0

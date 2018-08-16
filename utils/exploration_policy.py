import numpy as np

class OrnsteinUhlenbeckProcess(object):
    def __init__(self, action_dim, action_min, action_max, theta, mu, sigma):

        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max

        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.noise_t = self.mu

    def generate(self, greedy_action, step):
        # self.noise_t += self.theta * (self.mu - self.noise_t) + self.sigma * np.random.normal(np.zeros(self.action_dim), np.ones(self.action_dim))
        self.noise_t += np.random.normal(self.mu * np.ones(self.action_dim), self.sigma * np.ones(self.action_dim)) - self.noise_t * self.theta
        return np.clip(greedy_action + self.noise_t, self.action_min, self.action_max)

    def reset(self):
        self.noise_t = self.mu
        

class RandomUniform(object):
    def __init__(self, action_min, action_max, is_continuous):

        self.action_min = action_min
        self.action_max = action_max
        self.is_continuous = is_continuous

    def generate(self, greedy_action, step):
        if self.is_continuous:
            return np.random.uniform(self.action_min, self.action_max)
        else:
            return np.random.choice(range(int(self.action_max - self.action_min + 1)))
    def reset(self):
        pass

class EpsilonGreedy(object):
    def __init__(self, action_min, action_max, annealing_steps, min_epsilon, max_epsilon, is_continuous):

        self.action_min = action_min
        self.action_max = action_max
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon

        self.annealing_steps = annealing_steps
        self.epsilon_step = - (self.epsilon - self.min_epsilon) / float(self.annealing_steps)

        self.is_continuous = is_continuous

    # generates next action
    def generate(self, greedy_action, step):
        epsilon = max(self.min_epsilon, self.epsilon_step * step + self.epsilon)
        if np.random.random() < epsilon:
            if self.is_continuous:
                return np.random.uniform(self.action_min, self.action_max)
            else:
                return np.random.choice(range(int(self.action_max - self.action_min + 1)))
        else:
            return greedy_action

    def reset(self):
        pass
            





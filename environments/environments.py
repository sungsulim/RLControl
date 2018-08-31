import gym
import numpy as np
import random
import math

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
mpl.use('Agg')

import matplotlib.pyplot as plt


def create_environment(env_params):
    env_name = env_params['environment']

    if env_name == 'Bimodal1DEnv':
        return Bimodal1DEnvironment(env_params)
    elif env_name == 'Bimodal2DEnv':
        return Bimodal2DEnvironment(env_params)
    else:
        return ContinuousEnvironment(env_params)


# This file provide environments to interact with, consider actions as continuous, need to rewrite otherwise
class ContinuousEnvironment(object):
    def __init__(self, env_params):

        self.name = env_params['environment']
        self.eval_interval = env_params['EvalIntervalMilSteps'] * 1000000
        self.eval_episodes = env_params['EvalEpisodes']

        self.instance = gym.make(env_params['environment'])

        # total number of steps allowed in a run
        self.TOTAL_STEPS_LIMIT = env_params['TotalMilSteps'] * 1000000
        # self.TOTAL_EPISODES_LIMIT = env_params['TotalEpisodes']

        # maximum number of steps allowed for each episode
        # if -1 takes default setting from gym
        if env_params['EpisodeSteps'] != -1:
            self.EPISODE_STEPS_LIMIT = env_params['EpisodeSteps']
            self.instance._max_episode_steps = env_params['EpisodeSteps']

        else:
            self.EPISODE_STEPS_LIMIT = self.instance._max_episode_steps
        
        # state info
        self.state_dim = self.get_state_dim()
        self.state_range = self.get_state_range()
        self.state_min = self.get_state_min()
        self.state_max = self.get_state_max()
        self.state_bounded = False if np.any(np.isinf(self.instance.observation_space.high)) or np.any(np.isinf(self.instance.observation_space.low)) else True
        
        # action info
        self.action_dim = self.get_action_dim()
        self.action_range = self.get_action_range()
        self.action_min = self.get_action_min()
        self.action_max = self.get_action_max()


        #DEBUG
        # print('state_dim:',self.state_dim)
        # print('state_range:', self.state_range)
        # print('state_min:', self.state_min)
        # print('state_max:', self.state_max)
        # print("state_bounded :: ", self.state_bounded)

        # print("action_dim:", self.action_dim)
        # print('action_range:', self.action_range)
        # print("action_min:", self.action_min)
        # print('action_max', self.action_max)
        # exit()

    def seed(self, seed):
        self.instance.seed(seed)


    # Reset the environment for a new episode. return the initial state
    def reset(self):
        state = self.instance.reset()
        '''
        if self.state_bounded:
            # normalize to [-1,1]
            scaled_state = 2.*(state - self.state_min)/self.state_range - 1.
            return scaled_state
        '''
        return state

    def step(self, action):
        state, reward, done, info = self.instance.step(action)

        '''
        if self.state_bounded:
            scaled_state = 2.*(state - self.state_min)/self.state_range - 1.
            return (scaled_state, reward, done, info)
        '''
        return (state, reward, done, info)

    def get_state_dim(self):
        return self.instance.observation_space.shape[0]
  
    # this will be the output units in NN
    def get_action_dim(self):
        if hasattr(self.instance.action_space, 'n'):
            return int(self.instance.action_space.n-1)
        return int(self.instance.action_space.sample().shape[0])

    # Return action ranges, NOT IN USE
    def get_action_range(self):
        if hasattr(self.instance.action_space, 'high'):
            return self.instance.action_space.high - self.instance.action_space.low
        return self.instance.action_space.n - 1    

    # Return action ranges
    def get_action_max(self):
        #print self.instance.action_space.dtype
        if hasattr(self.instance.action_space, 'high'):
            #self.action_space = spaces.Box(low=self.instance.action_space.low, high=self.instance.action_space.high, shape=self.instance.action_space.low.shape, dtype = np.float64)
            return self.instance.action_space.high
        return self.instance.action_space.n - 1    

    # Return action min
    def get_action_min(self):
        if hasattr(self.instance.action_space, 'low'):
            return self.instance.action_space.low
        return 0

    # Return state range
    def get_state_range(self):
        return self.instance.observation_space.high - self.instance.observation_space.low
    
    # Return state min
    def get_state_min(self):
        return self.instance.observation_space.low

    def get_state_max(self):
        return self.instance.observation_space.high

    # Close the environment and clear memory
    def close(self):
        self.instance.close()


class Bimodal1DEnvironment(object):
    def __init__(self, env_params):

        self.name = env_params['environment']
        self.eval_interval = env_params['EvalIntervalMilSteps'] * 1000000
        self.eval_episodes = env_params['EvalEpisodes']


        # total number of steps allowed in a run
        self.TOTAL_STEPS_LIMIT = env_params['TotalMilSteps'] * 1000000

        # maximum number of steps allowed for each episode
        # if -1 takes default setting from gym
        if env_params['EpisodeSteps'] != -1:
            self.EPISODE_STEPS_LIMIT = env_params['EpisodeSteps']

        else:
            self.EPISODE_STEPS_LIMIT = 1 # only one state env
        
        # state info
        self.state_dim = 1
        # self.state_range = np.array([10.])
        # self.state_min = np.array([-5.])
        # self.state_max = np.array([5.])
        self.state_range = np.array([4.])
        self.state_min = np.array([-2.])
        self.state_max = np.array([2.])
        self.state_bounded = True
        
        # action info
        self.action_dim = 1
        # self.action_range = np.array([10.])
        # self.action_min = np.array([-5.])
        # self.action_max = np.array([5.])
        self.action_range = np.array([4.])
        self.action_min = np.array([-2.])
        self.action_max = np.array([2.])

        #DEBUG
        # print('stateDim:',self.stateDim)
        # print('stateRange:', self.stateRange)
        # print('stateMin:', self.stateMin)
        # print("stateBounded :: ", self.stateBounded)

        # print("actionDim", self.actionDim)
        # print('actRange', self.actRange)
        # print("actionBound :: ", self.actionBound)
        # print('actMin', self.actMin)
        # exit()
        
    def seed(self, seed):
        # No randomness in this env
        pass

    # Reset the environment for a new episode. return the initial state
    def reset(self):
        # starts at 0.
        self.state = np.array([0.])
        #self.state = np.array([np.random.uniform(low=-0.1, high=0.1)])
        return self.state

    def step(self, action):
        self.state = self.state + action # terminal state
        reward = self.reward_func(action)
        done = True
        info = {}

        return self.state, reward, done, info

    def reward_func(self, action):

        maxima1 = -1.0
        maxima2 = 1.0
        stddev = 0.2

        # Reward function.
        # Two gaussian functions.
        modal1 = 1. * math.exp(-0.5 * ((action - maxima1) / stddev)**2)
        modal2 = 1.5 * math.exp(-0.5 * ((action - maxima2) / stddev)**2)

        return modal1 + modal2

    # Close the environment and clear memory
    def close(self):
        pass


class Bimodal2DEnvironment(object):
    def __init__(self, env_params):

        self.name = env_params['environment']
        self.eval_interval = env_params['EvalIntervalMilSteps'] * 1000000
        self.eval_episodes = env_params['EvalEpisodes']

        # total number of steps allowed in a run
        self.TOTAL_STEPS_LIMIT = env_params['TotalMilSteps'] * 1000000

        # maximum number of steps allowed for each episode
        # if -1 takes default setting from gym
        if env_params['EpisodeSteps'] != -1:
            self.EPISODE_STEPS_LIMIT = env_params['EpisodeSteps']

        else:
            self.EPISODE_STEPS_LIMIT = 1  # only one state env

        # state info
        self.state_dim = 2
        self.state_range = np.array([12.0, 12.0])
        self.state_min = np.array([-6.0, -6.0])
        self.state_max = np.array([6.0, 6.0])
        self.state_bounded = True

        # action info
        self.action_dim = 2
        self.action_range = np.array([2.0, 2.0])
        self.action_min = np.array([-1.0, -1.0])
        self.action_max = np.array([1.0, 1.0])

        self.state = None
        self.goal_states = np.array([[-4.0, -4.0], [4.0, 4.0]])

        # DEBUG
        # print('stateDim:',self.stateDim)
        # print('stateRange:', self.stateRange)
        # print('stateMin:', self.stateMin)
        # print("stateBounded :: ", self.stateBounded)

        # print("actionDim", self.actionDim)
        # print('actRange', self.actRange)
        # print("actionBound :: ", self.actionBound)
        # print('actMin', self.actMin)
        # self.plot_reward_func()

        # optimal trajectory
        opt_return = 0.0

        # opt_return += self.reward_func([0, 0])
        opt_return += self.reward_func([1, -1])
        opt_return += self.reward_func([2, -2])
        opt_return += self.reward_func([3, -3])
        opt_return += self.reward_func([4, -4])
        print("====== opt_return", opt_return)
        # exit()

    def seed(self, seed):
        # No randomness in this env
        # TODO: Perhaps control randomness in setting initial states
        pass

    # Reset the environment for a new episode. return the initial state
    def reset(self):
        #val = np.random.uniform(-1.0, 1.0)
        val = 0.0
        # the agent is initialized along the diagonal axis
        self.state = np.array([val, val])

        return self.state

    def step(self, action):
        self.state = np.clip(self.state + action, self.state_min, self.state_max)  # terminal state
        reward = self.reward_func(self.state)

        done = self.reached_goal(self.state)
        info = {}

        return self.state, reward, done, info

    def reward_func(self, state):

        magnitude = 125
        stddev = 2.25

        # Reward function.
        # Bimodal Gaussian mixture
        coeff1 = 0.5
        coeff2 = 1 - coeff1

        modal1 = coeff1 * 1.0/(2 * np.pi * np.square(stddev)) * np.exp(-0.5*(np.square((state[0]-self.goal_states[0][0])/stddev) + np.square((state[1]-self.goal_states[0][1])/stddev)))
        modal2 = coeff2 * 1.0/(2 * np.pi * np.square(stddev)) * np.exp(-0.5 * (np.square((state[0] - self.goal_states[1][0])/stddev) + np.square((state[1] - self.goal_states[1][1])/stddev)))

        # state = [0, 0]
        # modal1 = coeff1 * 1.0 / (2 * np.pi * np.square(stddev)) * np.exp(-0.5 * (
        #         np.square((state[0] - self.goal_states[0][0]) / stddev) + np.square(
        #     (state[1] - self.goal_states[0][1]) / stddev)))
        # modal2 = coeff2 * 1.0 / (2 * np.pi * np.square(stddev)) * np.exp(-0.5 * (
        #         np.square((state[0] - self.goal_states[1][0]) / stddev) + np.square(
        #     (state[1] - self.goal_states[1][1]) / stddev)))
        #
        # reward = magnitude * (modal1 + modal2) - 2
        # print(state, reward)
        #
        # state = [-4, 4]
        # modal1 = coeff1 * 1.0 / (2 * np.pi * np.square(stddev)) * np.exp(-0.5 * (
        #             np.square((state[0] - self.goal_states[0][0]) / stddev) + np.square(
        #         (state[1] - self.goal_states[0][1]) / stddev)))
        # modal2 = coeff2 * 1.0 / (2 * np.pi * np.square(stddev)) * np.exp(-0.5 * (
        #             np.square((state[0] - self.goal_states[1][0]) / stddev) + np.square(
        #         (state[1] - self.goal_states[1][1]) / stddev)))
        #
        # reward = magnitude * (modal1 + modal2) - 2
        # print(state, reward)
        #
        # exit()

        reward = magnitude * (modal1 + modal2) - 2

        return reward

    def plot_reward_func(self):
        x = np.linspace(self.state_min[0], self.state_max[0], 50)
        y = np.linspace(self.state_min[1], self.state_max[1], 50)
        X, Y = np.meshgrid(x, y)
        Z = self.reward_func([X, Y])
        contours = plt.contour(X, Y, Z)

        plt.clabel(contours, inline=True, fontsize=5)

        plt.imshow(Z, extent=[self.state_min[0], self.state_max[0], self.state_min[1], self.state_max[1]], origin='lower',
                   cmap='inferno', alpha=0.5)
        plt.colorbar()
        plt.show()


    # Close the environment and clear memory
    def close(self):
        pass

    def reached_goal(self, state):

        for goal in self.goal_states:
            if np.sum(np.square(np.abs(goal - state))) <= 0.5:
                return True

        return False

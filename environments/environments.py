import gym
import numpy as np
import random
import math


# import pybullet as p
# import pybullet_envs


def create_environment(env_params):
    env_name = env_params['environment']

    if env_name == 'MountainCarContinuous-v0':
        return ContinuousEnvironment(env_params)
    elif env_name == 'Pendulum-v0':
        return ContinuousEnvironment(env_params)
    elif env_name == 'BipedalWalker-v2':
        return ContinuousEnvironment(env_params)
    elif env_name == 'LunarLanderContinuous-v2':
        return ContinuousEnvironment(env_params)
    elif env_name == 'BimodalEnv':
        return BimodalEnvironment(env_params)

    #### PYBULLET ENV
    # elif env_name == 'HalfCheetahBulletEnv-v0':
    #     return ContinuousEnvironment(env_params)
    # elif env_name == 'AntBulletEnv-v0':
    #     return ContinuousEnvironment(env_params)
    # elif env_name == 'HopperBulletEnv-v0':
    #     return ContinuousEnvironment(env_params)
    # elif env_name == 'Walker2DBulletEnv-v0':
    #     return ContinuousEnvironment(env_params)
    ####

    #### Mujoco ENV
    elif env_name == 'InvertedPendulum-v2':
        return ContinuousEnvironment(env_params)
    elif env_name == 'HalfCheetah-v2':
        return ContinuousEnvironment(env_params)
    elif env_name == 'Hopper-v2':
        return ContinuousEnvironment(env_params)
    else:
        print('Environment not found!!!!!')
    return None

#This file provide environments to interact with, consider actions as continuous, need to rewrite otherwise
class ContinuousEnvironment(object):
    def __init__(self, env_params):


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


class BimodalEnvironment(object):
    def __init__(self, env_params):


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
        self.state_range = np.array([10.])
        self.state_min = np.array([-5.])
        self.state_max = np.array([5.])
        self.state_bounded = True
        
        # action info
        self.action_dim = 1
        self.action_range = np.array([10.]) 
        self.action_min = np.array([-5.])
        self.action_max = np.array([5.])

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

        return (self.state, reward, done, info)


    def reward_func(self, action):

        maxima1 = -2.5
        maxima2 = 2.5
        stddev = 0.5

        # Reward function.
        # Two gaussian functions.
        modal1 = 1.5 * math.exp(-0.5 * ((action - maxima1) / stddev)**2)
        modal2 = 1. * math.exp(-0.5 * ((action - maxima2) / stddev)**2)

        return modal1 + modal2

    # Close the environment and clear memory
    def close(self):
        pass

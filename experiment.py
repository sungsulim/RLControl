import numpy as np
import random

from datetime import datetime
import time
import copy
import tensorflow as tf

#import cProfile, pstats, StringIO
#pr = cProfile.Profile()

## DEBUG PARAMS
## -1 to disable
output_ep_result_fq = 1 # print to console (not saved output) after this many episodes
save_maxQ_fq = -1 # plot cost-to-go after this many episodes
plot_maxA_fq = -1 # plot maxA after this many episodes

        
class Experiment(object):
    def __init__(self, agent, train_environment, test_environment, seed, summary_dir, write_log):
        self.agent = agent
        self.train_environment = train_environment
        self.train_environment.seed(seed)

        # for eval purpose
        self.test_environment = test_environment # copy.deepcopy(environment) # this didn't work for Box2D env

        self.train_rewards_per_episode = []        
        self.eval_mean_rewards_per_episode = []
        self.eval_std_rewards_per_episode = []

        self.total_step_count = 0

        self.writer = tf.summary.FileWriter(summary_dir)
        
        # set writer in agent too so we can log useful stuff
        self.agent.set_writer(self.writer)

        # boolean to log result for tensorboard
        self.write_log = write_log

    def run(self):

        episode_count = 0

        print("Start run at: " + str(datetime.now())+'\n')
        self.start_time = time.time()
        # pr.enable()

        # evaluate once at beginning
        self.eval()
        
        while self.total_step_count < self.train_environment.TOTAL_STEPS_LIMIT:
            # runs a single episode and returns the accumulated reward for that episode
            episode_reward, num_steps, force_terminated = self.run_episode_train(is_train=True)

            if output_ep_result_fq != -1 and episode_count % output_ep_result_fq == 0:

                self.end_time = time.time()
                elapsed_time = self.end_time - self.start_time
                self.start_time = self.end_time


                print("Train:: ep: "+ str(episode_count) + ", r: " + str(episode_reward) + ", n_steps: " + str(num_steps) + ", elapsed: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

            if not force_terminated: 
                self.train_rewards_per_episode.append(episode_reward)

            # write tf summary
            if not self.total_step_count == self.train_environment.TOTAL_STEPS_LIMIT:
                
                if self.write_log:
                    write_summary(self.writer, episode_count, episode_reward, "train/episode_reward")

        
            episode_count += 1

        self.train_environment.close() # clear environment memory
        print("End run at: " + str(datetime.now())+'\n')
        return (self.train_rewards_per_episode, self.eval_mean_rewards_per_episode, self.eval_std_rewards_per_episode)

    # Runs a single episode (TRAIN)
    def run_episode_train(self, is_train):
        
        obs = self.train_environment.reset()
        self.agent.reset() # Need to be careful in Agent not to reset the weight

        episode_reward = 0.
        done = False
        Aold = self.agent.start(obs, is_train)

        # print('initial state', obs)
        # print('action', Aold)
        # input()

        episode_step_count = 0

        while not (done or episode_step_count == self.train_environment.EPISODE_STEPS_LIMIT or self.total_step_count == self.train_environment.TOTAL_STEPS_LIMIT):

            episode_step_count += 1
            self.total_step_count += 1

            obs_n, reward, done, info = self.train_environment.step(Aold)
            episode_reward += reward

            # if the episode was externally terminated by episode step limit, don't do update
            if done and episode_step_count == self.train_environment.EPISODE_STEPS_LIMIT:
                is_truncated = True
            else:
                is_truncated = False

            self.agent.update(obs, obs_n, float(reward), Aold, done, is_truncated)

            if not done:
                Aold = self.agent.step(obs_n, is_train)

            obs = obs_n

            # print("reward", reward)
            # print('state', obs_n)
            # print('action', Aold)
            # input()

            if self.total_step_count % self.train_environment.eval_interval == 0:
                self.eval()

        # check if this episode is finished because of Total Training Step Limit
        if not (done or episode_step_count == self.train_environment.EPISODE_STEPS_LIMIT):
            force_terminated = True
        else:
            force_terminated = False
        return (episode_reward, episode_step_count, force_terminated)

    def eval(self):
        temp_rewards_per_episode = []

        for i in range(self.test_environment.eval_episodes):
            # with open('eval_log_exploration.txt', 'a+') as logfile:
            #     logfile.write('=== EP '+str(i)+' ===\n')
            episode_reward, num_steps = self.run_episode_eval(self.test_environment, is_train=False)

            temp_rewards_per_episode.append(episode_reward)

            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            self.start_time = self.end_time

            print("=== EVAL :: ep: "+ str(i) + ", r: " + str(episode_reward) + ", n_steps: " + str(num_steps) + ", elapsed: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        
        mean = np.mean(temp_rewards_per_episode)

        self.eval_mean_rewards_per_episode.append(mean)
        self.eval_std_rewards_per_episode.append(np.std(temp_rewards_per_episode))

        if self.write_log:
            write_summary(self.writer, self.total_step_count, mean, "eval/episode_reward/no_noise")


    # Runs a single episode (EVAL)
    def run_episode_eval(self, test_env, is_train):
        obs = test_env.reset()
        self.agent.reset()

        episode_reward = 0.
        done = False
        Aold = self.agent.start(obs, is_train)

        # print('initial state', obs)
        # print('action', Aold)
        # input()

        episode_step_count = 0
        while not (done or episode_step_count == test_env.EPISODE_STEPS_LIMIT):
            
            obs_n, reward, done, info = test_env.step(Aold)

            episode_reward += reward  
            if not done:          
                Aold = self.agent.step(obs_n, is_train)

            # with open('eval_log_exploration.txt', 'a+') as logfile:
            #     logfile.write(str(obs_n)+', '+str(Aold)+'\n')

            obs = obs_n
            episode_step_count += 1

            # print('state', obs_n)
            # print('action', Aold)
            # input()
        return (episode_reward, episode_step_count)


# write to tf Summary
def write_summary(writer, increment, stuff_to_log, tag):
    summary = tf.Summary() 
    summary.value.add(simple_value=stuff_to_log, tag=tag)
    writer.add_summary(summary, increment)
    writer.flush()




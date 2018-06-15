# -*- encoding:utf8 -*-
import gym

import environments.environments as envs
from utils.config import Config

import numpy as np
import sys
import json
import os
import datetime
from collections import OrderedDict

import argparse


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_json', type=str)
    parser.add_argument('--agent_json', type=str)
    parser.add_argument('--index', type=int)
    parser.add_argument('--monitor', default=False, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')

    args = parser.parse_args()

    # read env/agent json
    with open(args.env_json, 'r') as env_dat:
        env_json = json.load(env_dat, object_pairs_hook=OrderedDict)

    with open(args.agent_json, 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)




    # initialize env
    train_env =  envs.create_environment(env_json)
    test_env = envs.create_environment(env_json)

    from utils.main_utils import get_sweep_parameters, create_agent
    agent_params, total_num_sweeps = get_sweep_parameters(agent_json['sweeps'], args.index)    
    
    # init config and merge custom config settings from json
    config = Config()
    config.merge_config(agent_params)

    # get run idx and setting idx
    RUN_NUM = int(args.index / total_num_sweeps)
    SETTING_NUM = args.index % total_num_sweeps

    # set Random Seed (for training)
    RANDOM_SEED = RUN_NUM


    # initialize agent
    agent = create_agent(agent_json['agent'], train_env, config, RANDOM_SEED)



    # create save directory
    save_dir = 'results/' + env_json['environment'] + 'results/'
    if os.path.exists(save_dir)==False:
        os.makedirs(save_dir)

    # create summary directory (for tensorboard, gym monitor/render)
    START_DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    summary_dir = './results/{}results/log_summary/{}/{}_{}_{}'.format(str(env_json['environment']), str(agent_json['agent']), str(SETTING_NUM), str(RUN_NUM), str(START_DATETIME))

    # monitor/render
    if args.monitor or args.render:
        monitor_dir = summary_dir+'/monitor'

        if args.render:
            train_env.instance = gym.wrappers.Monitor(train_env.instance, monitor_dir, force=True)
        else:
            train_env.instance = gym.wrappers.Monitor(train_env.instance, monitor_dir, video_callable=False, force=True)

        
        


    from experiment import Experiment
    
    # initialize experiment
    experiment = Experiment(agent=agent, train_environment=train_env, test_environment= test_env, seed=RANDOM_SEED, summary_dir=summary_dir)
    
    # run experiment
    episode_rewards, eval_episode_mean_rewards, eval_episode_std_rewards  = experiment.run()


    # save to file
    prefix = save_dir + env_json['environment']+'_'+agent_json['agent']+'_setting_'+ str(SETTING_NUM) + '_run_'+str(RUN_NUM)

    train_rewards_filename = prefix +'_EpisodeRewardsLC.txt'
    np.array(episode_rewards).tofile(train_rewards_filename, sep=',', format='%15.8f')

    eval_mean_rewards_filename = prefix +'_EvalEpisodeMeanRewardsLC.txt'
    np.array(eval_episode_mean_rewards).tofile(eval_mean_rewards_filename, sep=',', format='%15.8f')

    eval_std_rewards_filename = prefix +'_EvalEpisodeStdRewardsLC.txt'
    np.array(eval_episode_std_rewards).tofile(eval_std_rewards_filename, sep=',', format='%15.8f')

    params = []
    params_names = '_'
    for key in agent_params:
       # for Python 2 since JSON load delivers "unicode" rather than pure string
       # then it will produce problem at plotting stage
       if isinstance(agent_params[key], type(u'')):
          params.append(agent_params[key].encode('utf-8'))
       else:
          params.append(agent_params[key])
       params_names += (key +'_')

    params = np.array(params)
    name = prefix + params_names + 'Params.txt'

    params.tofile(name, sep=',', format='%s')



if __name__ == '__main__':
    main()



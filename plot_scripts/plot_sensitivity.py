import matplotlib.pyplot as plt

import numpy as np
import glob
import sys
from pathlib import Path
from collections import OrderedDict
import json
from utils.main_utils import get_sweep_parameters


store_dir = str(sys.argv[1])
# agent_filename = str(sys.argv[2])

policy_lr_curve_data = {}
value_lr_curve_data = {}

# Bimodal
# env_name = 'Bimodal1DEnv'
# plt_yticks = [0., 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] # range(0, 1.6, 0.2)
# for agent_filename in ['ae', 'ae_plus', 'qt_opt', 'naf', 'picnn', 'wirefitting', 'sql', 'optimalq', 'ddpg', 'ac_separate', 'sac']:

# Pendulum
env_name = 'Pendulum-v0'
plt_yticks = [-1600, -1200, -800, -400, -200, 0]
for agent_filename in ['ae_separate', 'ae_plus_separate', 'qt_opt', 'naf', 'picnn', 'sql', 'ddpg', 'ac_separate', 'sac']:

    if agent_filename == 'ae':
        sweep_params = ['actor_lr', 'expert_lr']

    elif agent_filename == 'ae_plus':
        sweep_params = ['actor_lr', 'expert_lr']

    if agent_filename == 'ae_separate':
        sweep_params = ['actor_lr', 'expert_lr']

    elif agent_filename == 'ae_plus_separate':
        sweep_params = ['actor_lr', 'expert_lr']

    elif agent_filename == 'qt_opt':
        sweep_params = ['qnet_lr']

    elif agent_filename == 'naf':
        sweep_params = ['noise_scale', 'learning_rate']

    elif agent_filename == 'picnn':
        sweep_params = ['expert_lr']

    elif agent_filename == 'wirefitting':
        sweep_params = ['learning_rate']

    elif agent_filename == 'sql':
        sweep_params = ['actor_lr', 'expert_lr', 'reward_scale']

    elif agent_filename == 'optimalq':
        sweep_params = ['learning_rate']

    elif agent_filename == 'ddpg':
        sweep_params = ['actor_lr', 'critic_lr']

    elif agent_filename == 'ac_separate':
        sweep_params = ['actor_lr', 'critic_lr']

    elif agent_filename == 'sac':
        sweep_params = ['pi_lr', 'qf_vf_lr', 'reward_scale']
    else:
        raise ValueError("invalid agent filename: {}".format(agent_filename))
    # agent_name = 'sac'
    # sweep_params = ['pi_lr', 'qf_vf_lr', 'reward_scale']


    # load json
    json_dir = 'jsonfiles/agent/{}.json'.format(agent_filename)

    with open(json_dir, 'r') as agent_dat:

        json_data = json.load(agent_dat, object_pairs_hook=OrderedDict)

        agent_name = json_data['agent']
        agent_json = json_data['sweeps']


    eval_mean_filename = '{}/merged{}results/{}_{}_EvalEpisodeMeanRewardsLC.txt'.format(store_dir, env_name, env_name, agent_name)
    eval_mean_result = np.loadtxt(eval_mean_filename, delimiter=',')


    params_max_idx = [len(agent_json[p]) for p in sweep_params]
    num_settings = np.prod(params_max_idx)

    # actor: 0, 5, 10
    # expert: 0,1,2,3,4

    x = 1
    y = 1

    for p in sweep_params:
        print("cur sweep param: {}".format(p))

        plt_x = agent_json[p]
        plt_xticks = range(len(plt_x))

        cur_param_num = len(plt_x)

        x *= cur_param_num
        plt_cum_reward_y = []
        for i in range(0, x, y):

            idx_array = []
            for j in range(i, i + num_settings, x):
                for k in range(j, j+y, 1):
                    idx_array.append(k)

            # print("{}: {} - sweep idx: {}".format(p, i, idx_array))

            # find result and mean
            # param_sweep_array.append(idx_array)
            result_array = [np.mean(eval_mean_result[idx]) for idx in idx_array]
            plt_cum_reward_y.append(np.max(result_array))
        y *= cur_param_num

        # print("param {}: {}".format(p, param_sweep_array))

        # plot result
        plt_x = plt_x[::-1]
        # plt_xticks = plt_xticks[::-1]
        plt_cum_reward_y = plt_cum_reward_y[::-1]
        plt.plot(plt_xticks, plt_cum_reward_y)

        if p in ['actor_lr', 'pi_lr']:
            policy_lr_curve_data[agent_name] = (p, plt_xticks, plt_x, plt_cum_reward_y.copy())
        elif p in ['expert_lr', 'qnet_lr', 'learning_rate', 'critic_lr', 'qf_vf_lr']:
            value_lr_curve_data[agent_name] = (p, plt_xticks, plt_x, plt_cum_reward_y.copy())

        plt.title("{}: {} sensitivity curve".format(agent_name, p))
        plt.xlabel("{}".format(p))
        plt.ylabel("Avg. reward during evaluation", rotation=90)
        plt.xticks(plt_xticks, plt_x)
        # plt.yticks(,)

        # plt.show()
        plt.savefig("{}_{}_sensitivity_curve.png".format(agent_name, p))
        plt.clf()

# Combined plots
#
# print(value_lr_curve_data.keys())
# exit()


show_label = False

colors = [ '#377eb8', '#4daf4a', '#ff7f00',
                  '#f781bf', '#984ea3', '#999999','#a65628',
                  '#e41a1c', '#999999', '#dede00']


# q-learning methods
# ae, ae_plus, qt_opt, naf, picnn, wirefitting, sql, optimalq

# Bimodal
# for idx, a in enumerate(['ActorExpert', 'ActorExpert_Plus', 'SoftQlearning', 'NAF', 'PICNN', 'QT_OPT', 'WireFitting', 'OptimalQ']):
# Pendulum
for idx, a in enumerate(['ActorExpert_Separate', 'ActorExpert_Plus_Separate', 'SoftQlearning', 'NAF', 'PICNN', 'QT_OPT']):

    plt_xticks = value_lr_curve_data[a][1]
    plt_x = value_lr_curve_data[a][2]
    plt_y = value_lr_curve_data[a][3]
    plt.plot(plt_xticks, plt_y, colors[idx], label="{}: {}".format(a, value_lr_curve_data[a][0]))
    plt.ylim(0, 1.5)

    if show_label:
        plt.xticks(plt_xticks, plt_x)
        plt.yticks(plt_yticks, plt_yticks)

    else:
        plt.xticks(plt_xticks, [])
        plt.yticks(plt_yticks, [])

if show_label:
    plt.legend()
    plt.title("Q-learning: Value LR sensitivity curve")
    plt.xlabel("Value LR")
    plt.ylabel("Cum reward", rotation=90)
plt.show()
plt.clf()

# policy + value
for idx, a in enumerate(['ActorExpert_Separate', 'ActorExpert_Plus_Separate', 'SoftQlearning', 'ActorCritic_Separate', 'SoftActorCritic', 'DDPG']):

    plt_xticks = value_lr_curve_data[a][1]
    plt_x = value_lr_curve_data[a][2]
    plt_y = value_lr_curve_data[a][3]
    plt.plot(plt_xticks, plt_y, colors[idx], label="{}: {}".format(a, value_lr_curve_data[a][0]))
    plt.ylim(0, 1.5)

    if show_label:
        plt.xticks(plt_xticks, plt_x)
        plt.yticks(plt_yticks, plt_yticks)

    else:
        plt.xticks(plt_xticks, [])
        plt.yticks(plt_yticks, [])

if show_label:
    plt.legend()
    plt.title("Policy + Value: Value LR sensitivity curve")
    plt.xlabel("Value LR")
    plt.ylabel("Cum reward", rotation=90)
plt.show()
plt.clf()

for idx, a in enumerate(['ActorExpert_Separate', 'ActorExpert_Plus_Separate', 'SoftQlearning', 'ActorCritic_Separate', 'SoftActorCritic', 'DDPG']):

    plt_xticks = policy_lr_curve_data[a][1]
    plt_x = policy_lr_curve_data[a][2]
    plt_y = policy_lr_curve_data[a][3]
    plt.plot(plt_xticks, plt_y, colors[idx], label="{}: {}".format(a, policy_lr_curve_data[a][0]))
    plt.ylim(0, 1.5)

    if show_label:
        plt.xticks(plt_xticks, plt_x)
        plt.yticks(plt_yticks, plt_yticks)

    else:
        plt.xticks(plt_xticks, [])
        plt.yticks(plt_yticks, [])
if show_label:
    plt.legend()
    plt.title("Policy + Value: Policy LR sensitivity curve")
    plt.xlabel("Policy LR")
    plt.ylabel("Cum reward", rotation=90)
plt.show()

# ac_separate, sac, sql, ddpg, ae, ae_plus

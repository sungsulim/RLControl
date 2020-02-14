# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
from pathlib import Path
import json

from plot_agent_new import get_xyrange

############## USAGE
# You should be in actiongeneral/python/results when running the script

# python3 ../plot_scripts/plot_agent_comparison.py ../jsonfiles/environment/Pendulum-v0.json DIR_of_npy 

# example: python3 ../plot_scripts/plot_agent_comparison.py../jsonfiles/environment/Pendulum-v0.json  mergedPendulum-v0results 

# This plot_script is used for comparing different agents (using the saved .npy from plot_custom_new.py)
# You should place the .npy of different agents in one directory (DIR_of_npy)


show_labels = True

display_idx = 0 # IDX to determine whether to plot train_episode or eval_episode (0: Train, 1: Eval)

agents = ['ForwardKL', 'ReverseKL', 'ForwardKL_noent', 'ReverseKL_noent']
# agents = ['ForwardKL_full_auc', 'ReverseKL_full_auc']

####### CONFIG ####

# This should be the agent name between Pendulum-v0_(   )_EvalEpisode_BestResult_avg.npy
# You can change the agent name between ( ) to be anything, as long as other format is same.

###### thesis plots
# Q-learning methods (A)
# agents = ['ActorExpert_together_bimodal', 'ActorExpert_Plus_together_bimodal', 'SoftQlearning', 'NAF', 'PICNN', 'QT_OPT']
# agents = ['ActorExpert', 'ActorExpert_Plus', 'SoftQlearning', 'NAF', 'PICNN', 'QT_OPT']

# Value & Policy (B)
# agents = ['ActorExpert_Separate_bimodal', 'ActorExpert_Plus_Separate_bimodal', 'SoftQlearning', 'ActorCritic_Separate_bimodal_sample_eval', 'SoftActorCritic_sample_eval', 'DDPG']
# agents = ['ActorExpert_Separate', 'ActorExpert_Plus_Separate', 'SoftQlearning', 'ActorCritic_Separate', 'SoftActorCritic', 'DDPG']

# (C)
# agents = ['ActorExpert_Separate_bimodal', 'ActorExpert_Plus_Separate_bimodal', 'SoftQlearning', 'ActorCritic_Separate_bimodal_mean_eval', 'SoftActorCritic_mean_eval', 'DDPG']
# agents = ['ActorExpert_Separate_bimodal', 'ActorExpert_Plus_Separate_bimodal', 'ActorCritic_Separate_bimodal_sample_eval', 'ActorCritic_Separate_bimodal_mean_eval', 'SoftActorCritic_sample_eval', 'SoftActorCritic_mean_eval']
# agents = ['ActorExpert_Separate', 'ActorExpert_Plus_Separate', 'ActorCritic_Separate', 'ActorCritic_Separate_mean', 'SoftActorCritic', 'SoftActorCritic_mean']

# AE together & separate (bimodal) (D)
# agents = ['ActorExpert', 'ActorExpert_Separate']


# AE together bimodal w, w/o uniform sampling (E)
# agents = ['ActorExpert_together_bimodal', 'ActorExpert_together_bimodal_uniform_sampling']
# agents = ['ActorExpert', 'ActorExpert_uni']

# AE better target (F)
# agents = ['ActorExpert_together_bimodal_mean_mean', 'ActorExpert_together_bimodal_ga_mean_next_best', 'ActorExpert_together_bimodal_mean_ga', 'ActorExpert_together_bimodal_ga_ga_next_best']
# agents = ['ActorExpert_mean_mean', 'ActorExpert_ga_mean', 'ActorExpert_mean_ga', 'ActorExpert_ga_ga']


# Sandbox
# agents = ['ActorExpert_together_bimodal', 'ActorExpert_LL_bimodal', 'ActorExpert_LL_randomq_bimodal']
# agents = ['ActorExpert_together_unimodal', 'ActorExpert_LL_unimodal', 'ActorExpert_LL_randomq_unimodal']

# PD
# agents = ['ActorExpert', 'ActorExpert_LL_randomq_bimodal', 'ActorExpert_LL_randomq_unimodal']



####################


if display_idx == 0:
    print("==== Plotting Training ====")
elif display_idx == 1:
    print("==== Plotting Evaluation ====")
else:
    raise ValueError("Wrong display_idx")

###################

# Stored Directory
DIR = str(sys.argv[2])+'/'

env_filename = str(sys.argv[1])
with open(env_filename, 'r') as env_dat:
    env_json = json.load(env_dat)

envname = env_json['environment']
TOTAL_MIL_STEPS = env_json['TotalMilSteps']
EVAL_INTERVAL_MIL_STEPS = env_json['EvalIntervalMilSteps']
EVAL_EPISODES = env_json['EvalEpisodes']


result_type = ['TrainEpisode', 'EvalEpisode']

suffix = 'BestResult'
data_type = ['avg', 'se']


agents_avg = []
agents_se = []

max_length = 1

for ag in agents:

    avg = np.load(DIR + envname+'_'+ag+'_'+result_type[display_idx]+'_'+suffix+'_'+data_type[0]+'.npy')
    se = np.load(DIR + envname+'_'+ag+'_'+result_type[display_idx]+'_'+suffix+'_'+data_type[1]+'.npy')

    if max_length < len(avg):
        max_length = len(avg)
        print('agent: {}, max_length: {}'.format(ag, max_length))

    agents_avg.append(avg)
    agents_se.append(se)


# mpl.style.use('default')
# for KL paper
colors = ['#377eb8', '#4daf4a', '#377eb8', '#4daf4a']
# colors = ['blue', 'navy', 'green', 'teal']

linestyles = ['-', '-', '--', '--']
dashes=[(5,0), (5,0), (5,7), (5, 7)]

alphas=[1.0, 1.0,1.0, 1.0]
markers=['o', 'o', '^', '^']
# used in thesis
# colors = [ '#377eb8', '#4daf4a', '#ff7f00',
#                   '#f781bf', '#984ea3', '#999999','#a65628',
#                   '#999999', '#e41a1c', '#dede00']

# used in ss curve
# colors = [ '#377eb8', '#4daf4a', '#ff7f00',
#                   '#f781bf', '#984ea3', '#999999','#a65628',
#                   '#e41a1c', '#999999', '#dede00']


# Color scheme for actor-critic sample/mean eval comparison
# colors = [ '#377eb8', '#4daf4a', '#ff7f00', '#ff7f00',
#                   '#f781bf', '#f781bf', ]

plt.figure(figsize=(12,6))

#xmax = int(max_length)
xmax, ymin, ymax = get_xyrange(envname)

if xmax is None:
    xmax = int(max_length)


# Train Episode Rewards
if display_idx == 0:
    ylimt = (ymin[display_idx], ymax[display_idx])

    if show_labels:
        plt.title(envname)
        plt.xlabel('Training steps (per 1000 steps)')
        plt.ylabel("Cum. Reward per episode").set_rotation(90)

    opt_range = range(0, xmax)
    xlimt = (0, xmax-1)

    print('training xmax: {}'.format(xmax))

    if envname == 'Pendulum-v0':
        # plt.axhline(y=-150, linewidth=1.0, linestyle='--', color='darkslategrey', label='-150')
        # plt.xticks(opt_range[::20], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax-1)), min(int(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS)+1, xmax))[::20])
        # 0, 10, 20, 30, 40, 50, 60, 70, ..., 150
        # 9, 10, 20, 30, 40 ... 150

        # 0, 1, 11, 21, 31, ... 141

        loc_arr = np.array([0, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141])
        val_arr = np.array([0.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]) * 2

        if not show_labels:
            plt.xticks(loc_arr, []) #val_arr)
            plt.yticks([-1400, -1200, -800, -400, -200, -100],[]) # [-1600, -1200, -800, -400, -200, 0])
        else:
            plt.xticks(loc_arr, val_arr)
            plt.yticks([-1400, -1200, -800, -400, -200, -100],[-1400, -1200, -800, -400, -200, -100])

    elif envname == 'Swimmer-v2':
        # 1000 episodes, 991 points
        loc_arr = np.array([0, 191, 391, 591, 791, 991])
        val_arr = np.array([45, 200, 400, 600, 800, 1000])

        if show_labels:
            plt.xticks(loc_arr, val_arr)
            plt.yticks([20, 25, 30, 35, 40], [20, 25, 30, 35, 40])
        else:
            plt.xticks(loc_arr, [])
            plt.yticks([20, 25, 30, 35, 40], [])  # [0, 2000, 4000, 6000, 7000])

    elif envname == 'Reacher-v2':
        # 20000 episodes, 19991 points
        loc_arr = np.array([0, 3991, 7991, 11991, 15991, 19991])
        val_arr = np.array([45, 200, 400, 600, 800, 1000])


        if show_labels:
            plt.xticks(loc_arr, val_arr)
            plt.yticks([-40, -35, -30], [-40, -35, -30])
        else:
            plt.xticks(loc_arr, [])
            plt.yticks([-40, -35, -30], [])  # [0, 2000, 4000, 6000, 7000])
    else:
        plt.xticks(opt_range[::50], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax-1)), int(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS)+1)[::50])



# EvalEpisode Rewards
elif display_idx == 1:
    ylimt = (ymin[display_idx], ymax[display_idx])

    if show_labels:
        plt.title(envname)
        plt.xlabel('Training steps (per 1000 steps)')
        plt.ylabel("Cum. Reward per episode").set_rotation(90)

    opt_range = range(0, xmax) 
    xlimt = (0, xmax-1)

    # opt_range = range(1, xmax+1)
    # xlimt = (1, xmax)
    
    

    if envname == 'Pendulum-v0':
        # plt.axhline(y=-150, linewidth=1.0, linestyle='--', color='darkslategrey', label='-150')
        # plt.xticks(opt_range[::20], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax-1)), min(int(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS)+1, xmax))[::20])
        # 0, 10, 20, 30, 40, 50, 60, 70, ..., 150
        # 9, 10, 20, 30, 40 ... 150

        # 0, 1, 11, 21, 31, ... 141

        # loc_arr = np.array([0, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141])
        # val_arr = np.array([0.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        loc_arr = np.array([0, 1, 3, 5, 7, 9, 11])
        val_arr = np.array([9, 10, 12, 14, 16, 18, 20])

        if not show_labels:
            plt.xticks(loc_arr, []) #val_arr)
            plt.yticks([-1400, -1200, -800, -400, -200, -100],[]) # [-1600, -1200, -800, -400, -200, 0])
        else:
            plt.xticks(loc_arr, val_arr)
            plt.yticks([-1400, -1200, -800, -400, -200, -100],[-1400, -1200, -800, -400, -200, -100])

    # elif envname == 'HalfCheetah-v2' or envname == 'Hopper-v2':
    #     opt_range = range(1, xmax+1)
    #     xlimt = (1, xmax)

    #     x_tick_interval = 10

    #     ## HARD CODED!! For HalfCheetah and Hopper
    #     loc_arr = np.array([1, 11, 21, 31, 41, 51, 61, 71, 81, 91])
    #     val_arr = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    elif envname == 'LunarLanderContinuous-v2':
        # og length: xmax = 192
        # print('xmax', xmax)
        # print('len',np.shape(agents_avg))
        # exit()
        # tick_interval = 10
        # 0, 10, 20, 30, ... 200
        # 9, 50, 100, 150, 200
        # 0, 1, 11, 21,
        # 201 -> 192

        loc_arr = np.array([0, 41, 91, 141, 191])
        val_arr = np.array([45, 250, 500, 750, 1000])


        if show_labels:
            plt.xticks(loc_arr, val_arr)
            plt.yticks([-200, -100, 0, 100, 200, 250],[-200, -100, 0, 100, 200, 250])
        else:
            plt.xticks(loc_arr, [])
            plt.yticks([-200, -100, 0, 100, 200, 250],[]) #[-200, -100, 0, 100, 200, 250])

    elif envname == 'HalfCheetah-v2':
        loc_arr = np.array([0, 41, 91, 141, 191])
        val_arr = np.array([45, 250, 500, 750, 1000])


        if show_labels:
            plt.xticks(loc_arr, val_arr)
            plt.yticks([0, 2000, 4000, 6000, 8000],[0, 2000, 4000, 6000, 8000])
        else:
            plt.xticks(loc_arr, [])
            plt.yticks([0, 2000, 4000, 6000, 8000],[]) #[0, 2000, 4000, 6000, 7000])

    elif envname == 'Hopper-v2':
        loc_arr = np.array([0, 41, 91, 141, 191])
        val_arr = np.array([45, 250, 500, 750, 1000])

        if show_labels:
            plt.xticks(loc_arr, val_arr)
            plt.yticks([0, 500, 1000, 1500, 2000, 2500, 3000], [0, 500, 1000, 1500, 2000, 2500, 3000])
        else:
            plt.xticks(loc_arr, [])
            plt.yticks([0, 500, 1000, 1500, 2000, 2500, 3000], []) # [0, 500, 1000, 1500, 2000, 2500, 3000])

    elif envname == 'Ant-v2':
        loc_arr = np.array([0, 41, 91, 141, 191])
        val_arr = np.array([45, 250, 500, 750, 1000])

        if show_labels:
            plt.xticks(loc_arr, val_arr)
            plt.yticks([-500, 0, 500, 1000, 1500, 2000], [-500, 0, 500, 1000, 1500, 2000])
        else:
            plt.xticks(loc_arr, [])
            plt.yticks([-500, 0, 500, 1000, 1500, 2000], [])  # [0, 2000, 4000, 6000, 7000])

    elif envname == 'Swimmer-v2':
        loc_arr = np.array([0, 16, 41, 66, 91])
        val_arr = np.array([90, 250, 500, 750, 1000])


        if show_labels:
            plt.xticks(loc_arr, val_arr)
            plt.yticks([20, 25, 30, 35, 40], [20, 25, 30, 35, 40])
        else:
            plt.xticks(loc_arr, [])
            plt.yticks([20, 25, 30, 35, 40], [])  # [0, 2000, 4000, 6000, 7000])

    elif envname == 'Reacher-v2':
        loc_arr = np.array([0, 16, 41, 66, 91])
        val_arr = np.array([90, 250, 500, 750, 1000])


        if show_labels:
            plt.xticks(loc_arr, val_arr)
            plt.yticks([-10, -7.5, -5, -2.5], [-10, -7.5, -5, -2.5])
        else:
            plt.xticks(loc_arr, [])
            plt.yticks([-10, -7.5, -5, -2.5], [])  # [0, 2000, 4000, 6000, 7000])

    else:
        plt.xticks(opt_range[::50], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax-1)), int(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS)+1)[::50])


plt.xlim(xlimt)
plt.ylim(ylimt)

handle_arr=[]  
for idx in range(len(agents)):
    pad_length = len(opt_range) - len(agents_avg[idx][:xmax])

    # For AC, SAC sample/mean eval comparison
    # if idx == 2 or idx == 4:
    #     handle, = plt.plot(opt_range, np.append(agents_avg[idx][:xmax], np.zeros(pad_length) + np.nan), colors[idx], linestyle='-.', linewidth=1.2, label=agents[idx])
    # else:
    plt.fill_between(opt_range,
                     np.append(agents_avg[idx][:xmax] - agents_se[idx][:xmax], np.zeros(pad_length) + np.nan),
                     np.append(agents_avg[idx][:xmax] + agents_se[idx][:xmax], np.zeros(pad_length) + np.nan),
                     alpha=0.2, facecolor=colors[idx])
    handle, = plt.plot(opt_range, np.append(agents_avg[idx][:xmax], np.zeros(pad_length) + np.nan), colors[idx], linewidth=1.2, label=agents[idx], linestyle=linestyles[idx], dashes=dashes[idx])
    handle_arr.append(handle)
    
if show_labels:
    plt.legend(handle_arr, agents)
plt.show()




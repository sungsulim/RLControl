import matplotlib as mpl
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



####### CONFIG ####
# This should be the agent name between Pendulum-v0_(   )_EvalEpisode_BestResult_avg.npy
# You can change the agent name between ( ) to be anything, as long as other format is same.

# combined
# agents = ['DDPG_10runs', 'NAF_10runs', 'ICNN_10runs', 'AE_CCEM_10runs', 'AE_Supervised_10runs']
# agents = ['AE_CCEM_10runs', 'AE_Supervised_10runs', 'AE_CCEM_with_2GA_10runs', 'AE_Supervised_with_2GA_10runs']
# agents = ['ActorExpert_no_target', 'ActorExpert_Plus','ActorExpert_Plus_no_target']
# agents = ['DDPG_10runs', 'NAF_10runs', 'AE_10runs', 'AE_Plus_10runs', 'PICNN_10runs', 'QT_OPT_10runs']

# agents = ['ActorExpert', 'ActorExpert_Plus', 'DDPG', 'NAF', 'QT_OPT', 'PICNN'] #,'DDPG', 'NAF', 'WireFitting', 'QT_OPT'] # 'WireFitting'
# agents = ['ActorExpert', 'ActorExpert_Plus', 'ActorCritic(expected_cem)', 'ActorCritic(expected_ll)']
# agents = ['ActorExpert', 'ActorExpert_Plus', 'ActorExpert_better_q', 'ActorExpert_Plus_better_q', 'DDPG', 'NAF', 'QT_OPT']
# agents = ['ActorExpert_Plus', 'ActorExpert_Plus_better_q', 'ActorExpert_Plus_og']
# agents = ['ActorExpert_ll_uniform_weighted', 'ActorExpert_ll_policy_weighted']
# agents =['ActorExpert', 'ActorCritic_sampled_nonorm', 'ActorCritic_sampled', 'ActorCritic']

agents = ['ActorExpert', 'ActorExpert_Plus', 'NAF', 'PICNN', 'QT_OPT_2modal'] #,'OptimalQ']
# agents = ['QT_OPT_1modal', 'QT_OPT_2modal', 'QT_OPT']

# agents = ['ActorExpert', 'ActorExpert_Plus', 'ActorCritic_sample_eval', 'DDPG']# , 'ActorExpert_samplingforeval', 'ActorCritic_samplingforeval']
# agents = ['ActorExpert', 'ActorExpert_with_uni']
# agents = ['ActorExpert_mean_mean', 'ActorExpert_mean_mean_100eval']
# agents=['QT_OPT', 'QT_OPT_new', 'QT_OPT_gmm']



# agents =['ActorCritic', 'ActorCritic_cem', 'ActorCritic_sampled', 'ActorCritic_cem_sampled']

# agents = ['ActorExpert_mean_mean', 'ActorExpert_ga_mean', 'ActorExpert_mean_ga', 'ActorExpert_ga_ga']

# agents =['ActorCritic_alphamodal', 'ActorCritic_equalmodal']

# LL separate
# agents = ['DDPG_10runs', 'NAF_10runs', 'AE_CCEM_separate_10runs', 'AE_Supervised_separate_10runs']


# HC combined
# agents = ['DDPG_10runs', 'NAF_10runs', 'ICNN_10runs', 'AE_CCEM_10runs', 'AE_Supervised_5runs']

# HC separate
# agents = ['DDPG_10runs', 'NAF_10runs', 'ICNN_10runs', 'AE_CCEM_separate_5runs', 'AE_Supervised_separate_10runs']

# Hopper combined
# agents = ['DDPG_10runs', 'NAF_10runs', 'ICNN_8runs', 'AE_CCEM_10runs', 'AE_Supervised_5runs']

# Hopper separate
# agents = ['DDPG_10runs', 'NAF_10runs', 'ICNN_8runs', 'AE_CCEM_separate_5runs', 'AE_Supervised_separate_10runs']

# PD Combined
# agents = ['DDPG_10runs', 'NAF_10runs', 'ICNN_10runs', 'AE_CCEM_10runs', 'AE_Supervised_10runs']

# PD Separate
# agents = ['DDPG_10runs', 'NAF_10runs', 'ICNN_10runs', 'AE_CCEM_separate_10runs', 'AE_Supervised_separate_10runs']


display_idx = 1 # IDX to determine whether to plot train_episode or eval_episode (0: Train, 1: Eval)
show_labels = True

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


agents_avg =[]
agents_se = []

max_length = 1

for ag in agents:

    avg = np.load(DIR + envname+'_'+ag+'_'+result_type[display_idx]+'_'+suffix+'_'+data_type[0]+'.npy')
    se = np.load(DIR + envname+'_'+ag+'_'+result_type[display_idx]+'_'+suffix+'_'+data_type[1]+'.npy')

    if max_length < len(avg):
        max_length = len(avg)

    agents_avg.append(avg)
    agents_se.append(se)


mpl.style.use('default')

# 
colors = [ '#377eb8', '#4daf4a', '#ff7f00',
                  '#f781bf', '#984ea3', '#999999','#a65628',
                  '#999999', '#e41a1c', '#dede00'] # ['b', 'c', 'r', 'm', 'g', 'y', 'k', 'w'] # 

plt.figure(figsize=(12,6))

#xmax = int(max_length)
xmax, ymin, ymax = get_xyrange(envname)

if xmax is None:
    xmax = int(max_length)


# Train Episode Rewards
if display_idx == 0:
    ylimt = (ymin[display_idx], ymax[display_idx])
    print("Train Episode plot not implemented")
    exit()
    pass

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

        loc_arr = np.array([0, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141])
        val_arr = np.array([0.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

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

        # val_arr = np.array([int(EVAL_INTERVAL_MIL_STEPS*1000*9), ])

        # print(float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax+9 - 1)))
        # print(int(TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS)+1)
        # print(np.linspace(0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax+9)), int(TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS)+1)[10:20])
        # exit()
        # val_arr = np.append(int(EVAL_INTERVAL_MIL_STEPS*1000*9), np.linspace(float(EVAL_INTERVAL_MIL_STEPS * 1e3), float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax+9)), int(TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS))[4:20])
        
        # loc_arr = np.append(0, opt_range[10::tick_interval])  # np.array(range(0,xmax,tick_interval))
        #
        # # val_arr = np.array(range(250, 1001, 125))
        # val_arr = np.array(range(int(EVAL_INTERVAL_MIL_STEPS*1000*9), 1001, int(EVAL_INTERVAL_MIL_STEPS*1000*tick_interval)) )
        

        # plt.xticks(np.append(1, opt_range[4::5]), np.append(2.0, np.linspace(float(EVAL_INTERVAL_MIL_STEPS * 1e3),
        #                                                                      float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax)), int(TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS))[4::5]))

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
        loc_arr = np.array([0, 41, 91, 141, 191])
        val_arr = np.array([45, 250, 500, 750, 1000])

        if show_labels:
            plt.xticks(loc_arr, val_arr)
            plt.yticks([20, 40, 60, 80, 100, 120], [20, 40, 60, 80, 100, 120])
        else:
            plt.xticks(loc_arr, [])
            plt.yticks([20, 40, 60, 80, 100, 120], [])  # [0, 2000, 4000, 6000, 7000])

    else:
        plt.xticks(opt_range[::50], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax-1)), int(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS)+1)[::50])


plt.xlim(xlimt)
plt.ylim(ylimt)

handle_arr=[]  
for idx in range(len(agents)):
    pad_length = len(opt_range) - len(agents_avg[idx][:xmax])
    plt.fill_between(opt_range, np.append(agents_avg[idx][:xmax] - agents_se[idx][:xmax], np.zeros(pad_length) + np.nan), np.append(agents_avg[idx][:xmax] + agents_se[idx][:xmax], np.zeros(pad_length) + np.nan), alpha = 0.2, facecolor=colors[idx])
    handle, = plt.plot(opt_range, np.append(agents_avg[idx][:xmax], np.zeros(pad_length) + np.nan), colors[idx], linewidth=1.2, label=agents[idx])
    handle_arr.append(handle)
    
if show_labels:
    plt.legend(handle_arr, agents)
plt.show()




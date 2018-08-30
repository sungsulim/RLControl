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

# LL combined
# agents = ['DDPG_10runs', 'NAF_10runs', 'AE_CCEM_9runs', 'AE_Supervised_10runs']

# LL separate
agents = ['DDPG_10runs', 'NAF_10runs', 'AE_CCEM_separate_10runs', 'AE_Supervised_separate_10runs']


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

colors = ['b', 'g', 'r', 'c', 'm', 'k']

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

    plt.title(envname)
    plt.xlabel('Training steps (per 1000 steps)')
    plt.ylabel("Cum. Reward per episode").set_rotation(90)

    opt_range = range(0, xmax) 
    xlimt = (0, xmax-1)

    # opt_range = range(1, xmax+1)
    # xlimt = (1, xmax)
    
    

    if envname == 'Pendulum-v0':
        # plt.axhline(y=-150, linewidth=1.0, linestyle='--', color='darkslategrey', label='-150')
        plt.xticks(opt_range[::20], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax-1)), min(int(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS)+1, xmax))[::20])

    # elif envname == 'HalfCheetah-v2' or envname == 'Hopper-v2':
    #     opt_range = range(1, xmax+1)
    #     xlimt = (1, xmax)

    #     x_tick_interval = 10

    #     ## HARD CODED!! For HalfCheetah and Hopper
    #     loc_arr = np.array([1, 11, 21, 31, 41, 51, 61, 71, 81, 91])
    #     val_arr = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    elif envname == 'LunarLanderContinuous-v2':
        # print('xmax', xmax)
        # print('len',np.shape(agents_avg))
        # exit()
        tick_interval = 30
        loc_arr = np.array(range(0,xmax,tick_interval))
        
        # val_arr = np.array(range(250, 1001, 125)) 
        val_arr = np.array(range(int(EVAL_INTERVAL_MIL_STEPS*1000*9), 1001, int(EVAL_INTERVAL_MIL_STEPS*1000*tick_interval)) )

        plt.xticks(loc_arr, val_arr)

        plt.yticks([-150, -100, -50, 0, 50, 100, 150, 200, 250],[-150, -100, -50, 0, 50, 100, 150, 200, 250])

    else:
        plt.xticks(opt_range[::50], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax-1)), int(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS)+1)[::50])


plt.xlim(xlimt)
plt.ylim(ylimt)

handle_arr=[]  
for idx in range(len(agents)):
    pad_length = len(opt_range) - len(agents_avg[idx][:xmax])
    plt.fill_between(opt_range, np.append(agents_avg[idx][:xmax] - agents_se[idx][:xmax], np.zeros(pad_length) + np.nan), np.append(agents_avg[idx][:xmax] + agents_se[idx][:xmax], np.zeros(pad_length) + np.nan), alpha = 0.2, facecolor=colors[idx])
    handle, = plt.plot(opt_range, np.append(agents_avg[idx][:xmax], np.zeros(pad_length) + np.nan), colors[idx], linewidth=1.0, label=agents[idx])
    handle_arr.append(handle)
    
 
plt.legend(handle_arr, agents)
plt.show()




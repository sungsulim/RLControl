import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
from pathlib import Path
import json

############## USAGE
# You should be in actiongeneral/python/results when running the script

# python3 ../plot_scripts/plot_agent_comparison.py ../jsonfiles/environment/Pendulum-v0.json DIR_of_npy 

# example: python3 ../plot_scripts/plot_agent_comparison.py../jsonfiles/environment/Pendulum-v0.json  mergedPendulum-v0results 

# This plot_script is used for comparing different agents (using the saved .npy from plot_custom_new.py)
# You should place the .npy of different agents in one directory (DIR_of_npy)



####### CONFIG ####
# This should be the agent name between Pendulum-v0_(   )_EvalEpisode_BestResult_avg.npy
# You can change the agent name between ( ) to be anything, as long as other format is same.

# PD
agents = ['CrossEntropy_CriticAssistant','Simple_CriticAssistant', 'NAF', 'Wire_fitting']
#agents = ['CEM_hydra_multimodal_setting2', 'CriticAssistant_hydra_setting2', 'NAF_batch_setting1', 'NAF_layer', 'Wire_fitting'] 

# LL
#agents = ['CrossEntropy_CriticAssistant', 'Simple_CriticAssistant', 'NAF']
# agents = ['CEM_hydra_multimodal_setting3', 'CriticAssistant_hydra_setting2', 'NAF_layer_setting7', 'NAF_batch_setting3', 'NAF_none_setting8']
display_idx = 1 # IDX to determine whether to plot train_episode or eval_episode (0: Train, 1: Eval)


# HC
#agents = ['CrossEntropy_CriticAssistant','Simple_CriticAssistant', 'NAF']
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


xmax = int(max_length)


if envname == 'HalfCheetah-v2':

    # Train Episode Rewards
    if display_idx == 0:
        pass

    # EvalEpisode Rewards
    elif display_idx == 1:
        plt.title('Eval Episode Rewards: '+str(agents))

        ylimt = (-500, 4000)
        plt.xlabel('Training steps (per 1000 steps)')
        plt.ylabel("Cum. Reward per episode").set_rotation(90)
        opt_range = range(1, xmax+1)
        xlimt = (1, xmax)

        x_tick_interval = 5
        #plt.xticks(opt_range, np.linspace(float(EVAL_INTERVAL_MIL_STEPS * 1e3), float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax)), TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS))


elif envname == 'LunarLanderContinuous-v2':

    # Train Episode Rewards
    if display_idx == 0:
        plt.title('Train Episode Rewards: '+str(agents))

        ylimt = (-400, 250)
        plt.xlabel('Episodes')
        plt.ylabel("Cum. Reward per episode").set_rotation(90)
        opt_range = range(0, xmax)
        xlimt = (0, xmax)

    # EvalEpisode Rewards
    elif display_idx == 1:
        plt.title('Eval Episode Rewards: '+str(agents))

        ylimt = (-150, 250)
        plt.xlabel('Training steps (per 1000 steps)')
        plt.ylabel("Cum. Reward per episode").set_rotation(90)
        opt_range = range(1, xmax+1)
        xlimt = (1, xmax)

        x_tick_interval = 5
        #plt.xticks(opt_range, np.linspace(float(EVAL_INTERVAL_MIL_STEPS * 1e3), float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax)), TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS))

elif envname == 'MountainCarContinuous-v0':

    # Train Episode Rewards
    if display_idx == 0:
        plt.title('Train Episode Rewards: '+str(agents))

        ylimt = (-50, 100)
        plt.xlabel('Episodes')
        plt.ylabel("Reward").set_rotation(90)
        opt_range = range(0, xmax)
        xlimt = (0, xmax)

    # EvalEpisode Rewards
    elif display_idx == 1:
        plt.title('Eval Episode Rewards: '+str(agents))

        ylimt = (-50, 100)
        plt.xlabel('Training steps (per 1000 steps)')
        plt.ylabel("Reward").set_rotation(90)
        opt_range = range(1, xmax+1)
        xlimt = (1, xmax)
        #plt.xticks(opt_range[9::10], np.linspace(float(EVAL_INTERVAL_MIL_STEPS * 1e3), float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax)), TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS)[9::10])


elif envname == 'Pendulum-v0':

    # Train Episode Rewards
    if display_idx == 0:
        plt.title('Train Episode Rewards: '+str(agents))

        ylimt = (-1800, 0)
        plt.xlabel('Episodes')
        plt.ylabel("Cum. Reward per episode").set_rotation(90)
        opt_range = range(0, xmax)
        xlimt = (0, xmax)

    # EvalEpisode Rewards
    elif display_idx == 1:
        plt.title('Eval Episode Rewards: '+str(agents))
        #plt.title('Comparison between Value-based methods')
        ylimt = (-1800, 0)
        plt.xlabel('Training steps (per 1000 steps)')
        plt.ylabel("Cum. Reward per episode").set_rotation(90)
        opt_range = range(1, xmax+1)
        xlimt = (1, xmax)

        x_tick_interval = 5

plt.xticks(np.append(1, opt_range[x_tick_interval-1::x_tick_interval]), np.append(EVAL_INTERVAL_MIL_STEPS * 1e3, np.linspace(float(EVAL_INTERVAL_MIL_STEPS * 1e3), float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax)), float(xmax))[x_tick_interval-1::x_tick_interval]))

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




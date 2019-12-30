import numpy as np
import re
import glob
from pathlib import Path
from shutil import copyfile
import os
import sys
import json
import statistics

from utils import get_agent_parse_info
from collections import OrderedDict
from shutil import copyfile

# old usage:
# python3 /Users/sungsulim/Documents/projects/ActorExpert/RLControl/plot_scripts/mergefile_new_ma.py
# /Users/sungsulim/Documents/projects/ActorExpert/RLControl/jsonfiles/environment/${ENV_NAME}.json
# ${ENV_NAME}results $NUM_SETTINGS $NUM_RUNS $AGENT_NAME

# new usage:
# python3 /Users/sungsulim/Documents/projects/ActorExpert/RLControl/plot_scripts/merge_results_refactored.py
# $RESULT_DIR $ROOT_LOC $ENV_NAME $AGENT_NAME $NUM_RUNS $USE_MOVING_AVG

###### SETTINGS ######
moving_avg_window = 10

######################

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


if len(sys.argv)!=7:
    print('Incorrect Input')
    print('type: merge_results_refactored.py RESULT_DIR ROOT_LOC ENV_NAME AGENT_NMAE NUM_RUNS USE_MOVING_AVG')
    exit(0)

root_dir = str(sys.argv[2])
env_name = str(sys.argv[3])
agent_name = str(sys.argv[4])
num_runs = int(sys.argv[5])
use_moving_avg = eval(sys.argv[6])
if not isinstance(use_moving_avg, bool): raise TypeError('use_moving_avg should be a valid bool')

result_dir = str(sys.argv[1])



# load env info
with open('{}/jsonfiles/environment/{}.json'.format(root_dir, env_name), 'r') as env_dat:
    env_json = json.load(env_dat)

TOTAL_MIL_STEPS = env_json['TotalMilSteps']
EVAL_INTERVAL_MIL_STEPS = env_json['EvalIntervalMilSteps']
EVAL_EPISODES = env_json['EvalEpisodes']

# location of results and future merged results
store_dir = '{}/{}results/'.format(result_dir, env_name)
merged_dir = '{}/merged{}results'.format(result_dir, env_name)

if not os.path.exists(merged_dir):
    os.makedirs(merged_dir)
merged_dir += '/'

# get num_settings from agent_json
agent_json_name = '{}_{}_agent_Params.json'.format(env_name, agent_name)
with open('{}/{}'.format(store_dir, agent_json_name), 'r') as agent_dat:
    agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)
_, _, _, _, num_settings = get_agent_parse_info(agent_json)

# copy json to mergedresult folder
try:
    copyfile('{}/{}'.format(store_dir, agent_json_name), '{}/{}'.format(merged_dir, agent_json_name))
    print("Copied... {} to merged dir".format(agent_json_name))
except:
    raise ValueError("Json not copied properly")

print("Environment: {}".format(env_name))
print("Agent: {}".format(agent_name))
print("Num settings: {}".format(num_settings))
print("Num runs: {}".format(num_runs))
print("Use moving avg: {}".format(use_moving_avg))


suffix = ['_EpisodeRewardsLC.txt','_EvalEpisodeMeanRewardsLC.txt','_EvalEpisodeStdRewardsLC.txt','_Params.txt']
save_suffix = ['_TrainEpisodeMeanRewardsLC.txt','_TrainEpisodeStdRewardsLC.txt','_EvalEpisodeMeanRewardsLC.txt','_EvalEpisodeStdRewardsLC.txt','_Params.txt']

missingindexes = []
train_mean_rewards = []
train_std_rewards = []
eval_mean_rewards = []
eval_std_rewards = []

params = []
params_fn = None

eval_lc_length = int(TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS) + 1
if use_moving_avg:
    eval_lc_length = eval_lc_length - (moving_avg_window-1)

max_median_length = 1

# for each setting
for setting_num in range(num_settings):
    run_non_count = 0

    train_lc_arr = []
    train_lc_length_arr = []
    eval_mean_lc_arr = []
    
    # for each run
    for run_num in range(num_runs):
        train_rewards_filename = store_dir + env_name + '_' + agent_name + '_setting_' + str(setting_num) + '_run_' + str(run_num) + suffix[0]
        eval_mean_rewards_filename = store_dir + env_name + '_' + agent_name + '_setting_' + str(setting_num) + '_run_' + str(run_num) + suffix[1]
        #eval_std_rewards_filename = storedir+envname+'_'+agent_name+'_setting_'+str(setting_num)+'_run_'+str(run_num)+suffix[2]
        
        # skip if file does not exist
        if not Path(train_rewards_filename).exists():
            run_non_count += 1

            # add dummy
            lc_0 = np.zeros(1) + np.nan # will be padded
            lc_1 = np.zeros(eval_lc_length) + np.nan

            train_lc_arr.append(lc_0)
            eval_mean_lc_arr.append(lc_1)

            print(' setting ' + train_rewards_filename + ' does not exist')
            missingindexes.append(num_settings * run_num + setting_num)
            continue

        lc_0 = np.loadtxt(train_rewards_filename, delimiter=',')
        lc_1 = np.loadtxt(eval_mean_rewards_filename, delimiter=',') # [:eval_lc_length+9] temporary solution for Pendulum-v0

        # compute moving window
        if use_moving_avg:
            lc_0 = movingaverage(lc_0, moving_avg_window)
            lc_1 = movingaverage(lc_1, moving_avg_window)

        train_lc_arr.append(lc_0)
        train_lc_length_arr.append(len(lc_0))
        eval_mean_lc_arr.append(lc_1)

    # find median train ep length (truncate or pad with nan)
    try:
        num_train_length = int(statistics.median(train_lc_length_arr))
    except:
        num_train_length = 0
    

    if num_train_length > max_median_length:
        max_median_length = num_train_length

    for i in range(len(train_lc_arr)):

        # truncate
        if len(train_lc_arr[i]) > num_train_length:
            train_lc_arr[i] = train_lc_arr[i][:num_train_length]

        # pad with nan
        elif len(train_lc_arr[i]) < num_train_length:
            pad_length = num_train_length - len(train_lc_arr[i])
            train_lc_arr[i] = np.append(train_lc_arr[i], np.zeros(pad_length) + np.nan)

    train_lc_arr = np.array(train_lc_arr)
    eval_mean_lc_arr = np.array(eval_mean_lc_arr)


    if run_non_count == num_runs:
        print('setting ' + str(setting_num) + ' does not exist')
        print(np.shape(train_lc_arr), train_lc_arr)
        print(np.shape(eval_mean_lc_arr), eval_mean_lc_arr)
        # exit() ## Perhaps continue?? TODO

    #### Need to have same size
    train_mean_rewards.append(np.nanmean(train_lc_arr, axis=0))
    train_std_rewards.append(np.nanstd(train_lc_arr, axis=0))
    


    # TODO: process eval_mean_lc_arr, eval_std_lc together and append to eval_mean_rewards, eval_std_rewards
    eval_combined_mean_lc = np.nanmean(eval_mean_lc_arr, axis=0)
    eval_mean_rewards.append(eval_combined_mean_lc)
    eval_std_rewards.append(np.nanstd(eval_mean_lc_arr, axis=0))
    
    #std_mean = np.nanmean(np.square(eval_std_lc), axis=0)
    #diff_squared_mean = np.nanmean(np.square(eval_mean_lc_arr - eval_combined_mean_lc), axis=0)

    #eval_std_rewards.append(np.sqrt(std_mean + diff_squared_mean))
    # print(np.shape(np.nanmean(train_lc_arr, axis=0)))
    # print(np.shape(np.nanmean(eval_mean_lc_arr, axis=0)))
    # input()

    '''read in params file'''
    paramfile = store_dir + env_name + '_' + agent_name + '_setting_' + str(setting_num) + '_run_*' + suffix[3]
    files = glob.glob(paramfile)
    if len(files)<1:
        continue
    onefile = files[0]
    newfilename = re.sub(store_dir, '', files[0])
    newfilename = re.sub('_setting_[0-9]+_','_',newfilename)
    newfilename = merged_dir + re.sub('_run_[0-9]+_', '_', newfilename)
    params_fn = newfilename
    setting_params = np.loadtxt(onefile, delimiter=',', dtype='str')
    setting_params = np.insert(setting_params, 0, setting_num) 
    #print params_fn
    params.append(setting_params)
params = np.array(params)

for idx, item in enumerate(train_mean_rewards):
    if len(item) < max_median_length:
        # pad with nan
        pad_length = max_median_length - len(item)
        train_mean_rewards[idx] = np.append(train_mean_rewards[idx], np.zeros(pad_length) + np.nan)
        train_std_rewards[idx] = np.append(train_std_rewards[idx], np.zeros(pad_length) + np.nan)

print("max train median length: ", max_median_length)
# print(train_mean_rewards[0])
# print(eval_mean_rewards[0])
# print(np.shape(train_mean_rewards), np.shape(eval_mean_rewards))
# input()
eval_mean_rewards = np.array(eval_mean_rewards)
eval_std_rewards = np.array(eval_std_rewards)

allres = [train_mean_rewards, train_std_rewards, eval_mean_rewards, eval_std_rewards, params]
for i in range(len(save_suffix)):
    name = merged_dir + env_name + '_' + agent_name + save_suffix[i]
    if i == 4:
        name = params_fn

    print('Saving...' + name)
    np.savetxt(name, allres[i], fmt='%s', delimiter=',')


print('missed indexes are:  -- - - - - - - - - --')
missed = ''
for missid in missingindexes:
   missed += (str(missid)+',')
print(missed)



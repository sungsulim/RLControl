import numpy as np
import re
import glob
from pathlib import Path
from shutil import copyfile
import os
import sys
import json
import statistics

#### USAGE
# You should be in actiongeneral/python/results when running the script

# python3 ../plot_scripts/mergefile.py  ENV.json DIR_RAW_RESULT(without / at the end)  NUM_SETTINGS   NUM_RUNS   AGENT_NAME   


# example: python3 ../plot_scripts/mergefile.py ../jsonfile/environment/LunarLanderContinuous-v2.json LunarLanderContinuous-v2results 9 5 NAF 
# This will generate mergedRESULT Directory (i.e. mergedLunarLanderContinuous-v2results)


if len(sys.argv)!=6:
    print('Incorrect Input')
    print('type: mergefile_custom_new.py env_json srcdir nparams nruns algname ')
    exit(0)

env_filename = str(sys.argv[1])
with open(env_filename, 'r') as env_dat:
    env_json = json.load(env_dat)


NUM_SETTINGS = int(sys.argv[3])
NUM_RUNS = int(sys.argv[4]) 

TOTAL_MIL_STEPS = env_json['TotalMilSteps']
EVAL_INTERVAL_MIL_STEPS = env_json['EvalIntervalMilSteps']
EVAL_EPISODES = env_json['EvalEpisodes']


suffix = ['_EpisodeRewardsLC.txt','_EvalEpisodeMeanRewardsLC.txt','_EvalEpisodeStdRewardsLC.txt','_Params.txt']
save_suffix = ['_TrainEpisodeMeanRewardsLC.txt','_TrainEpisodeStdRewardsLC.txt','_EvalEpisodeMeanRewardsLC.txt','_EvalEpisodeStdRewardsLC.txt','_Params.txt']



storedir = sys.argv[2]+'/'
envname = sys.argv[2].replace('results','')


cleaneddir = 'merged' + sys.argv[2]
if os.path.exists(cleaneddir)==False:
    os.makedirs(cleaneddir)
cleaneddir += '/'


agent_name = sys.argv[5]

missingindexes = []
train_mean_rewards = []
train_std_rewards = []
eval_mean_rewards = []
eval_std_rewards = []

params = []
params_fn = None


eval_lc_length = int(TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS) + 1

max_median_length = 1

# for each setting
for setting_num in range(NUM_SETTINGS):
    run_non_count = 0

    train_lc_arr = []
    train_lc_length_arr = []
    eval_mean_lc_arr = []
    
    # for each run
    for run_num in range(NUM_RUNS):
        train_rewards_filename = storedir+envname+'_'+agent_name+'_setting_'+str(setting_num)+'_run_'+str(run_num)+suffix[0]
        eval_mean_rewards_filename = storedir+envname+'_'+agent_name+'_setting_'+str(setting_num)+'_run_'+str(run_num)+suffix[1]
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
            missingindexes.append(NUM_SETTINGS*run_num+setting_num)
            continue

        lc_0 = np.loadtxt(train_rewards_filename, delimiter=',')
        lc_1 = np.loadtxt(eval_mean_rewards_filename, delimiter=',')


        train_lc_arr.append(lc_0)
        train_lc_length_arr.append(len(lc_0))
        eval_mean_lc_arr.append(lc_1)


    # find median train ep length (truncate or pad with nan)
    num_train_length = statistics.median(train_lc_length_arr)
    

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


    if run_non_count == NUM_RUNS:
        print('setting ' + str(setting_num) + ' does not exist')
        exit() ## Perhaps continue?? TODO

    #### Need to have same size
    train_mean_rewards.append(np.nanmean(train_lc_arr, axis=0))
    train_std_rewards.append(np.nanstd(train_lc_arr, axis=0))
    


    # TODO: process eval_mean_lc_arr, eval_std_lc together and append to eval_mean_rewards, eval_std_rewards
    # print(eval_mean_lc_arr)
    # exit()
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
    paramfile = storedir+envname+'_'+agent_name+'_setting_'+str(setting_num)+'_run_*' + suffix[3]
    files = glob.glob(paramfile)
    if len(files)<1:
        continue
    onefile = files[0]
    newfilename = re.sub(storedir, '', files[0])
    newfilename = re.sub('_setting_[0-9]+_','_',newfilename)
    newfilename = cleaneddir + re.sub('_run_[0-9]+_', '_', newfilename)
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
allres = [train_mean_rewards, train_std_rewards, eval_mean_rewards, eval_std_rewards, params]
for i in range(len(save_suffix)):
    name = cleaneddir+envname+'_'+agent_name+save_suffix[i]
    if i == 4:
        name = params_fn

    print('Saving...' + name)
    np.savetxt(name, allres[i], fmt='%s', delimiter=',')


print('missed indexes are:  -- - - - - - - - - --')
missed = ''
for missid in missingindexes:
   missed += (str(missid)+',')
print(missed)



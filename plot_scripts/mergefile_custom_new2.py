import numpy as np
import re
import glob
from pathlib import Path
from shutil import copyfile
import os
import sys
import json

#### USAGE
# You should be in actiongeneral/python/results when running the script

# python3 ../plot_scripts/mergefile_custom_new2.py   DIR_RAW_RESULT(without / at the end)  NUM_SETTINGS   NUM_RUNS   AGENT_NAME   ENV.json


# example: python3 ../plot_scripts/mergefile_custom_new2.py LunarLanderContinuous-v2results 9 5 NAF ../jsonfile/environment/LunarLanderContinuous-v2.json
# This will generate mergedRESULT Directory (i.e. mergedLunarLanderContinuous-v2results)


if len(sys.argv)!=6:
    print('Incorrect Input')
    print('type: mergefile_custom_new.py srcdir nparams nruns algname env_json')
    exit(0)

env_filename = str(sys.argv[5])
with open(env_filename, 'r') as env_dat:
    env_json = json.load(env_dat)


NUM_SETTINGS = int(sys.argv[2])
NUM_RUNS = int(sys.argv[3]) 

TOTAL_MIL_STEPS = env_json['TotalMilSteps']
EVAL_INTERVAL_MIL_STEPS = env_json['EvalIntervalMilSteps']
EVAL_EPISODES = env_json['EvalEpisodes']


suffix = ['_EpisodeRewardsLC.txt','_EvalEpisodeMeanRewardsLC.txt','_EvalEpisodeStdRewardsLC.txt','_Params.txt']
save_suffix = ['_TrainEpisodeMeanRewardsLC.txt','_TrainEpisodeStdRewardsLC.txt','_EvalEpisodeMeanRewardsLC.txt','_EvalEpisodeStdRewardsLC.txt','_Params.txt']



storedir = sys.argv[1]+'/'
envname = sys.argv[1].replace('results','')


cleaneddir = 'merged' + sys.argv[1]
if os.path.exists(cleaneddir)==False:
    os.makedirs(cleaneddir)
cleaneddir += '/'


agent_name = sys.argv[4]

missingindexes = []
train_mean_rewards = []
train_std_rewards = []
eval_mean_rewards = []
eval_std_rewards = []

params = []
params_fn = None


lc_1_2_length = int(TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS)

cur_max_length = 1
for setting_num in range(NUM_SETTINGS):
    run_non_count = 0

    train_lc = []
    eval_mean_lc = []
    #eval_std_lc = []

    
    for run_num in range(NUM_RUNS):
        train_rewards_filename = storedir+envname+'_'+agent_name+'_setting_'+str(setting_num)+'_run_'+str(run_num)+suffix[0]
        eval_mean_rewards_filename = storedir+envname+'_'+agent_name+'_setting_'+str(setting_num)+'_run_'+str(run_num)+suffix[1]
        #eval_std_rewards_filename = storedir+envname+'_'+agent_name+'_setting_'+str(setting_num)+'_run_'+str(run_num)+suffix[2]
        
        # skip if file does not exist
        if not Path(train_rewards_filename).exists():
            run_non_count += 1

            # add dummy
            lc_0 = np.zeros(cur_max_length) + np.nan
            lc_1 = np.zeros(lc_1_2_length) + np.nan
            #lc_2 = np.zeros(lc_1_2_length) + np.nan

            train_lc.append(lc_0)
            eval_mean_lc.append(lc_1)
            #eval_std_lc.append(lc_2)

            print(' setting ' + train_rewards_filename + ' does not exist')
            missingindexes.append(NUM_SETTINGS*run_num+setting_num)
            continue

        lc_0 = np.loadtxt(train_rewards_filename, delimiter=',')
        lc_1 = np.loadtxt(eval_mean_rewards_filename, delimiter=',')
        #lc_2 = np.loadtxt(eval_std_rewards_filename, delimiter=',')

        if len(lc_0) > cur_max_length:
            cur_max_length = len(lc_0)
            # pad previous results
            for idx, lc in enumerate(train_lc):
                pad_length = cur_max_length - len(lc)
                train_lc[idx] = np.append(lc, np.zeros(pad_length) + np.nan)


        elif len(lc_0) < cur_max_length:
            # pad current result
            pad_length = cur_max_length - len(lc_0)
            lc_0 = np.append(lc_0, np.zeros(pad_length) + np.nan)


        train_lc.append(lc_0)
        eval_mean_lc.append(lc_1)
        #eval_std_lc.append(lc_2)

    train_lc = np.array(train_lc)
    eval_mean_lc = np.array(eval_mean_lc)
    #eval_std_lc = np.array(eval_std_lc)


    if run_non_count == NUM_RUNS:
        print('setting ' + str(setting_num) + ' does not exist')
        exit() ## Perhaps continue?? TODO

    #### TODO:::: Need to have same size
    train_mean_rewards.append(np.nanmean(train_lc, axis=0))
    train_std_rewards.append(np.nanstd(train_lc, axis=0))


    # TODO: process eval_mean_lc, eval_std_lc together and append to eval_mean_rewards, eval_std_rewards
    # print(eval_mean_lc)
    # exit()
    eval_combined_mean_lc = np.nanmean(eval_mean_lc, axis=0)
    eval_mean_rewards.append(eval_combined_mean_lc)
    eval_std_rewards.append(np.nanstd(eval_mean_lc, axis=0))
    
    #std_mean = np.nanmean(np.square(eval_std_lc), axis=0)
    #diff_squared_mean = np.nanmean(np.square(eval_mean_lc - eval_combined_mean_lc), axis=0)

    #eval_std_rewards.append(np.sqrt(std_mean + diff_squared_mean))


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
    if len(item) < cur_max_length:
        # pad with nan
        pad_length = cur_max_length - len(item)
        train_mean_rewards[idx] = np.append(train_mean_rewards[idx], np.zeros(pad_length) + np.nan)
        train_std_rewards[idx] = np.append(train_std_rewards[idx], np.zeros(pad_length) + np.nan)


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



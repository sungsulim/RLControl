import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
from pathlib import Path
import json

############## USAGE
# You should be in actiongeneral/python/results when running the script

# python3 ../plot_scripts/plot_custom_new.py    mergedDIR    NUM_SETTINGS    Agent_name   ../jsonfiles/environment/Pendulum-v0.json

#example: python3 ../plot_scripts/plot_custom_new.py mergedPendulum-v0results 2 CriticAssistant ../jsonfiles/environment/Pendulum-v0.json

# This will generate the plots of the best setting among the merged results for TRAIN and EVAL.
# Note that the best setting for TRAIN could be different for best setting for EVAL.
# Also, if the number of steps for each episode is different for each run, train_plot could be a bit weird.. (because for each run it would have different num of episodes)

# If you want to view a specific setting, see below.

# The script will also generate .npy for the best setting, so you can compare the result with other agents.
# If you want to save .npy for a specific setting, set only one idx in selected_idx array. 
######################




### CONFIG BEFORE RUNNING ###
# Use if you want to plot specific settings, put the idx of the setting below.
# You can also see *_Params.txt to see the idx for each setting.

selected_idx= [1]
selected_type = selected_idx[:] # This will be the labels for those idx.
selected_type = ['1e-3, 1e-2']
# Example: selected_type = ['NAF', 'Wire_fitting']
##############################




if len(sys.argv)!=5:
    print('Incorrect Input')
    print('type: plot_custom_new.py merged_dir nruns algname env_json')
    exit(0)

# Stored Directory
storedir = str(sys.argv[1])+'/'

# Environment Name
envname = storedir.replace('merged','').replace('results/','')
print('Environment: ' + envname)

# Num Runs
num_runs = int(sys.argv[2])

# Agent
agent = str(sys.argv[3])
print('Agent: ' + agent)

# Num Episode, Num Steps per episode
env_filename = str(sys.argv[4])
with open(env_filename, 'r') as env_dat:
    env_json = json.load(env_dat)

TOTAL_MIL_STEPS = env_json['TotalMilSteps']
EVAL_INTERVAL_MIL_STEPS = env_json['EvalIntervalMilSteps']
EVAL_EPISODES = env_json['EvalEpisodes']

if envname == 'HalfCheetah-v2':
    ymin = [-500, -500]
    ymax = [4000, 4000]

elif envname == 'LunarLanderContinuous-v2':
    ymin = [-250, -250]
    ymax = [250, 250]

elif envname == 'MountainCarContinuous-v0':
    ymin = [-50, -50]
    ymax = [100, 100]

elif envname == 'HalfCheetahBulletEnv-v0':
    ymin = [-100, -100]
    ymax = [1000, 1000]

elif envname == 'Pendulum-v0':
    ymin = [-1600, -1600]
    ymax = [0, 0]
else:
    print("Environment plot setting not found!")
    exit()



#colors = ['b','r','g','c']


# Plot type
result_type = ['TrainEpisode', 'EvalEpisode']



for result_idx, result in enumerate(result_type):

    lcfilename = storedir + envname + '_' + agent + '_'+ result +'MeanRewardsLC.txt'
    print('Reading lcfilename.. ' + lcfilename)
    lc = np.loadtxt(lcfilename, delimiter=',')
    
    stdfilename = storedir + envname + '_' + agent + '_' + result + 'StdRewardsLC.txt'
    print('Reading stdfilename.. ' + stdfilename)
    lcstd = np.loadtxt(stdfilename, delimiter=',')

    paramfile = storedir+envname+'_'+agent+'_*'+'_Params.txt'
    print('Reading paramfile.. ' + paramfile)

    files = glob.glob(paramfile)
    params = np.loadtxt(files[0], delimiter=',', dtype='str')

    xmax = np.shape(lc)[-1] #len(lc)#len(lc[0])


    #title = result+": " + agent + ', ' + str(num_runs) + ' runs'
    plt.figure(figsize=(12,6))
    #plt.title(title)
    
    
    plt.legend(loc="best")
    

    if result == 'TrainEpisode':
        plt.xlabel('Episodes')
        opt_range = range(0, xmax)
        xlimt = (0, xmax)
        num_samples = num_runs

    elif result == 'EvalEpisode':
        plt.xlabel('Training Steps (per 1000 steps)')
        opt_range = range(1, xmax+1) 

        xlimt = (1, xmax)
        #print(float(EVAL_INTERVAL_MIL_STEPS * 1e3),float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax)))
        #exit() 
        
        # print(opt_range[9::10])
        # exit()
        plt.xticks(np.append(1,opt_range[4::5]), np.append(2.0, np.linspace(float(EVAL_INTERVAL_MIL_STEPS * 1e3), float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax)), int(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS))[4::5]))
        num_samples = num_runs # * EVAL_EPISODES

        #print(range(int(EVAL_INTERVAL_MIL_STEPS * 1000), int(EVAL_INTERVAL_MIL_STEPS * 1000 * (xmax+1)), int(EVAL_INTERVAL_MIL_STEPS *1000)))
        #exit()

    h = plt.ylabel("Cum. Reward per episode")
    h.set_rotation(90)


    ylimt = (ymin[result_idx], ymax[result_idx])
    plt.ylim(ylimt)
    plt.xlim(xlimt)

    # Plot selected idx
    if len(selected_idx) != 0 :

        handle_arr=[]
        for idx, item in enumerate(selected_idx):

            # only one line
            if len(np.shape(lc))== 1:
                draw_lc = lc[:xmax]
                draw_lcse = lcstd[:xmax] /np.sqrt(num_samples)
            else:
                draw_lc = lc[item,:xmax]
                draw_lcse = lcstd[item,:xmax] /np.sqrt(num_samples)

            print('drawing.. '+ agent + ' setting: '+str(item), np.sum(draw_lc))

            plt.fill_between(opt_range, draw_lc - draw_lcse, draw_lc + draw_lcse, alpha = 0.2)#, facecolor=colors[idx])
            #handle, = plt.plot(opt_range, draw_lc, colors[idx], linewidth=1.0) 
            handle, = plt.plot(opt_range, draw_lc, linewidth=1.0) 
            handle_arr.append(handle)

        legend_arr = []
        for i in range(len(selected_idx)):
            legend_arr.append(str(selected_type[i]))
        plt.legend(handle_arr, legend_arr)
        plt.show()
        plt.close()

        savelc = draw_lc
        savelcse = draw_lcse

    else:
        # if only one line in _StepLC.txt)
        if not np.shape(lc[0]):
            bestlc = lc[:xmax]
            lcse = lcstd[:xmax]/np.sqrt(num_samples)

            print('The best param setting of '+ agent + ' is ')
            print(params[:])
        else:

            if result == 'TrainEpisode':
                #print(np.shape(lc))
                #BestInd = np.argmax(np.sum(lc[:,xmax], axis = 1))
                BestInd = np.argmax(np.sum(lc, axis = 1))

            elif result == 'EvalEpisode':
                #BestInd = np.argmax(lc[:,[xmax[-1]]])
                #BestInd = np.argmax(np.sum(lc[:,xmax], axis = 1))
                BestInd = np.argmax(np.sum(lc, axis = 1))

            else:
                print("Result Type Error")
                exit()
            # print(np.sum(lc[:,opt_range], axis = 1))
            # exit()            
            bestlc = lc[BestInd,:xmax]
            lcse = lcstd[BestInd,:xmax]/np.sqrt(num_samples)
            print('The best param setting of '+ agent + ' is ')
            print(params[BestInd,:])
            assert(BestInd == float(params[BestInd,0]))


        legends = [agent + ', ' + str(num_runs) + ' runs']

        plt.fill_between(opt_range, bestlc - lcse, bestlc + lcse, alpha = 0.2)#, facecolor=colors[0])
        #plt.plot(opt_range, bestlc, colors[0], linewidth=1.0, label=legends)
        plt.plot(opt_range, bestlc, linewidth=1.0, label=legends)
        plt.show()
        plt.close()
        savelc = bestlc
        savelcse = lcse

    if len(selected_idx) == 0 or len(selected_idx) == 1:
        # save the best params
        savefilename_avg = storedir + envname + '_' + agent + '_'+ result +'_BestResult_avg.npy'
        savefilename_se = storedir + envname + '_' + agent + '_'+ result +'_BestResult_se.npy'
        np.save(savefilename_avg, savelc)
        np.save(savefilename_se, savelcse)

plt.close()

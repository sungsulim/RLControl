# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import glob
import sys
from pathlib import Path
import json

############## USAGE
# You should be in actiongeneral/python/results when running the script

# python3 ../plot_scripts/plot_agent.py  ../jsonfiles/environment/Pendulum-v0.json  mergedDIR    NUM_SETTINGS    Agent_name   

#example: python3 ../plot_scripts/plot_agent.py ../jsonfiles/environment/Pendulum-v0.json mergedPendulum-v0results 2 CriticAssistant 

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

eval_last_N = False
last_N = 50

selected_idx = [] # range(0 ,392, 1) # range(392,784, 1) # range(49,98,1) # #
selected_type = selected_idx[:] # ['ae_sep bimodal', 'ae_sep unimodal', 'ae_sep bimodal w uniform', 'ae_sep unimodal w uniform'] #


cutoff= 8
# ['mean, mean', 'ga, mean', 'mean, ga', 'ga, ga']#
# ['Q-learning: x, Evaluation: x', 'Q-learning: o, Evaluation: x', 'Q-learning: x, Evaluation: o', 'Q-learning: o, Evaluation: o']

# truncate training ep not being used
# truncate_train_ep = 2000
##############################


def get_xyrange(envname):

    xmax = None

    if envname == 'Bimodal1DEnv':
        ymin = [-0.5, -0.5]
        ymax = [2.0, 2.0]

    elif envname == 'Bimodal1DEnv_uneq_var1':
        ymin = [-0.5, -0.5]
        ymax = [2.0, 2.0]

    elif envname == 'Bimodal1DEnv_uneq_var2':
        ymin = [-0.5, -0.5]
        ymax = [2.0, 2.0]

    elif envname == 'Bimodal1DEnv_uneq_var3':
        ymin = [-0.5, -0.5]
        ymax = [2.0, 2.0]

    elif envname == 'Bimodal1DEnv_eq_var1':
        ymin = [-0.5, -0.5]
        ymax = [2.0, 2.0]

    elif envname == 'Bimodal1DEnv_eq_var2':
        ymin = [-0.5, -0.5]
        ymax = [2.0, 2.0]

    elif envname == 'Bimodal1DEnv_eq_var3':
        ymin = [-0.5, -0.5]
        ymax = [2.0, 2.0]

    elif envname == 'Bimodal1DEnvCustom':
        ymin = [-0.5, -0.5]
        ymax = [2.0, 2.0]

    elif envname == 'HalfCheetah-v2':
        ymin = [-500, -500]
        ymax = [9000, 9000]

    elif envname == 'Hopper-v2':
        ymin = [0, 0]
        ymax = [3000, 3000]

    elif envname == 'Ant-v2':
        ymin = [-500, -1000]
        ymax = [2000, 2000]

    elif envname == 'Walker2d-v2':
        ymin = [-500, -1000]
        ymax = [2000, 4000]

    elif envname == 'Swimmer-v2':
        ymin = [20, 20]
        ymax = [40, 40]

    elif envname == 'Reacher-v2':
        ymin = [-45, -10]
        ymax = [-30, -3]

    elif envname == 'Humanoid-v2':
        ymin = [0, 0]
        ymax = [5000, 5000]

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
        ymin = [-1400, -1400]
        ymax = [-100, -100]
        # xmax = 72

    elif envname == 'InvertedPendulum-v2':
        ymin = [0, 0]
        ymax = [1200, 1200]

    elif envname == 'Bimodal2DEnv':
        ymin = [-400, -400]
        ymax = [0, 0]

    else:
        print("Environment plot setting not found!")
        exit()

    if xmax is None:

        return None, ymin, ymax
    else:
        return xmax, ymin, ymax


if __name__ == "__main__":

    if len(sys.argv)!=5:
        print('Incorrect Input')
        print('type: plot_custom_new.py merged_dir nruns algname env_json')
        exit(0)

    # Stored Directory
    storedir = str(sys.argv[2])+'/'

    # Environment Name
    envname = storedir.replace('merged','').replace('results/','')
    print('Environment: ' + envname)

    # Num Runs
    num_runs = int(sys.argv[3])

    # Agent
    agent = str(sys.argv[4])
    print('Agent: ' + agent)

    # Num Episode, Num Steps per episode
    env_filename = str(sys.argv[1])
    with open(env_filename, 'r') as env_dat:
        env_json = json.load(env_dat)

    TOTAL_MIL_STEPS = env_json['TotalMilSteps']
    EVAL_INTERVAL_MIL_STEPS = env_json['EvalIntervalMilSteps']
    EVAL_EPISODES = env_json['EvalEpisodes']


    # _, ymin, ymax = get_xyrange(envname)
    

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


        xmax, ymin, ymax = get_xyrange(envname)

        title = "result: %s, %d runs" %(agent, num_runs)

        plt.figure(figsize=(12,6))
        plt.title(title)
        
        
        plt.legend(loc="best")
        

        if result == 'TrainEpisode':
            xmax = np.shape(lc)[-1]
            # if xmax > truncate_train_ep:
            #     xmax = truncate_train_ep
            print(xmax)

            plt.xlabel('Episodes')
            opt_range = range(0, xmax)
            xlimt = (0, xmax-1)
            num_samples = num_runs

        elif result == 'EvalEpisode':
            plt.xlabel('Training Steps (per 1000 steps)')

            if xmax is None:
                xmax = np.shape(lc)[-1] # int(max_length)

            # xmax=12
            opt_range = range(0, xmax) 
            xlimt = (0, xmax-1)

            is_label_custom_defined = False

            if envname == 'Pendulum-v0':
                x_loc_arr = np.array([0, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141])
                x_val_arr = np.array([0.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
                y_val_arr = [-1600, -1200, -800, -400, -200, 0]                
                is_label_custom_defined = True

            elif envname == 'LunarLanderContinuous-v2':
                x_loc_arr = np.array([0, 41, 91, 141, 191])
                x_val_arr = np.array([45, 250, 500, 750, 1000])
                y_val_arr = [-200, -100, 0, 100, 200, 250]
                is_label_custom_defined = True

            elif envname == 'HalfCheetah-v2':
                x_loc_arr = np.array([0, 41, 91, 141, 191])
                x_val_arr = np.array([45, 250, 500, 750, 1000])
                y_val_arr = [0, 2000, 4000, 6000, 8000, 10000]
                is_label_custom_defined = True

            elif envname == 'Hopper-v2':
                x_loc_arr = np.array([0, 41, 91, 141, 191])
                x_val_arr = np.array([45, 250, 500, 750, 1000])
                y_val_arr = [0, 500, 1000, 1500, 2000, 2500, 3000]
                is_label_custom_defined = True

            elif envname == 'Swimmer-v2':
                x_loc_arr = np.array([0, 41, 91, 141, 191])
                x_val_arr = np.array([45, 250, 500, 750, 1000])
                y_val_arr = [20, 40, 60, 80, 100, 120]
                is_label_custom_defined = True

            if is_label_custom_defined:
                plt.xticks(x_loc_arr, x_val_arr)
                plt.yticks(y_val_arr, y_val_arr)
            else:
                plt.xticks(opt_range[::50], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax-1)), int(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS)+1)[::50])

            num_samples = num_runs # * EVAL_EPISODES

        h = plt.ylabel("Cum. Reward per episode")
        h.set_rotation(90)


        ylimt = (ymin[result_idx], ymax[result_idx])
        plt.ylim(ylimt)
        plt.xlim(xlimt)


        # Plot selected idx
        if len(selected_idx) != 0 :
            print("\n\n###### Plotting specific idx ########\n\n")
            sort_performance_arr = []

            handle_arr=[]
            for idx, item in enumerate(selected_idx):

                # only one line
                if len(np.shape(lc))== 1:
                    draw_lc = lc[:xmax]
                    draw_lcse = lcstd[:xmax] /np.sqrt(num_samples)
                else:
                    draw_lc = lc[item,:xmax]
                    draw_lcse = lcstd[item,:xmax] /np.sqrt(num_samples)



                if eval_last_N:
                    print('drawing.. ' + agent + ' setting: ' + str(item), np.nansum(draw_lc[xmax-last_N:xmax]))
                    sort_performance_arr.append([item, np.nansum(draw_lc[xmax-last_N:xmax])])
                else:
                    print('drawing.. ' + agent + ' setting: ' + str(item), np.nansum(draw_lc[:xmax]))
                    sort_performance_arr.append([item, np.nansum(draw_lc[:xmax])])

                plt.fill_between(opt_range, draw_lc - draw_lcse, draw_lc + draw_lcse, alpha = 0.2)#, facecolor=colors[idx])
                #handle, = plt.plot(opt_range, draw_lc, colors[idx], linewidth=1.0) 
                handle, = plt.plot(opt_range, draw_lc, linewidth=1.0) 
                handle_arr.append(handle)

            best_belowcutoff = None
            best_abovecutoff = None

            for pair in sorted(sort_performance_arr, key=lambda x: x[1], reverse=True):
                print('setting ' + str(pair[0]) + ': ' + str(pair[1]))

                if best_belowcutoff is None and pair[0] < cutoff:
                    best_belowcutoff = pair

                if best_abovecutoff is None and pair[0] >= cutoff:
                    best_abovecutoff = pair
            print("*** best below cutoff setting {} : {}".format(best_belowcutoff[0], best_belowcutoff[1]))
            print("*** best above cutoff setting {} : {}".format(best_abovecutoff[0], best_abovecutoff[1]))


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
                sort_performance_arr = []
                for i in range(len(lc)):
                    if eval_last_N:
                        sort_performance_arr.append([i, np.nansum(lc[i, xmax-last_N:xmax])])
                    else:
                        sort_performance_arr.append([i, np.nansum(lc[i,:xmax])])

                # for pair in sorted(sort_performance_arr, key=lambda x: x[1], reverse=True):
                #     print('setting ' + str(pair[0]) + ': ' + str(pair[1]))

                best_belowcutoff = None
                best_abovecutoff = None

                for pair in sorted(sort_performance_arr, key=lambda x: x[1], reverse=True):
                    print('setting ' + str(pair[0]) + ': ' + str(pair[1]))

                    if best_belowcutoff is None and pair[0] < cutoff:
                        best_belowcutoff = pair

                    if best_abovecutoff is None and pair[0] >= cutoff:
                        best_abovecutoff = pair
                print("*** best below cutoff setting {} : {}".format(best_belowcutoff[0], best_belowcutoff[1]))
                print("*** best above cutoff setting {} : {}".format(best_abovecutoff[0], best_abovecutoff[1]))


                if eval_last_N:
                    BestInd = np.argmax(np.nansum(lc[:, xmax-last_N:xmax], axis=1))
                else:
                    BestInd = np.argmax(np.nansum(lc[:, :xmax], axis=1))


                 
                bestlc = lc[BestInd,:xmax]
                lcse = lcstd[BestInd,:xmax]/np.sqrt(num_samples)
                
                try:
                    assert(BestInd == float(params[BestInd,0]))
                    print('The best param setting of '+ agent + ' is ')
                    print(params[BestInd,:])
                except:
                    # occurring because there aren't any results for some settings
                    print('the best param setting of ' + agent + ' is ' +str(BestInd))


            legends = [agent + ', ' + str(num_runs) + ' runs']

            print(len(opt_range), len(bestlc), len(lcse))
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

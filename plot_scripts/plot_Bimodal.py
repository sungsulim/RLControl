
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


# Needed
# BimodalEnv_NAF_setting_0_run_0_EpisodeRewardsLC


# Usage
# python3 ../plot_scripts/plot_Bimodal.py   DIR_RAW_RESULT(without / at the end)  ENV.json  NUM_RUNS AGENT_NAME  SETTING_NUM

show_label = True

DIR = str(sys.argv[1])

env_filename = str(sys.argv[2])
with open(env_filename, 'r') as env_dat:
    env_json = json.load(env_dat)

ENV_NAME = env_json['environment']
TOTAL_MIL_STEPS = env_json['TotalMilSteps']
EVAL_INTERVAL_MIL_STEPS = env_json['EvalIntervalMilSteps']
EVAL_EPISODES = env_json['EvalEpisodes']

NUM_RUNS = int(sys.argv[3])
AGENT_NAME = str(sys.argv[4])
SETTING_NUM = int(sys.argv[5])



#### Plot Settings #####
# opt_range = range(1, 150+1) 
# xlimt = (1, 150)


xmax = int(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS)+1
print('xmax', xmax)
# xmax = 501
opt_range = range(0, xmax) 
xlimt = (0, xmax-1)

ylimt = (-0.2, 1.8) # Bimodal1DEnv
# ylimt = (-1400, -100) # Pendulum
#ylimt = (0,3000)
# ylimt =(0,4000)

plt.figure(figsize=(12,6))

plt.ylim(ylimt)
plt.xlim(xlimt)

if show_label:
    plt.title(AGENT_NAME+": " +str(NUM_RUNS)+" runs")
    plt.xlabel('Training Steps (per 1000 steps)')
    h = plt.ylabel("Cum. Reward per episode")
    h.set_rotation(90)


tick_interval = 50
# loc = np.append(1,opt_range[tick_interval-1::tick_interval])

# # substituted math.ceil(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS) for 200
# labels = np.append(float(EVAL_INTERVAL_MIL_STEPS*1e3*1e3), np.linspace(float(EVAL_INTERVAL_MIL_STEPS * 1e3*1e3), float(EVAL_INTERVAL_MIL_STEPS * 1e3*1e3 * (xlimt[1])), int(150))[tick_interval-1::tick_interval])

# plt.xticks(loc, labels)

if show_label:
    plt.xticks(opt_range[::50], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax-1)), int(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS)+1)[::50])
    #plt.yticks([0,0.5,1,1.5], [0.0, 0.5, 1.0, 1.5])
else:
    plt.xticks(opt_range[::50], [])
    plt.yticks([0,0.5,1,1.5], [])

#####

# Read each run

eval_rewards_total_arr = []
eval_action_total_arr = []
train_sigma_total_arr = []

for i in range(NUM_RUNS):
    eval_rewards_arr = []

    # Filenames
    eval_rewards_filename = DIR + '/' + ENV_NAME + '_' + AGENT_NAME + '_setting_' + str(SETTING_NUM) + '_run_' + str(i) + '_EvalEpisodeMeanRewardsLC.txt' 
    #eval_action_filename = DIR + '/' + ENV_NAME + '_' + AGENT_NAME + '_setting_' + str(SETTING_NUM) + '_run_' + str(i) + '_EvalActionTaken.txt' 
    #train_sigma_filename = DIR + '/' + ENV_NAME + '_' + AGENT_NAME + '_setting_' + str(SETTING_NUM) + '_run_' + str(i) + '_Sigma1.txt' 

    eval_rewards_arr = np.loadtxt(eval_rewards_filename, delimiter=',')[:xmax]
    eval_rewards_arr = eval_rewards_arr
    plt.plot(eval_rewards_arr, color='b',alpha=0.1)
    eval_rewards_total_arr.append(eval_rewards_arr)
    #eval_action_total_arr.append(np.loadtxt(eval_action_filename, delimiter=','))
    #train_sigma_total_arr.append(np.loadtxt(train_sigma_filename, delimiter=','))
    # print('each run eval rewards length: ',len(eval_rewards_arr))
    # print('run: {}'.format(i))
    # print(eval_rewards_arr)
    # input()



eval_rewards_mean = np.mean(eval_rewards_total_arr, axis=0)[:xmax]
plt.plot(opt_range, eval_rewards_mean, color='b', linewidth=1.5)
plt.show()
plt.close()
# print('eval_rewards_total_arr', np.shape(eval_rewards_total_arr))
# print('eval_rewards_mean', np.shape(eval_rewards_mean))

# print(eval_rewards_mean)




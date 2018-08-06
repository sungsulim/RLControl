import sys
import os.path

# Usage: python check_missing_runs.py DIR NUM_SETTINGS, NUM_RUNS Agent_name Environment_name


search_dir = sys.argv[1]
num_settings = int(sys.argv[2])
num_runs = int(sys.argv[3])
agent_name = sys.argv[4]
env_name = sys.argv[5]

# HalfCheetah-v2results/HalfCheetah-v2_NAF_setting_0_run_0_EpisodeRewardsLC.txt

print(sys.argv)
idx_inc = 0
missing_counter = 0
missing_idx = ""
for cur_run in range(num_runs):
    for cur_setting in range(num_settings):

        search_str = "%s/%s_%s_setting_%s_run_%s_EpisodeRewardsLC.txt" % (search_dir, env_name, agent_name, cur_setting, cur_run)

        if not os.path.isfile(search_str):
            missing_idx += str(idx_inc)+","
            missing_counter += 1

        idx_inc += 1

print("num. missing idx: " + str(missing_counter))
print("Missing idx: " + missing_idx)
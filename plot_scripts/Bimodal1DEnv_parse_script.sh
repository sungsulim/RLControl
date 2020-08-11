#!/bin/bash

CUSTOM_NAME=$1
WORKING_DIR=$2

AGENT_NAME=ForwardKL
NUM_RUNS=3

USE_MOVING_AVG=False
ROOT_LOC=/Users/sungsulim/Documents/projects/ActorExpert/RLControl

arr=( Bimodal1DEnv_eq_var1 Bimodal1DEnv_eq_var2 Bimodal1DEnv_eq_var3 Bimodal1DEnv_uneq_var1 Bimodal1DEnv_uneq_var2)

for ENV_NAME in "${arr[@]}"
do
    ## merge results
    python3 /Users/sungsulim/Documents/projects/ActorExpert/RLControl/plot_scripts/merge_results_refactored.py $WORKING_DIR $ROOT_LOC $ENV_NAME $AGENT_NAME $NUM_RUNS $USE_MOVING_AVG

    ## find best setting and save/show plots
    python3 /Users/sungsulim/Documents/projects/ActorExpert/RLControl/plot_scripts/find_agent_best_setting.py $WORKING_DIR $ROOT_LOC $ENV_NAME $AGENT_NAME $NUM_RUNS $CUSTOM_NAME
done


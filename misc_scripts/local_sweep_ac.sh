#!/bin/bash

ENV_NAME=Bimodal1DEnv
AGENT_NAME=ac_separate

source /Users/sungsulim/Documents/projects/ActorExpert/RLControl/venv/bin/activate
#echo "Bash version ${BASH_VERSION}..."

# Inclusive 
start_idx=$1
increment=$2
end_idx=$3
for i in $(seq ${start_idx} ${increment} ${end_idx})
do
   echo Running..$i
   python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" 
done

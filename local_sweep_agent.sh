#!/bin/bash

ENV_NAME=$1
AGENT_NAME=$2

source /Users/sungsulim/Documents/projects/ActorExpert/RLControl/venv/bin/activate
#echo "Bash version ${BASH_VERSION}..."

# Inclusive 
start_idx=$3
increment=$4
end_idx=$5
for i in $(seq ${start_idx} ${increment} ${end_idx})
do
   echo Running..$i
   python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" # --write_plot
done

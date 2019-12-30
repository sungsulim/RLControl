#!/bin/bash

#echo "Bash version ${BASH_VERSION}..."

# inclusive of last
#for i in $(seq 72 144 2808) 
#do
#   python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_log
#done

ENV_NAME=Bimodal1DEnv

AGENT_NAME=wirefitting
for i in $(seq 0 3 29)
do
    python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done

# Not Implemented
# AGENT_NAME=optimalq
# python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index 0 --write_plot

AGENT_NAME=ac
# AC
for i in $(seq 0 18 179)
do
    python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done

# AC with OU
for i in $(seq 1 18 179)
do
    python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done

AGENT_NAME=ddpg
for i in $(seq 0 9 89)
do
    python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done

AGENT_NAME=ae_picnn
for i in $(seq 0 9 89)
do
    python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done


#!/bin/bash

#echo "Bash version ${BASH_VERSION}..."

# inclusive of last
#for i in $(seq 72 144 2808) 
#do
#   python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_log
#done

ENV_NAME=Bimodal1DEnv


AGENT_NAME=qt_opt
# QT_OPT
for i in $(seq 2 12 119)
do
    python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done

# QT_OPT_single
for i in $(seq 8 12 119)
do
    python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done

# QT_OPT_ou(double)
for i in $(seq 3 12 119)
do
    python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done


AGENT_NAME=picnn
for i in $(seq 0 3 29)
do
    python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done

AGENT_NAME=naf
for i in $(seq 0 9 89)
do
    python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done

for i in $(seq 1 9 89)
do
    python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done

for i in $(seq 2 9 89)
do
    python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done

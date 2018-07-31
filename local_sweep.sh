#!/bin/bash

ENV_NAME=Pendulum-v0
AGENT_NAME=ddpg

echo "Bash version ${BASH_VERSION}..."
#for i in {0..4..1}
for i in $(seq 0 1 6)
do
   python3 main.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i"
done

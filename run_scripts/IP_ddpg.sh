#!/bin/bash
#SBATCH --job-name=IP_ddpg
#SBATCH --output=/home/sungsu/scratch/output_log/IP/ddpg/%A%a.out
#SBATCH --error=/home/sungsu/scratch/output_log/IP/ddpg/%A%a.err

#SBATCH --array=0-89:1

# SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=2048M

#SBATCH --account=def-whitem
#SBATCH --mail-user=slurmjob@gmail.com
#SBATCH --mail-type=FAIL

ENV_NAME=InvertedPendulum-v2
AGENT_NAME=ddpg

# module load python/3.6.3
# source /home/sungsu/tensorflow-cpu3/bin/activate
# DIR=/home/sungsu/workspace/actiongeneral/python

module load singularity/2.5
echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID
singularity exec -B /scratch /home/sungsu/rl-docker-private-tf1.8.0-gym0.10.3-py35.simg python3 ../main.py --env_json ../jsonfiles/environment/$ENV_NAME.json --agent_json ../jsonfiles/agent/$AGENT_NAME.json --index $SLURM_ARRAY_TASK_ID

#!/bin/bash
#SBATCH --job-name=central_node
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=25G
#SBATCH --num-nodes=1
#SBATCH --cpus-per-task=16

source ~/miniconda3/etc/profile.d/conda.sh
conda activate curri
#export PYTHONPATH=$HOME/
cd $HOME/auto-curriculum/maze_representations/
/home/sgar/miniconda3/envs/curri/bin/python run_experiments.py -m run_name=hyperparameter_exploration +sweeper=wandb +launcher=submitit_remote num_cpus=16 offline=true
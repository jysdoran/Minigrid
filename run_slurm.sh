#!/bin/bash
#SBATCH --job-name=central_node
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --mem=25G
#SBATCH --cpus-per-task=16

source ~/miniconda3/etc/profile.d/conda.sh
conda activate curri
#export PYTHONPATH=$HOME/
cd $HOME/auto-curriculum-design/maze_representations/
export WANDB_START_METHOD=thread
#wandb offline
#export WANDB_MODE=offline
# export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
/home/sgar/miniconda3/envs/curri/bin/python run_experiments.py -m run_name=new_interpolation +sweeper=wandb_VAE_10k +launcher=submitit_remote num_cpus=16 offline=false

#!/bin/bash
#SBATCH --job-name=central_node
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --mem=25G
#SBATCH --cpus-per-task=16

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dcd2
#export PYTHONPATH=$HOME/
cd $HOME/auto-curriculum-design/_dcd/
export WANDB_START_METHOD=thread
#wandb offline
#export WANDB_MODE=offline
# export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
$HOME/miniconda3/envs/dcd2/bin/python3.8 maze_representations/run_experiments.py -m run_name=minigrid_dense_graph_22k accelerator=gpu results.num_stored_samples=256 results.num_image_samples=32 results.num_embedding_samples=256 results.num_generated_samples=32 results.sample_interpolation.num_samples=32 +sweeper=wandb_VAE_densegraph hydra.job.chdir=False +launcher=submitit_remote num_cpus=6 offline=false

#!/bin/bash
#SBATCH --job-name=central_node
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=40
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dcd2
cd $HOME/auto-curriculum-design/_dcd/maze_representations/multirun/2022-08-25/03-46-46
wandb sync --sync-all
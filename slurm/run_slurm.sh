#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)
#            Samuel Garcin (s.garcin@ed.ac.uk)
#
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12
# sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_arrayjob.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```


# ====================
# Options for sbatch
# ====================

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/%u/auto-curriculum-design/_dcd/maze_representations/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/auto-curriculum-design/_dcd/maze_representations/slurm_logs/slurm-%A_%a.out

#SBATCH --job-name=central_node
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --mem=25G
#SBATCH --cpus-per-task=16

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source /home/${USER}/.bashrc

# Make script bail out after first error
set -e

# Explicitly activate conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
CONDA_ENV_NAME=dcd2
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

# input data directory path on the DFS - change line below if loc different
REPO=/home/${USER}/auto-curriculum-design/_dcd

# Wandb options
# ===================
export WANDB_START_METHOD=thread
#wandb offline
#export WANDB_MODE=offline
# export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Launch the job
# ===================
echo "Code repository located at: $REPO"
cd $REPO
$HOME/miniconda3/envs/${CONDA_ENV_NAME}/bin/python3.8 maze_representations/run_experiments.py -m run_name=minigrid_dense_graph_22k accelerator=gpu results.num_stored_samples=256 results.num_image_samples=32 results.num_embedding_samples=256 results.num_generated_samples=32 results.sample_interpolation.num_samples=32 +sweeper=wandb_VAE_densegraph hydra.job.chdir=False +launcher=submitit_remote num_cpus=6 offline=false

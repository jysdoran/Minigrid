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
#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

#SBATCH --job-name=data_generation
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --mem=120G

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
source ${HOME}/.bashrc

# Make script bail out after first error
set -e

# Explicitly activate conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
CONDA_ENV_NAME=dcd3
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

# input data directory path on the DFS - change line below if loc different
REPO= ${HOME}/auto-curriculum-design/_dcd

if [ -z ${DEBUG+x} ] ; then
    CONF = "minigrid_dense_graph_1M.yaml"
    DATASET_DIRNAME = "minigrid_dense_graph_1M"
else
    echo "DEBUG MODE"
    CONF = "minigrid_dense_graph_debug"
    DATASET_DIRNAME = "test_minigrid_dense_graph"
fi

# Launch the job
# ===================
echo "Code repository located at: $REPO"
cd $REPO
$HOME/miniconda3/envs/${CONDA_ENV_NAME}/bin/python maze_representations/data_generators.py models=data_generation num_cpus=$NUM_PROC +data_generation=$CONF accelerator=cpu +multiprocessing=1 data_generation.dir_name=$DATASET_DIRNAME
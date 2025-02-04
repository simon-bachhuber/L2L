#!/bin/bash -l
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=200:00:00
#SBATCH --partition=imes.gpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=10G
 
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Change to my work dir
cd $SLURM_SUBMIT_DIR

# Load modules
module load Miniforge3

# Activate Env
conda activate jax_l2l

python train_BPTT_rnn.py

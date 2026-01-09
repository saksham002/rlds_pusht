#!/bin/bash

#SBATCH --job-name=build_robocoin_dataset
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --output=logs/build_robocoin_dataset.out
#SBATCH --error=logs/build_robocoin_dataset.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlds

# print diagnostics
echo "Running on host: $(hostname)"
echo "Current conda environment: $CONDA_DEFAULT_ENV"

tfds build --overwrite --data_dir=/data/group_data/rl/saksham3/
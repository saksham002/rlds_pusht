#!/bin/bash

#SBATCH --job-name=build_sim_bimanual_assembly_dataset
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --output=logs/build_sim_bimanual_assembly_dataset.out
#SBATCH --error=logs/build_sim_bimanual_assembly_dataset.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlds

# print diagnostics
echo "Running on host: $(hostname)"
echo "Current conda environment: $CONDA_DEFAULT_ENV"

tfds build --overwrite --data_dir=/data/group_data/rl/saksham3/
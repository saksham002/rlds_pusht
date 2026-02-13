#!/bin/bash

#SBATCH --job-name=build_robocoin_dataset
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=16
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

export FFMPEG_PREFIX=""
export PATH="/home/saksham3/miniconda3/envs/rlds/bin:/data/user_data/saksham3/vla/bin:/home/saksham3/.local/node_modules/.bin:/usr/share/Modules/bin:/data/user_data/saksham3/ffmpeg-7/bin:/home/saksham3/.local/node_modules/.bin:/home/saksham3/.nvm/versions/node/v25.0.0/bin:/home/saksham3/miniconda3/condabin:/home/saksham3/.local/bin:/home/saksham3/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
export LD_LIBRARY_PATH=""
export PKG_CONFIG_PATH=""
export TEST_MODE=1
export PROFILE_BUILD=1

echo "=== Run 1: DataLoader mode ==="
export USE_DATALOADER=1
tfds build --overwrite --data_dir=/data/group_data/rl/saksham3/robocoin_test/

echo "=== Run 2: Direct indexing mode ==="
export USE_DATALOADER=0
tfds build --overwrite --data_dir=/data/group_data/rl/saksham3/robocoin_test/

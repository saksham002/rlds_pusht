#!/bin/bash

#SBATCH --job-name=compute_stats
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --array=0-7
#SBATCH --output=logs/compute_stats_%a.out
#SBATCH --error=logs/compute_stats_%a.err

cd /home/saksham3/projects/AIRe/rlds_dataset_builder/robocoin

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlds

echo "Running on host: $(hostname)"
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

export FFMPEG_PREFIX=""
export PATH="/home/saksham3/miniconda3/envs/rlds/bin:/data/user_data/saksham3/vla/bin:/home/saksham3/.local/node_modules/.bin:/usr/share/Modules/bin:/data/user_data/saksham3/ffmpeg-7/bin:/home/saksham3/.local/node_modules/.bin:/home/saksham3/miniconda3/condabin:/home/saksham3/.local/bin:/home/saksham3/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
export LD_LIBRARY_PATH=""
export PKG_CONFIG_PATH=""

# 131 repos split across 8 jobs: 131 = 3*17 + 5*16
TOTAL=131
NUM_JOBS=8
BASE=$(( TOTAL / NUM_JOBS ))
REMAINDER=$(( TOTAL % NUM_JOBS ))
if [ $SLURM_ARRAY_TASK_ID -lt $REMAINDER ]; then
    CHUNK=$(( BASE + 1 ))
    START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
else
    CHUNK=$BASE
    START=$(( REMAINDER * (BASE + 1) + (SLURM_ARRAY_TASK_ID - REMAINDER) * BASE ))
fi
END=$(( START + CHUNK ))

echo "Processing repos [$START:$END]"

python compute_stats.py --disable_global --calc_indices "${START},${END}"

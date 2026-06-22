#!/bin/bash
#SBATCH --job-name=merge_xarm_packing
#SBATCH --partition=general
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/merge.out
#SBATCH --error=logs/merge.err

# Merge + scan + fix for the 7-worker realworld_xarm_packing build.
# Writes the final dataset to ${ROOT}/realworld_xarm_packing/1.0.0 (no merged/ level).

source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlds

export PYTHONUNBUFFERED=1
export CURL_CA_BUNDLE="/data/user_data/saksham3/conda-envs/rlds/lib/python3.12/site-packages/certifi/cacert.pem"
export SSL_CERT_FILE="$CURL_CA_BUNDLE"

ROOT=/data/group_data/rl/saksham3/realworld_xarm_packing_rlds
FRAMEWORK_DIR=/home/saksham3/projects/AIRe/rlds_dataset_builder/slurm_rlds
CONFIG=/home/saksham3/projects/AIRe/rlds_dataset_builder/lerobot/realworld_xarm_packing_config.py

cd "$FRAMEWORK_DIR"
python -u scripts/pipeline.py \
    --config "$CONFIG" \
    --data_root "${ROOT}/workers" \
    --output "${ROOT}" \
    --num_workers 7 \
    --dataset_name realworld_xarm_packing \
    --dataset_version 1.0.0 \
    --framework_dir "$FRAMEWORK_DIR" \
    --skip_build \
    --overwrite \
    --copy_workers 32 \
    --splits train val \
    --log_dir "${FRAMEWORK_DIR}/logs"

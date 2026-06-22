#!/bin/bash
#SBATCH --job-name=xarm_pipeline
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pipeline.out
#SBATCH --error=logs/pipeline.err

# Single orchestrator job: pipeline.py drives build (nested 7-worker array via
# sbatch --wait) -> merge, with scan/fix skipped. This job reaching COMPLETED
# means the merged dataset is fully written.

source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlds

export PYTHONUNBUFFERED=1
export CURL_CA_BUNDLE="/data/user_data/saksham3/conda-envs/rlds/lib/python3.12/site-packages/certifi/cacert.pem"
export SSL_CERT_FILE="$CURL_CA_BUNDLE"

ROOT=/data/group_data/rl/saksham3/realworld_xarm_packing_rlds
FW=/home/saksham3/projects/AIRe/rlds_dataset_builder/slurm_rlds
CONFIG=/home/saksham3/projects/AIRe/rlds_dataset_builder/lerobot/realworld_xarm_packing_config.py

cd "$FW"
python -u scripts/pipeline.py \
    --config "$CONFIG" \
    --data_root "${ROOT}/workers" \
    --output "${ROOT}" \
    --num_workers 7 \
    --dataset_name realworld_xarm_packing \
    --dataset_version 1.0.0 \
    --framework_dir "$FW" \
    --mode slurm \
    --slurm_partition general \
    --slurm_gres gpu:1 \
    --slurm_time 24:00:00 \
    --slurm_cpus 8 \
    --slurm_mem 64G \
    --skip_scan \
    --overwrite \
    --copy_workers 16 \
    --log_dir "${FW}/logs"

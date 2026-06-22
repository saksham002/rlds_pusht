#!/bin/bash
#SBATCH --job-name=build_xarm_packing
#SBATCH --array=0-6
#SBATCH --partition=general
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/build_%a.out
#SBATCH --error=logs/build_%a.err

# ── User config ───────────────────────────────────────────────────────────────
USER_CONFIG="/home/saksham3/projects/AIRe/rlds_dataset_builder/lerobot/realworld_xarm_packing_config.py"
DATA_ROOT="/data/group_data/rl/saksham3/realworld_xarm_packing_rlds/workers"
DATASET_NAME="realworld_xarm_packing"
DATASET_VERSION="1.0.0"
FRAMEWORK_DIR="/home/saksham3/projects/AIRe/rlds_dataset_builder/slurm_rlds"
# ─────────────────────────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlds

export PYTHONUNBUFFERED=1
export CURL_CA_BUNDLE="/data/user_data/saksham3/conda-envs/rlds/lib/python3.12/site-packages/certifi/cacert.pem"
export SSL_CERT_FILE="$CURL_CA_BUNDLE"

# NUM_WORKERS is FIXED at 7: unit numbering / episode slicing assume 7 workers.
# Never derive from SLURM_ARRAY_TASK_COUNT — resubmitting a subset (e.g. failed
# workers) would otherwise change the slicing and corrupt the dataset.
WORKER_ID=$SLURM_ARRAY_TASK_ID
NUM_WORKERS=7
DATA_DIR="${DATA_ROOT}/${WORKER_ID}"
MARKER="${DATA_DIR}/${DATASET_NAME}/${DATASET_VERSION}/dataset_info.json"

echo "[BUILD] worker ${WORKER_ID}/${NUM_WORKERS} -> ${DATA_DIR}"

if [ -f "${MARKER}" ]; then
    echo "[SKIP] worker ${WORKER_ID} already done"
    exit 0
fi

cd "$FRAMEWORK_DIR"
if python -u -m framework.runner \
    --config "${USER_CONFIG}" \
    --data_dir "${DATA_DIR}" \
    --worker_id "${WORKER_ID}" \
    --num_workers "${NUM_WORKERS}"; then
    echo "[SUCCESS] worker ${WORKER_ID}"
else
    echo "[FAILED] worker ${WORKER_ID}"
    exit 1
fi

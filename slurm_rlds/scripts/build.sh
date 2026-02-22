#!/bin/bash
#SBATCH --job-name=build_rlds
#SBATCH --array=0-15
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/build_%a.out
#SBATCH --error=logs/build_%a.err

# ── User config ───────────────────────────────────────────────────────────────
USER_CONFIG="/path/to/my_dataset_config.py"
DATA_ROOT="gs://my-bucket/my_dataset_workers"
DATASET_NAME="my_dataset"
DATASET_VERSION="1.0.0"
FRAMEWORK_DIR="/path/to/slurm_rlds"
# ─────────────────────────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlds

export CURL_CA_BUNDLE="/home/saksham3/miniconda3/envs/rlds/lib/python3.12/site-packages/certifi/cacert.pem"
export SSL_CERT_FILE="$CURL_CA_BUNDLE"

WORKER_ID=$SLURM_ARRAY_TASK_ID
NUM_WORKERS=$SLURM_ARRAY_TASK_COUNT
DATA_DIR="${DATA_ROOT}/${WORKER_ID}"
MARKER="${DATA_DIR}/${DATASET_NAME}/${DATASET_VERSION}/dataset_info.json"

echo "[BUILD] worker ${WORKER_ID}/${NUM_WORKERS} → ${DATA_DIR}"

# Check completion marker via Python (handles GCS paths)
if python -c "
import tensorflow as tf, time
while True:
    try:
        exit(0 if tf.io.gfile.exists('${MARKER}') else 1)
    except Exception:
        time.sleep(5)
" 2>/dev/null; then
    echo "[SKIP] worker ${WORKER_ID} already done"
    exit 0
fi

cd "$FRAMEWORK_DIR"
if python -m framework.runner \
    --config "${USER_CONFIG}" \
    --data_dir "${DATA_DIR}" \
    --worker_id "${WORKER_ID}" \
    --num_workers "${NUM_WORKERS}"; then
    echo "[SUCCESS] worker ${WORKER_ID}"
else
    echo "[FAILED] worker ${WORKER_ID}"
    exit 1
fi

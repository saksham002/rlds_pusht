#!/bin/bash

#SBATCH --job-name=build_robocoin_local
#SBATCH --array=0-31
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=logs/build_local_%a.out
#SBATCH --error=logs/build_local_%a.err

cd /home/saksham3/projects/AIRe/rlds_dataset_builder/robocoin

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlds

echo "Running on host: $(hostname)"
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

export CURL_CA_BUNDLE="/home/saksham3/miniconda3/envs/rlds/lib/python3.12/site-packages/certifi/cacert.pem"
export SSL_CERT_FILE="$CURL_CA_BUNDLE"
export FFMPEG_PREFIX=""
export PATH="/home/saksham3/miniconda3/envs/rlds/bin:/data/user_data/saksham3/vla/bin:/home/saksham3/.local/node_modules/.bin:/usr/share/Modules/bin:/data/user_data/saksham3/ffmpeg-7/bin:/home/saksham3/.local/node_modules/.bin:/home/saksham3/miniconda3/condabin:/home/saksham3/.local/bin:/home/saksham3/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
export LD_LIBRARY_PATH=""
export PKG_CONFIG_PATH=""

export WORKER_ID=$SLURM_ARRAY_TASK_ID
export NUM_WORKERS=$SLURM_ARRAY_TASK_COUNT
export REPO_IDS_FILE=repos.txt

DATA_ROOT="gs://saksham-euw4/robocoin_bimanual"
export DATA_DIR_ROOT="$DATA_ROOT"

while IFS= read -r REPO_SUFFIX; do
    [ -z "$REPO_SUFFIX" ] && continue

    DATA_DIR="${DATA_ROOT}/${REPO_SUFFIX}/${WORKER_ID}"
    MARKER="${DATA_DIR}/robocoin/1.0.0/dataset_info.json"

    # Skip if completion marker exists
    if python -c "import tensorflow as tf; exit(0 if tf.io.gfile.exists('${MARKER}') else 1)" 2>/dev/null; then
        echo "[SKIP] ${REPO_SUFFIX} (worker ${WORKER_ID})"
        continue
    fi

    echo "[BUILD] ${REPO_SUFFIX} (worker ${WORKER_ID}) -> ${DATA_DIR}"
    TMPFILE=$(mktemp)
    echo "$REPO_SUFFIX" > "$TMPFILE"
    REPO_IDS_FILE="$TMPFILE" tfds build --overwrite --data_dir="${DATA_DIR}"
    rm -f "$TMPFILE"

done < "$REPO_IDS_FILE"

echo "Worker ${WORKER_ID} done."

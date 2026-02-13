#!/bin/bash

#SBATCH --job-name=build_robocoin_bimanual_df
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --output=logs/build_robocoin_bimanual_dataflow.out
#SBATCH --error=logs/build_robocoin_bimanual_dataflow.err

# Build the RoboCOIN bimanual dataset using Google Cloud Dataflow.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlds

echo "Running on host: $(hostname)"
echo "Current conda environment: $CONDA_DEFAULT_ENV"
date

# Point libcurl/OpenSSL to the RHEL CA bundle (TF defaults to Debian path)
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

export FFMPEG_PREFIX=""
export PATH="/home/saksham3/miniconda3/envs/rlds/bin:/data/user_data/saksham3/vla/bin:/home/saksham3/.local/node_modules/.bin:/usr/share/Modules/bin:/data/user_data/saksham3/ffmpeg-7/bin:/home/saksham3/.local/node_modules/.bin:/home/saksham3/.nvm/versions/node/v25.0.0/bin:/home/saksham3/miniconda3/condabin:/home/saksham3/.local/bin:/home/saksham3/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
export LD_LIBRARY_PATH=""
export PKG_CONFIG_PATH=""

cd /home/saksham3/projects/AIRe/rlds_dataset_builder/robocoin/robocoin_bimanual

# --- GCP configuration ---
PROJECT=cmu-aidm-v2
REGION=europe-west4
GCS_BUCKET=gs://saksham-euw4

# Dataset output path (GCS)
DATA_DIR=${DATA_DIR:-"${GCS_BUCKET}/robocoin_bimanual/"}

# Dataflow staging and temp locations
STAGING_LOCATION=${GCS_BUCKET}/dataflow/staging
TEMP_LOCATION=${GCS_BUCKET}/dataflow/tmp

# Custom SDK container (must be pre-built; see prerequisites above)
CONTAINER_IMAGE=europe-west4-docker.pkg.dev/${PROJECT}/robocoin-dataflow/worker:latest

# --- Worker configuration ---
MACHINE_TYPE=n1-highmem-8
DISK_SIZE_GB=2000
MAX_NUM_WORKERS=16

# --- Test mode (set to 1 for small test run) ---
export TEST_MODE=0

echo "DATA_DIR=${DATA_DIR}"
echo "STAGING_LOCATION=${STAGING_LOCATION}"
echo "CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "MAX_NUM_WORKERS=${MAX_NUM_WORKERS}"

tfds build --overwrite \
    --data_dir="${DATA_DIR}" \
    --beam_pipeline_options="\
runner=DataflowRunner,\
project=${PROJECT},\
region=${REGION},\
staging_location=${STAGING_LOCATION},\
temp_location=${TEMP_LOCATION},\
sdk_container_image=${CONTAINER_IMAGE},\
machine_type=${MACHINE_TYPE},\
disk_size_gb=${DISK_SIZE_GB},\
max_num_workers=${MAX_NUM_WORKERS},\
setup_file=./setup.py,\
sdk_worker_parallelism=1,\
experiments=use_runner_v2"

echo ""
echo "tfds build finished at $(date)"

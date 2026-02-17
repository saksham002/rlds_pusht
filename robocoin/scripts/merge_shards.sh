#!/bin/bash

#SBATCH --job-name=merge_shards
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=logs/merge_shards.out
#SBATCH --error=logs/merge_shards.err

cd /home/saksham3/projects/AIRe/rlds_dataset_builder/robocoin

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlds

echo "Running on host: $(hostname)"

export CURL_CA_BUNDLE="/home/saksham3/miniconda3/envs/rlds/lib/python3.12/site-packages/certifi/cacert.pem"
export SSL_CERT_FILE="$CURL_CA_BUNDLE"
export PATH="/home/saksham3/miniconda3/envs/rlds/bin:/data/user_data/saksham3/vla/bin:/home/saksham3/.local/node_modules/.bin:/usr/share/Modules/bin:/data/user_data/saksham3/ffmpeg-7/bin:/home/saksham3/.local/node_modules/.bin:/home/saksham3/miniconda3/condabin:/home/saksham3/.local/bin:/home/saksham3/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
export LD_LIBRARY_PATH=""
export PKG_CONFIG_PATH=""

echo "Split_aloha_plate_storage" > /tmp/merge_repo_list.txt

python scripts/merge_shards.py \
    --root gs://saksham-euw4/robocoin_bimanual \
    --output gs://saksham-euw4/robocoin_merged_test \
    --num_workers 32 \
    --repo_list /tmp/merge_repo_list.txt \
    --splits val \
    --overwrite

echo "Merge done."

#!/bin/bash
# Script to visualize episode_0.hdf5 from different dataset directories using trained value functions

#SBATCH --job-name=visualize_episodes
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --output=logs/visualize_episodes.out
#SBATCH --error=logs/visualize_episodes.err

# Set default parameters - same value functions as advantage_computation.sh
VALUE_FUNCTION_CHECKPOINT_1="/data/user_data/saksham3/double_insert_rl_checkpoints/td-n_n60_reward_typesparse_lr0.0001_seed3/checkpoint_step_40000.eqx"
VALUE_FUNCTION_CHECKPOINT_2="/data/user_data/saksham3/double_insert_rl_checkpoints/td-n_n60_reward_typeneg_lr0.0001_seed3/checkpoint_step_20000.eqx"
USE_EMA=false

# Dataset directories (select first 5 from config)
DATASET_DIRS=(
    "/data/group_data/rl/dexterous_robot_data/sim_double_insert_0226_hdf5"
    "/data/group_data/rl/dexterous_robot_data/sim_double_insert_round2_0522_hdf5"
    "/data/group_data/rl/dexterous_robot_data/sim_double_insert_round4_0606_hdf5"
    "/data/group_data/rl/dexterous_robot_data/sim_double_insert_full_success_r5_hdf5"
    "/data/group_data/rl/dexterous_robot_data/sim_double_insert_round6_0820_hdf5"
)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dataloader_new

# Print diagnostics
echo "Running on host: $(hostname), job ID: $SLURM_JOB_ID"
echo "Current conda environment: $CONDA_DEFAULT_ENV"

# Visualize episode_0.hdf5 from each directory with VALUE_FUNCTION_CHECKPOINT_1 (sparse reward)
# echo "==================================="
# echo "Visualizing with VALUE_FUNCTION_CHECKPOINT_1 (sparse reward, gamma=0.9995)"
# echo "==================================="

# for i in "${!DATASET_DIRS[@]}"; do
#     DATASET_DIR="${DATASET_DIRS[$i]}"
#     EPISODE_FILE="${DATASET_DIR}/episode_0.hdf5"
    
#     if [ ! -f "$EPISODE_FILE" ]; then
#         echo "Warning: Episode file not found: $EPISODE_FILE"
#         continue
#     fi
    
#     echo "Processing dataset $((i+1))/5: $(basename $DATASET_DIR)"
    
#     if [ "$USE_EMA" = true ]; then
#         echo "Using EMA model"
#         python visualize_episode.py \
#             --episode_file_path="${EPISODE_FILE}" \
#             --value_function_checkpoint="${VALUE_FUNCTION_CHECKPOINT_1}" \
#             --use_ema
#     else
#         python visualize_episode.py \
#             --episode_file_path="${EPISODE_FILE}" \
#             --value_function_checkpoint="${VALUE_FUNCTION_CHECKPOINT_1}"
#     fi
    
#     echo "Completed visualization for $(basename $DATASET_DIR) with VF1"
# done

# Visualize episode_0.hdf5 from each directory with VALUE_FUNCTION_CHECKPOINT_2 (neg reward)
echo ""
echo "==================================="
echo "Visualizing with VALUE_FUNCTION_CHECKPOINT_2 (neg reward, gamma=1.0)"
echo "==================================="

for i in "${!DATASET_DIRS[@]}"; do
    DATASET_DIR="${DATASET_DIRS[$i]}"
    EPISODE_FILE="${DATASET_DIR}/episode_0.hdf5"
    
    if [ ! -f "$EPISODE_FILE" ]; then
        echo "Warning: Episode file not found: $EPISODE_FILE"
        continue
    fi
    
    echo "Processing dataset $((i+1))/5: $(basename $DATASET_DIR)"
    
    if [ "$USE_EMA" = true ]; then
        echo "Using EMA model"
        python visualize_episode.py \
            --episode_file_path="${EPISODE_FILE}" \
            --value_function_checkpoint="${VALUE_FUNCTION_CHECKPOINT_2}" \
            --use_ema
    else
        python visualize_episode.py \
            --episode_file_path="${EPISODE_FILE}" \
            --value_function_checkpoint="${VALUE_FUNCTION_CHECKPOINT_2}"
    fi
    
    echo "Completed visualization for $(basename $DATASET_DIR) with VF2"
done

echo ""
echo "==================================="
echo "All visualizations complete!"
echo "Videos logged to wandb"
echo "==================================="


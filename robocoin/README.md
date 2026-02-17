# RoboCOIN Dataset

## Overview

The RoboCOIN dataset builder merges multiple bimanual robotics datasets from the [RoboCOIN](https://huggingface.co/RoboCOIN) collection on Hugging Face into a unified RLDS format. It filters to three bimanual robot platforms (`Split_aloha`, `Cobot_Magic`, `R1_Lite` with `state_dim = 14`), standardises state/action to a canonical 14-D joint representation, resizes all camera images to 224x224 JPEG, and computes rich subtask timing features.

**Source**: [RoboCOIN on Hugging Face](https://huggingface.co/RoboCOIN) -- all datasets prefixed with `RoboCOIN/`

## Builders

There are two TFDS builders that produce the same schema:

| Builder | File | Superclass | Parallelism |
|---------|------|------------|-------------|
| `Robocoin` (HPC array) | `robocoin_dataset_builder.py` | `GeneratorBasedBuilder` | SLURM array job with episode-level sharding across workers |
| `RobocoinBimanual` (Beam) | `robocoin_bimanual/robocoin_bimanual_dataset_builder.py` | `BeamBasedBuilder` | Apache Beam (local Prism or Cloud Dataflow) |

The **HPC array builder** is the production version. It runs as a SLURM `--array=0-31` job where every worker processes all repos but only its slice of episodes. Episodes are divided among `effective_workers = min(NUM_WORKERS, ceil(total_steps / MIN_STEPS_PER_WORKER))` to avoid tiny shards. Workers that exceed the effective count skip the repo. Each worker writes to `<root>/<repo>/<worker_id>/` on GCS; a separate merge script combines the per-worker outputs into a single flat TFDS dataset. File locking (`fcntl`) prevents concurrent workers from racing on download and NaN-fix operations for the same repo.

The **Beam builder** provides an alternative via Apache Beam, where each repo is an independent element enabling parallel downloads and episode processing. It also writes per-repo normalization statistics to GCS as a side effect.

### `robocoin_bimanual/` Directory

```
robocoin_bimanual/
  robocoin_bimanual_dataset_builder.py   # BeamBasedBuilder (main logic)
  setup.py                               # Minimal setuptools for Beam --setup_file staging
  Dockerfile                             # Custom Dataflow worker container (Beam 2.69.0, Python 3.12)
```

- **`setup.py`**: Allows Beam to stage the builder code to remote Dataflow workers via `--setup_file=./setup.py`.
- **`Dockerfile`**: Builds on `apache/beam_python3.12_sdk:2.69.0`, installs ffmpeg, TensorFlow, lerobot, CPU-only PyTorch, etc. Builder code is staged at runtime, not baked in.

## Features

Each example is one episode with the following schema:

### Step-Level Features

#### Observations
- **`observation/image/cam_{0,1,2}`**: Up to 3 camera views (front/high, left wrist, right wrist) resized to 224x224 and JPEG-encoded. Unused slots contain empty bytes.
- **`observation/state`**: `float32[14]` -- canonical joint state: 6 left arm joints + left gripper + 6 right arm joints + right gripper.

#### Actions
- **`action`**: `float32[14]` -- canonical joint action (same 14-D layout as state).
- **`action_diff`**: `float32[14]` -- `action[t+1] - action[t]`, zeros at the last step.
- **`state_diff`**: `float32[14]` -- `state[t+1] - state[t]`, zeros at the last step.

#### End-Effector Poses
- **`eef_sim_pose_state`**: `float32[12]` -- left/right EEF position (x,y,z) + orientation (x,y,z).
- **`eef_sim_pose_action`**: `float32[12]` -- same layout for actions.
- **`eef_sim_pose_action_diff`**: `float32[12]` -- `eef_action[t+1] - eef_action[t]`, zeros at the last step.

#### Subtask Annotations
- **`subtask_{1..5}`**: Text labels for up to 5 concurrent subtasks.
- **`subtask_mask`**: `bool[5]` -- which of the 5 slots are active (non-null).
- **`steps_to_subtask_end`**: `int32[5]` -- countdown to each subtask's end.
- **`subtask_len`**: `int32[5]` -- total length of each subtask's current interval.
- **`subtask_is_first`**: `bool[5]` -- whether this step is the first in each subtask interval.
- **`subtask_is_last`**: `bool[5]` -- whether this step is the last in each subtask interval.
- **`first_null_index`**: `int32` -- index of the first null slot in the subtask annotation.
- **`scene_annotation`**: `int32` -- scene identifier.

#### Standard RLDS Fields
- **`is_first`**: `bool` -- first step of the episode.
- **`is_terminal`**: `bool` -- last step of the episode.
- **`frame_index`**: `int64` -- frame index within the episode.
- **`task`**: `text` -- natural language task description.
- **`episode_index`**: `int64` -- episode identifier.
- **`index`**: `int64` -- global step index.
- **`repo_index`**: `int32` -- index of the source repo in the build order.

### Episode Metadata
- **`repo_id`**: Hugging Face repository ID (e.g., `RoboCOIN/Split_aloha_plate_storage`).
- **`robot_type`**: Robot platform (`Split_aloha`, `Cobot_Magic`, or `R1_Lite`).
- **`fps`**: Frame rate (`float32`).
- **`camera_names`**: Ordered list of 3 camera names.
- **`camera_shapes`**: Original camera resolutions `(H, W, C)` per camera.
- **`num_cameras`**: Number of active cameras (`int64`).
- **`state_feature_names`**: Names of all state vector components from the source repo.
- **`action_feature_names`**: Names of all action vector components from the source repo.
- **`subtasks`**: List of all subtask descriptions for this repo.
- **`task_description`**: Natural language task description.

## Train/Val Split

Episodes are split with a 95/5 train/val ratio using a fixed random seed (86). Three repos (`Split_aloha_plate_storage`, `Cobot_Magic_cut_banana`, `R1_Lite_tableware_cleaning`) are assigned entirely to the validation split.

## Building the Dataset

### Prerequisites

- `robocoin-download` CLI tool
- Hugging Face Hub access token
- Conda environment `rlds` (Python 3.12, apache-beam 2.69.0, tensorflow 2.20.0, tensorflow-datasets 4.9.9, lerobot 0.3.3, robocoin 0.1.0.1)

### HPC Array Builder (production)

Each SLURM array worker processes all repos but only its slice of episodes. Workers write per-(repo, worker) outputs to GCS, then a merge script combines them.

```bash
cd robocoin

# Launch 32 parallel workers
sbatch scripts/build_local.sh

# After all workers complete, merge per-worker shards into a single dataset
python scripts/merge_shards.py \
    --root gs://saksham-euw4/robocoin_bimanual \
    --output gs://saksham-euw4/robocoin_merged \
    --num_workers 32 \
    --repo_list repos.txt \
    --overwrite
```

Key environment variables consumed by the builder:
- `WORKER_ID` / `NUM_WORKERS`: Set automatically by SLURM (`SLURM_ARRAY_TASK_ID` / `SLURM_ARRAY_TASK_COUNT`).
- `REPO_IDS_FILE`: Path to a text file listing repo suffixes (one per line).
- `TEST_MODE=1`: Build only `Split_aloha_plate_storage` with 10 episodes per worker.
- `DATA_DIR_ROOT`: GCS root for completion marker checks.

### Beam Builder (local Prism runner)

```bash
cd robocoin/robocoin_bimanual
tfds build --data_dir=/path/to/output
```

### Beam Builder (Google Cloud Dataflow)

1. Build and push the worker container:
   ```bash
   cd robocoin/robocoin_bimanual
   gcloud builds submit \
       --tag europe-west4-docker.pkg.dev/cmu-aidm-v2/robocoin-dataflow/worker:latest
   ```

2. Launch the Dataflow job:
   ```bash
   sbatch robocoin/scripts/build_bimanual_dataflow.sh
   ```

   This runs up to 16 `n1-highmem-8` Dataflow workers with 2TB disks, writing output to `gs://saksham-euw4/robocoin_bimanual/`.

## Normalization Statistics

### Per-Repo Stats (written by the Beam builder)

The bimanual builder computes streaming stats (Welford's algorithm) during the train split and writes per-repo `norm_stats.json` files to GCS under `gs://saksham-euw4/robocoin/norm_stats/<repo_name>/`.

### Standalone Stats Computation

```bash
# Compute stats for specific repos
python compute_stats.py --repo_ids RoboCOIN/Split_aloha_plate_storage RoboCOIN/Cobot_Magic_cut_banana

# Auto-discover all eligible repos
python compute_stats.py

# Only aggregate existing per-repo stats into global norm_stats.json
python compute_stats.py --only_global
```

### Statistics Format

```json
{
  "observation.state": {
    "names": ["left_arm_joint_1_rad", "..."],
    "mean": [...],
    "std": [...],
    "min": [...],
    "max": [...]
  },
  "action": { "..." },
  "action_diff": { "..." },
  "eef_sim_pose_state": { "..." },
  "eef_sim_pose_action": { "..." }
}
```

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/build_local.sh` | SLURM array job (32 workers) for the HPC parallel builder |
| `scripts/merge_shards.py` | Merges per-(repo, worker) TFDS outputs into a single flat dataset |
| `scripts/merge_shards.sh` | SLURM job for running `merge_shards.py` |
| `scripts/build.sh` | SLURM job for the sequential builder (test mode) |
| `scripts/build_bimanual_dataflow.sh` | SLURM job that launches on Google Cloud Dataflow |
| `scripts/compute_stats.sh` | SLURM job for `compute_stats.py` |
| `scripts/finalize_build.py` | Post-build cleanup/finalization |
| `scripts/monitor_dataflow.sh` | Monitors Dataflow job progress |

## Related Resources

- [RoboCOIN on Hugging Face](https://huggingface.co/RoboCOIN)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [TensorFlow Datasets Documentation](https://www.tensorflow.org/datasets)
- [RLDS Format Specification](https://github.com/google-research/rlds)

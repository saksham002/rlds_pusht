# RoboCOIN Dataset

## Overview

The RoboCOIN dataset builder merges multiple robotics datasets from the [RoboCOIN](https://huggingface.co/RoboCOIN) collection on Hugging Face into a unified RLDS format. RoboCOIN contains diverse robot manipulation demonstrations across different robot platforms, tasks, and environments with rich annotations including subtask labels and scene descriptions.

**Source**: [RoboCOIN on Hugging Face](https://huggingface.co/RoboCOIN) - All datasets prefixed with `RoboCOIN/`

The builder automatically discovers and processes all available RoboCOIN datasets from Hugging Face Hub, creating a single unified dataset with standardized features.

## Features

The dataset contains the following features per timestep:

### Observations

#### Images (up to 8 cameras, JPEG-encoded)
- **`observation/image/cam_0` through `observation/image/cam_7`**: Camera views encoded as JPEG bytes (unused cameras contain empty bytes)

#### State
- **`observation/state`**: Robot state vector `(N,)` - dimensions vary by robot type

### Actions
- **`action`**: Action vector `(N,)` - dimensions vary by robot type

### Annotations
- **`subtask_annotation`**: Subtask labels `(5,)` - integer annotations for task decomposition
- **`scene_annotation`**: Scene label (int32) - environment/scene identifier
- **`eef_sim_pose_state`**: End-effector simulation pose state `(12,)` - 6-DOF per arm
- **`eef_sim_pose_action`**: End-effector simulation pose action `(12,)` - 6-DOF per arm

### Standard RLDS Fields
- **`is_first`**: First step flag (bool)
- **`is_terminal`**: Terminal step flag (bool)
- **`frame_index`**: Frame index within episode (int64)
- **`task_index`**: Task identifier (int64)
- **`episode_index`**: Episode identifier (int64)
- **`index`**: Global step index (int64)

### Episode Metadata
- **`repo_id`**: Hugging Face repository ID (e.g., `RoboCOIN/dataset_name`)
- **`robot_type`**: Robot platform type (e.g., `franka`, `ur5`, etc.)
- **`fps`**: Frame rate (float32)
- **`camera_names`**: List of camera names
- **`camera_shapes`**: Camera resolutions `(H, W, C)` per camera
- **`num_cameras`**: Number of active cameras (int64)
- **`state_feature_names`**: Names of state vector components
- **`action_feature_names`**: Names of action vector components
- **`subtasks`**: List of subtask descriptions
- **`scene_description`**: Natural language scene description
- **`task_description`**: Natural language task description

## Dataset Processing

The `robocoin_dataset_builder.py` script:

1. **Discovers datasets**: Lists all `RoboCOIN/*` datasets from Hugging Face Hub
2. **Downloads data**: Uses `robocoin-download` CLI with rate-limit retry handling
3. **Extracts metadata**: Parses `info.json`, `tasks.jsonl`, `episodes.jsonl`, and annotation files
4. **Processes images**: Converts LeRobot format (C, H, W float) to (H, W, C uint8) JPEG
5. **Computes robot-wise statistics**: Tracks running mean/std/min/max for each robot type using Welford's algorithm
6. **Handles variable dimensions**: Supports datasets with different state/action dimensions per robot
7. **Cleans up**: Deletes downloaded data after processing to save disk space

## Building the Dataset

Prerequisites:
- Install `robocoin-download` CLI tool
- Hugging Face Hub access token (for rate limit handling)

To build the dataset:

```bash
cd robocoin
tfds build --data_dir=/path/to/output
```

Robot-wise normalization statistics are saved to:
`/data/group_data/rl/saksham3/robocoin/norm_stats/robot_stats.json` (can be changed in code).

## Statistics Format

The `robot_stats.json` file contains per-robot-type statistics:

```json
{
  "robot_type": {
    "observation.state": {
      "count": 12345,
      "min": [...],
      "max": [...],
      "mean": [...],
      "std": [...]
    },
    "action": { ... },
    "eef_sim_pose_state": { ... },
    "eef_sim_pose_action": { ... }
  }
}
```

## Related Resources

- [RoboCOIN on Hugging Face](https://huggingface.co/RoboCOIN)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [TensorFlow Datasets Documentation](https://www.tensorflow.org/datasets)
- [RLDS Format Specification](https://github.com/google-research/rlds)

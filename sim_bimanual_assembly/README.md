# Sim Bimanual Assembly Dataset

## Overview

The Sim Bimanual Assembly dataset is a simulated robotics dataset containing visuomotor control data for a particular bimanual manipulation task. The dataset features dual-arm robots performing an assembly task (double insertion) with image observations from multiple camera views, proprioceptive state information, and actions.

**Source**: Local HDF5 files from simulated environments located at `/data/group_data/rl/dexterous_robot_data/sim_double_insert_*/` (can be changed in code).

The dataset includes data from multiple collection rounds:
- `sim_double_insert_0226_hdf5`
- `sim_double_insert_round1_0520_v2_hdf5`
- `sim_double_insert_round2_0522_hdf5`
- `sim_double_insert_round3_0605_hdf5`
- `sim_double_insert_round4_0606_hdf5`
- `sim_double_insert_full_success_r4_hdf5`
- `sim_double_insert_round5_0812_hdf5`
- `sim_double_insert_full_success_r5_hdf5`
- `sim_double_insert_round6_0820_hdf5`

## Features

The dataset contains the following features per timestep:

### Observations

#### Images (4 cameras, JPEG-encoded)
- **`obses/images/left/top`**: Left arm top camera view (encoded as JPEG bytes)
- **`obses/images/left/wrist`**: Left arm wrist camera view (encoded as JPEG bytes)
- **`obses/images/right/top`**: Right arm top camera view (encoded as JPEG bytes)
- **`obses/images/right/wrist`**: Right arm wrist camera view (encoded as JPEG bytes)

#### State (per arm: left & right)
- **`obses/state/{arm}/ego_tcp_pose`**: Ego-centric TCP pose `(7,)` - position (3) + quaternion (4)
- **`obses/state/{arm}/ego_tcp_vel`**: Ego-centric TCP velocity `(6,)` - linear (3) + angular (3)
- **`obses/state/{arm}/gripper_pos`**: Gripper position (scalar)
- **`obses/state/{arm}/joint_qpos`**: Joint positions `(7,)` - 7-DOF arm
- **`obses/state/{arm}/relative2_tcp_pose`**: Relative TCP pose `(7,)` - position (3) + quaternion (4)
- **`obses/state/{arm}/relative2_tcp_vel`**: Relative TCP velocity `(6,)` - linear (3) + angular (3)
- **`obses/state/{arm}/tcp_pose`**: TCP pose `(7,)` - position (3) + quaternion (4)
- **`obses/state/{arm}/tcp_vel`**: TCP velocity `(6,)` - linear (3) + angular (3)
- **`obses/state/{arm}/wrist_tcp_pose`**: Wrist TCP pose `(7,)` - position (3) + quaternion (4)
- **`obses/state/{arm}/wrist_tcp_vel`**: Wrist TCP velocity `(6,)` - linear (3) + angular (3)

### Actions
- **`actions/global_action`**: Global action vector `(14,)` - 7 per arm (6-DOF + gripper)
- **`actions/relative_action`**: Relative action vector `(14,)` - 7 per arm (6-DOF + gripper)

### Episode Information
- **`dones`**: Episode termination flag (bool)
- **`rewards`**: Step reward (float32, 0-3 scale)
- **`truncateds`**: Truncation flag (bool)
- **`is_intervention_step`**: Whether this step was a human intervention (bool)

### Episode Metadata
- **`file_path`**: Path to the original HDF5 file

## Dataset Processing

This repository contains the `sim_bimanual_assembly_dataset_builder.py` script that:

1. **Loads HDF5 episode files**: Reads episodes from multiple data collection folders
2. **Processes images**: Decodes compressed images and re-encodes as JPEG for storage efficiency
3. **Handles interventions**: Marks intervention steps based on episode metadata
4. **Corrects rewards**: Normalizes reward scale from 0-4 to 0-3 where applicable
5. **Computes normalization statistics**: Generates mean, std, min, max for all action and state features
6. **Splits data**: 90% training / 10% validation split per folder (deterministic with seed)

## Building the Dataset

To convert the HDF5 files to TFRecords for use with dlimp:

```bash
cd sim_bimanual_assembly
tfds build --data_dir=/path/to/output
```

Normalization statistics are saved to:
`/data/group_data/rl/saksham3/sim_bimanual_assembly/norm_stats/norm_stats.npy` (can be changed in code).

## Related Resources

- [TensorFlow Datasets Documentation](https://www.tensorflow.org/datasets)
- [RLDS Format Specification](https://github.com/google-research/rlds)

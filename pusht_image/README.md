# PushT Image Dataset

## Overview

The PushT image dataset is a robotics dataset from [LeRobot](https://github.com/huggingface/lerobot) containing visuomotor control data for a pushing task. This dataset is part of the [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) research work and provides image observations along with state information, actions, and rewards for learning manipulation policies.

**Source**: [lerobot/pusht_image on Hugging Face](https://huggingface.co/datasets/lerobot/pusht_image)

The dataset was created using LeRobot v2.0 and contains time-series image data collected from a robotic pushing task. Each sample includes RGB image observations, state vectors, actions, rewards, and episode metadata.

## Dataset Statistics

- **Total Episodes**: 206
- **Total Frames**: 25,650
- **Frame Rate**: 10 FPS
- **Total Samples**: 48,336 rows
- **Dataset Size**: 56.2 MB (Parquet format)
- **Format**: Parquet
- **License**: MIT

## Features

The dataset contains the following features:

### Observations
- **`observation.image`**: RGB image observations with shape `(96, 96, 3)` (height, width, channels)
- **`observation.state`**: State vector with shape `(2,)` containing motor positions:
  - `motor_0`
  - `motor_1`

### Actions
- **`action`**: Action vector with shape `(2,)` containing motor commands:
  - `motor_0`
  - `motor_1`

### Episode Information
- **`episode_index`**: Episode identifier (int64)
- **`frame_index`**: Frame index within the episode (int64)
- **`timestamp`**: Timestamp in seconds (float32)
- **`task_index`**: Task identifier (int64, currently 0)

### Reward and Termination
- **`next.reward`**: Reward value (float32, typically 0 to 1)
- **`next.done`**: Episode termination flag (bool)
- **`next.success`**: Task success flag (bool)

## Dataset Processing

This repository contains preprocessing scripts that:

1. **Load and combine datasets**: The `load_data_from_huggingface.py` script:
   - Loads the `pusht_image` dataset from Hugging Face
   - Loads the corresponding `pusht_keypoints` dataset (pusht_state)
   - Sets `next.done` flags correctly for both datasets
   - Removes duplicate `(episode_index, frame_index)` pairs from the image dataset

2. **Preprocess data**: The `preprocess_pusht.py` script:
   - Concatenates `observation.state` from the image dataset with `observation.environment_state` from the state dataset
   - Transforms `observation.state` to contain `(s_{t-1}, s_t)` pairs with shape `(N, 2, 18)`
   - Transforms `observation.image` to contain `(img_{t-1}, img_t)` pairs with shape `(N, 2, 96, 96, 3)`
   - Creates action horizons of length 16 with shape `(N, 16, 2)`
   - Adds padding masks for states (`state_is_pad`) and actions (`action_is_pad`)
   - Uses `next.reward` from the state dataset to compute final rewards
   - Validates consistency between image and state datasets through sanity checks
   - Saves preprocessed data as individual episode files in `/data/group_data/rl/saksham3/pusht/episode_data/episode_<i>.npy`

Each preprocessed episode file contains a list of dictionaries, where each dictionary represents one timestep and includes all preprocessed features.

## Citation

```bibtex
@article{chi2024diffusionpolicy,
    author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
    title ={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
    journal = {The International Journal of Robotics Research},
    year = {2024},
}
```

**Paper**: [arxiv:2303.04137](https://arxiv.org/abs/2303.04137)  
**Project Page**: [diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu/)

## Related Resources

- [Diffusion Policy GitHub Repository](https://github.com/real-stanford/diffusion_policy)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)

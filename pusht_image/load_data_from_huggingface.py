from datasets import load_dataset
import numpy as np
import os
import pdb

# --- Process pusht_keypoints dataset ---

# Sets done correctly for both state and image datasets.
# Removes duplicate (episode_index, frame_index) pairs from the image dataset.

# 1. Load the dataset from Hugging Face Hub
ds = load_dataset("lerobot/pusht_keypoints")
train_ds = ds["train"]

# 2. Convert to a dict of arrays
data_dict = {
    key: np.array(train_ds[key]) for key in train_ds.features.keys()
}

# 3. Modify next.done
# episode_indices = data_dict['episode_index']
# frame_indices = data_dict['frame_index']
# new_next_done = np.zeros_like(data_dict['next.done'], dtype=bool)
# unique_episodes = np.unique(episode_indices)

# for episode_id in unique_episodes:
#     episode_mask = (episode_indices == episode_id)
#     episode_frame_indices = frame_indices[episode_mask]
#     if len(episode_frame_indices) > 0:
#         last_frame_in_episode_idx = np.argmax(episode_frame_indices)
#         # Get the global index
#         global_indices = np.where(episode_mask)[0]
#         last_frame_global_idx = global_indices[last_frame_in_episode_idx]
#         new_next_done[last_frame_global_idx] = True

# data_dict['next.done'] = new_next_done

# 4. Save to .npy file
p = "/data/group_data/rl/saksham3/pusht/pusht_state.npy"
os.makedirs(os.path.dirname(p), exist_ok = True)
np.save(p, data_dict)
print(f"Saved PushT state dataset to: {p}")


# --- Process pusht_image dataset ---

# 1. Load the dataset from Hugging Face Hub
ds = load_dataset("lerobot/pusht_image")
train_ds = ds["train"]

# 2. Convert to a dict of arrays
data_dict = {
}

# 3. Remove duplicate (episode_index, frame_index) pairs
episode_indices = train_ds['episode_index']
frame_indices = train_ds['frame_index']
# Stack arrays and find unique pairs
stacked_pairs = np.column_stack([episode_indices, frame_indices])
_, unique_indices = np.unique(stacked_pairs, axis = 0, return_index = True)

# Filter all arrays in data_dict using the unique_indices
for key in train_ds.features.keys():
    if key == 'observation.image':
        images = [np.array(img.convert("RGB")) for img in train_ds["observation.image"][unique_indices]]
        data_dict[key] = np.stack(images)
    else:
        data_dict[key] = np.array(train_ds[key])[unique_indices]
    print(f"key: {key}, shape: {data_dict[key].shape}")


# 4. Modify next.done for the deduplicated data
# episode_indices = data_dict['episode_index']
# frame_indices = data_dict['frame_index']
# new_next_done = np.zeros_like(data_dict['next.done'], dtype=bool)
# unique_episodes = np.unique(episode_indices)

# for episode_id in unique_episodes:
#     episode_mask = (episode_indices == episode_id)
#     episode_frame_indices = frame_indices[episode_mask]
#     if len(episode_frame_indices) > 0:
#         last_frame_in_episode_idx = np.argmax(episode_frame_indices)
#         # Get the global index from the deduplicated data
#         global_indices = np.where(episode_mask)[0]
#         last_frame_global_idx = global_indices[last_frame_in_episode_idx]
#         new_next_done[last_frame_global_idx] = True

# data_dict['next.done'] = new_next_done

# 5. Save to .npy file
p = "/data/group_data/rl/saksham3/pusht/pusht_image.npy"
os.makedirs(os.path.dirname(p), exist_ok = True)
np.save(p, data_dict)
print(f"Saved PushT image dataset to: {p}")

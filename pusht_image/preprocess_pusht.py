import numpy as np
import os

import pdb


# Sets rewards according to the state-only dataset.
# Sets 18-D states for state-only dataset.
# Sets ground truth actions over a horizon of length 16, sets (states, images) to include previous timestep observations.
# Creates padding masks for the states and actions.

def preprocess_pusht_data(is_state_only: bool = True):
    """
    Loads, preprocesses, and combines the PushT image and state datasets.
    """
    # 1. Load the datasets
    try:
        if not is_state_only:
            pusht_image_data = np.load('/data/group_data/rl/saksham3/pusht/pusht_image.npy', allow_pickle=True).item()
        pusht_state_data = np.load('/data/group_data/rl/saksham3/pusht/pusht_state.npy', allow_pickle=True).item()
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    # 2. Sanity check the observation.state and action values
    # state_diff_norm = np.linalg.norm(pusht_image_data['observation.state'] - pusht_state_data['observation.state'])
    # if state_diff_norm > 1e-5:
    #     print(f"Warning: Sanity check failed. The norm of the difference between observation.state is {state_diff_norm}")
    # else:
    #     print("Sanity check passed: observation.state values are consistent.")
    
    # action_diff_norm = np.linalg.norm(pusht_image_data['action'] - pusht_state_data['action'])
    # if action_diff_norm > 1e-5:
    #     print(f"Warning: Sanity check failed. The norm of the difference between action is {action_diff_norm}")
    # else:
    #     print("Sanity check passed: action values are consistent.")
    
    # # Check episode_index and frame_index match
    # episode_index_match = np.array_equal(pusht_image_data['episode_index'], pusht_state_data['episode_index'])
    # frame_index_match = np.array_equal(pusht_image_data['frame_index'], pusht_state_data['frame_index'])
    # if not episode_index_match or not frame_index_match:
    #     print(f"Warning: Sanity check failed. episode_index match: {episode_index_match}, frame_index match: {frame_index_match}")
    #     if not episode_index_match:
    #         episode_diff_count = np.sum(pusht_image_data['episode_index'] != pusht_state_data['episode_index'])
    #         print(f"  Number of mismatched episode_index values: {episode_diff_count}")
    #     if not frame_index_match:
    #         frame_diff_count = np.sum(pusht_image_data['frame_index'] != pusht_state_data['frame_index'])
    #         print(f"  Number of mismatched frame_index values: {frame_diff_count}")
    # else:
    #     print("Sanity check passed: episode_index and frame_index values are consistent.")

    # 3. Create a new dictionary for the preprocessed data
    preprocessed_data = pusht_image_data.copy() if not is_state_only else pusht_state_data.copy()

    # Get episode information
    episode_indices = preprocessed_data['episode_index']
    unique_episodes = np.unique(episode_indices)
    num_samples = len(episode_indices)
    
    if is_state_only:
        preprocessed_data['observation.state'] = np.concatenate([preprocessed_data['observation.state'], preprocessed_data['observation.environment_state']], axis = 1)

    # Initialize new arrays
    new_obs_state = np.zeros((num_samples, 2, 2), dtype = np.float32) if not is_state_only else np.zeros((num_samples, 2, 18), dtype = np.float32)
    new_obs_image = np.zeros((num_samples, 2, 96, 96, 3), dtype = np.uint8) if not is_state_only else None
    state_is_pad = np.zeros((num_samples, 2), dtype = bool)
    new_action = np.zeros((num_samples, 16, 2), dtype = np.float32)
    action_is_pad = np.zeros((num_samples, 16), dtype = bool)

    # Process each episode
    for episode_id in unique_episodes:
        episode_mask = (episode_indices == episode_id)
        episode_indices_in_data = np.where(episode_mask)[0]
        
        episode_states = preprocessed_data['observation.state'][episode_mask]
        episode_actions = preprocessed_data['action'][episode_mask]
        
        # Redefine observation.state to be (s_{t-1}, s_t)
        new_obs_state[episode_indices_in_data, 1, :] = episode_states
        new_obs_state[episode_indices_in_data[1:], 0, :] = episode_states[:-1]
        new_obs_state[episode_indices_in_data[0], 0, :] = episode_states[0]
        state_is_pad[episode_indices_in_data[0], 0] = True
        
        # Redefine observation.image to be (img_{t-1}, img_t)
        if not is_state_only:
            episode_images = preprocessed_data['observation.image'][episode_mask]
            new_obs_image[episode_indices_in_data, 1, :, :, :] = episode_images
            new_obs_image[episode_indices_in_data[1:], 0, :, :, :] = episode_images[:-1]
            new_obs_image[episode_indices_in_data[0], 0, :, :, :] = episode_images[0]

        # Redefine action to be an action horizon of length 16
        num_episode_steps = len(episode_actions)
        # Create indices for the sliding window
        indices = np.arange(16) + np.arange(num_episode_steps)[:, np.newaxis]
        # Clip indices to num_episode_steps - 1 so that padded_actions is not needed
        clipped_indices = np.clip(indices, 0, num_episode_steps - 1)
        new_action[episode_indices_in_data] = episode_actions[clipped_indices]

        # Create the padding mask
        is_pad_mask = indices >= num_episode_steps
        action_is_pad[episode_indices_in_data] = is_pad_mask

        preprocessed_data['next.done'][episode_indices_in_data][-1] = True
        preprocessed_data['next.done'][episode_indices_in_data][-2] = False

    # Update the dictionary with the new arrays
    preprocessed_data['observation.state'] = new_obs_state
    if not is_state_only:
        preprocessed_data['observation.image'] = new_obs_image
    preprocessed_data['action'] = new_action
    preprocessed_data['state_is_pad'] = state_is_pad
    preprocessed_data['action_is_pad'] = action_is_pad
    # reward in pusht_image_data is wrong so state_data has to be used for both datasets
    preprocessed_data['next.done'] = preprocessed_data['next.done'].astype(np.bool)
    preprocessed_data['reward'] = (preprocessed_data['next.done']).astype(np.float32)
    preprocessed_data['next.reward'] = pusht_state_data['next.reward']

    # Save each episode to a separate file
    output_dir = '/data/group_data/rl/saksham3/pusht/episode_data_' + ('state' if is_state_only else 'image')  
    os.makedirs(output_dir, exist_ok = True)
    
    for episode_id in unique_episodes:
        episode_mask = (episode_indices == episode_id)
        episode_indices_in_data = np.where(episode_mask)[0]
        episode_len = len(episode_indices_in_data)
        
        # Create a list of dictionaries for this episode
        episode_data = []
        for i, idx in enumerate(episode_indices_in_data):
            timestep_dict = {}
            for key in preprocessed_data.keys():
                timestep_dict[key] = preprocessed_data[key][idx]
            episode_data.append(timestep_dict)
        
        # Save to file
        output_path = os.path.join(output_dir, f'episode_{episode_id}.npy')
        np.save(output_path, episode_data)
        print(f"Saved episode {episode_id} to {output_path}")

    np.save("/data/group_data/rl/saksham3/pusht/pusht_" + ("state" if is_state_only else "image") + ".npy", preprocessed_data)

if __name__ == '__main__':
    preprocess_pusht_data(True)
    preprocess_pusht_data(False)
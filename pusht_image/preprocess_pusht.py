import numpy as np
import os

import pdb

def preprocess_pusht_data():
    """
    Loads, preprocesses, and combines the PushT image and state datasets.
    """
    # 1. Load the datasets
    try:
        pusht_image_data = np.load('/data/group_data/rl/saksham3/pusht/pusht_image.npy', allow_pickle=True).item()
        pusht_state_data = np.load('/data/group_data/rl/saksham3/pusht/pusht_state.npy', allow_pickle=True).item()
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # 2. Sanity check the observation.state and action values
    state_diff_norm = np.linalg.norm(pusht_image_data['observation.state'] - pusht_state_data['observation.state'])
    if state_diff_norm > 1e-5:
        print(f"Warning: Sanity check failed. The norm of the difference between observation.state is {state_diff_norm}")
    else:
        print("Sanity check passed: observation.state values are consistent.")
    
    action_diff_norm = np.linalg.norm(pusht_image_data['action'] - pusht_state_data['action'])
    if action_diff_norm > 1e-5:
        print(f"Warning: Sanity check failed. The norm of the difference between action is {action_diff_norm}")
    else:
        print("Sanity check passed: action values are consistent.")
    
    # Check episode_index and frame_index match
    episode_index_match = np.array_equal(pusht_image_data['episode_index'], pusht_state_data['episode_index'])
    frame_index_match = np.array_equal(pusht_image_data['frame_index'], pusht_state_data['frame_index'])
    if not episode_index_match or not frame_index_match:
        print(f"Warning: Sanity check failed. episode_index match: {episode_index_match}, frame_index match: {frame_index_match}")
        if not episode_index_match:
            episode_diff_count = np.sum(pusht_image_data['episode_index'] != pusht_state_data['episode_index'])
            print(f"  Number of mismatched episode_index values: {episode_diff_count}")
        if not frame_index_match:
            frame_diff_count = np.sum(pusht_image_data['frame_index'] != pusht_state_data['frame_index'])
            print(f"  Number of mismatched frame_index values: {frame_diff_count}")
    else:
        print("Sanity check passed: episode_index and frame_index values are consistent.")

    # 3. Create a new dictionary for the preprocessed data
    preprocessed_data = pusht_image_data.copy()

    # 4. Concatenate observation.state and observation.environment_state
    concatenated_state = np.concatenate([
        pusht_image_data['observation.state'],
        pusht_state_data['observation.environment_state']
    ], axis=1)
    preprocessed_data['observation.state'] = concatenated_state

    # Get episode information
    episode_indices = preprocessed_data['episode_index']
    unique_episodes = np.unique(episode_indices)
    num_samples = len(episode_indices)

    # Initialize new arrays
    new_obs_state = np.zeros((num_samples, 2, 18), dtype=np.float32)
    new_obs_image = np.zeros((num_samples, 2, 96, 96, 3), dtype=preprocessed_data['observation.image'].dtype)
    state_is_pad = np.zeros((num_samples, 2), dtype=bool)
    new_action = np.zeros((num_samples, 16, 2), dtype=np.float32)
    action_is_pad = np.zeros((num_samples, 16), dtype=bool)

    # Process each episode
    for episode_id in unique_episodes:
        episode_mask = (episode_indices == episode_id)
        episode_indices_in_data = np.where(episode_mask)[0]
        
        episode_states = preprocessed_data['observation.state'][episode_mask]
        episode_images = preprocessed_data['observation.image'][episode_mask]
        episode_actions = preprocessed_data['action'][episode_mask]
        
        # 5. Redefine observation.state to be (s_{t-1}, s_t)
        new_obs_state[episode_indices_in_data, 1, :] = episode_states
        new_obs_state[episode_indices_in_data[1:], 0, :] = episode_states[:-1]
        state_is_pad[episode_indices_in_data[0], 0] = True
        
        # Redefine observation.image to be (img_{t-1}, img_t)
        new_obs_image[episode_indices_in_data, 1, :, :, :] = episode_images
        new_obs_image[episode_indices_in_data[1:], 0, :, :, :] = episode_images[:-1]

        # 6. Redefine action to be an action horizon of length 16
        num_episode_steps = len(episode_actions)
        # Create indices for the sliding window
        indices = np.arange(16) + np.arange(num_episode_steps)[:, np.newaxis]
        # Pad actions with zeros for easier indexing.
        # The padded part should not be reachable by valid indices.
        padded_actions = np.concatenate([episode_actions, np.zeros((16, 2), dtype=episode_actions.dtype)])
        new_action[episode_indices_in_data] = padded_actions[indices]

        # Create the padding mask
        is_pad_mask = indices >= num_episode_steps
        action_is_pad[episode_indices_in_data] = is_pad_mask

    # Update the dictionary with the new arrays
    preprocessed_data['observation.state'] = new_obs_state
    preprocessed_data['observation.image'] = new_obs_image
    preprocessed_data['action'] = new_action
    preprocessed_data['state_is_pad'] = state_is_pad
    preprocessed_data['action_is_pad'] = action_is_pad
    preprocessed_data['reward'] = (preprocessed_data['next.done'] & (pusht_state_data['next.reward'] > 0.9025)).astype(np.float32)
    preprocessed_data['next.reward'] = pusht_state_data['next.reward']

    # Save each episode to a separate file
    output_dir = '/data/group_data/rl/saksham3/pusht/episode_data'
    os.makedirs(output_dir, exist_ok=True)
    
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

if __name__ == '__main__':
    preprocess_pusht_data()

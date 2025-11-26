import glob
import h5py
import numpy as np
import os
import pprint

def find_max_stats():
    """
    Finds directory-wise maximums for image dimensions, overall reward,
    and terminal reward across HDF5 episode files.
    """
    paths = [
        '/data/group_data/rl/dexterous_robot_data/sim_double_insert_0226_hdf5/episode_*.hdf5',
        '/data/group_data/rl/dexterous_robot_data/sim_double_insert_round1_0520_v2_hdf5/episode_*.hdf5',
        '/data/group_data/rl/dexterous_robot_data/sim_double_insert_round2_0522_hdf5/episode_*.hdf5',
        '/data/group_data/rl/dexterous_robot_data/sim_double_insert_round3_0605_hdf5/episode_*.hdf5',
        '/data/group_data/rl/dexterous_robot_data/sim_double_insert_round4_0606_hdf5/episode_*.hdf5',
        '/data/group_data/rl/dexterous_robot_data/sim_double_insert_full_success_r4_hdf5/episode_*.hdf5',
        '/data/group_data/rl/dexterous_robot_data/sim_double_insert_round5_0812_hdf5/episode_*.hdf5',
        '/data/group_data/rl/dexterous_robot_data/sim_double_insert_full_success_r5_hdf5/episode_*.hdf5',
        '/data/group_data/rl/dexterous_robot_data/sim_double_insert_round6_0820_hdf5/episode_*.hdf5'
    ]

    image_keys = [
        "obses/images/left/top",
        "obses/images/left/wrist",
        "obses/images/right/top",
        "obses/images/right/wrist"
    ]

    # Extract unique directory paths from the glob patterns
    dir_paths = sorted(list(set([os.path.dirname(p) for p in paths])))
    all_results = {}

    for dir_path in dir_paths:
        dir_name = os.path.basename(dir_path)
        episode_paths = glob.glob(os.path.join(dir_path, 'episode_*.hdf5'))

        if not episode_paths:
            print(f"No HDF5 files found in {dir_path}. Skipping.")
            continue

        # Initialize stats for this directory
        dir_stats = {
            'max_image_dims': {key: 0 for key in image_keys},
            'max_terminal_reward': -np.inf,
            'max_reward': -np.inf
        }

        print(f"Processing directory: {dir_name} ({len(episode_paths)} episodes)...")

        for episode_path in episode_paths:
            try:
                with h5py.File(episode_path, 'r') as f:
                    # 1. Update max image dimensions
                    for key in image_keys:
                        if key in f:
                            dataset = f[key]
                            if len(dataset.shape) > 1:
                                second_dim = dataset.shape[1]
                                if second_dim > dir_stats['max_image_dims'][key]:
                                    dir_stats['max_image_dims'][key] = second_dim

                    # 2. Update max rewards
                    if 'rewards/rewards' in f:
                        rewards = f['rewards/rewards'][:]
                        if rewards.size > 0:
                            # Update max terminal reward
                            terminal_reward = rewards[-1]
                            if terminal_reward > dir_stats['max_terminal_reward']:
                                dir_stats['max_terminal_reward'] = terminal_reward
                            
                            # Update max overall reward
                            max_in_episode = np.max(rewards)
                            if max_in_episode > dir_stats['max_reward']:
                                dir_stats['max_reward'] = max_in_episode
            except Exception as e:
                print(f"  Error processing file {episode_path}: {e}")

        all_results[dir_name] = dir_stats

    # Print the final results
    print("\n" + "="*50)
    print("      DIRECTORY-WISE MAXIMUMS      ")
    print("="*50)
    if not all_results:
        print("No data processed.")
    else:
        pprint.pprint(all_results)
    print("="*50 + "\n")


if __name__ == "__main__":
    find_max_stats()
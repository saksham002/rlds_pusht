"""Example slurm_rlds config: PushT Image dataset.

To run two workers locally:
    cd slurm_rlds/
    python -m framework.runner --config example_config.py --data_dir /tmp/test/0 --worker_id 0 --num_workers 2
    python -m framework.runner --config example_config.py --data_dir /tmp/test/1 --worker_id 1 --num_workers 2

Then merge:
    python scripts/merge_shards.py \\
        --root /tmp/test \\
        --output /tmp/test_merged \\
        --num_workers 2 \\
        --dataset_name pusht_image \\
        --dataset_version 1.0.0
"""
import glob

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

DATASET_NAME    = 'pusht_image'
DATASET_VERSION = '1.0.0'

_EPISODE_GLOB = '/data/group_data/rl/saksham3/pusht/episode_data_image/episode_*.npy'


def get_features():
    return tfds.features.FeaturesDict({
        'steps': tfds.features.Dataset({
            'observation.image': tfds.features.Tensor(
                shape = (2, 96, 96, 3),
                dtype = np.int64,
                doc   = 'Image of the current environment state and previous state.',
            ),
            'observation.state': tfds.features.Tensor(
                shape = (2, 2),
                dtype = np.float32,
                doc   = 'Current and previous gripper and T-block positions.',
            ),
            'action': tfds.features.Tensor(
                shape = (16, 2),
                dtype = np.float32,
                doc   = 'Next 16 actions in the action horizon.',
            ),
            'discount': tfds.features.Scalar(
                dtype = np.float32,
                doc   = 'Discount if provided, default to 0.99.',
            ),
            'reward': tfds.features.Scalar(
                dtype = np.float32,
                doc   = 'Reward if provided, 1 on final step for demos that cross a reward threshold of 0.9025.',
            ),
            'next.reward': tfds.features.Scalar(
                dtype = np.float32,
                doc   = 'Percentage of the goal state covered divided by 0.95.',
            ),
            'next.done': tfds.features.Scalar(
                dtype = np.int64,
                doc   = 'True on last 2 steps of the episode.',
            ),
            'episode_index': tfds.features.Scalar(dtype = np.int64),
            'frame_index':   tfds.features.Scalar(dtype = np.int64),
            'timestamp':     tfds.features.Scalar(dtype = np.float32),
            'task_index':    tfds.features.Scalar(dtype = np.int64),
            'index':         tfds.features.Scalar(dtype = np.int64),
            'next.success':  tfds.features.Scalar(dtype = np.int64),
            'state_is_pad':  tfds.features.Tensor(shape = (2,),  dtype = np.int64),
            'action_is_pad': tfds.features.Tensor(shape = (16,), dtype = np.int64),
            'episode_length': tfds.features.Scalar(dtype = np.int64),
        }),
        'episode_metadata': tfds.features.FeaturesDict({
            'file_path': tfds.features.Text(doc = 'Path to the original data file.'),
        }),
    })


def get_episodes(split):
    """Return a deterministically ordered list of episode paths for the given split.

    PushT has only a train split, so val returns an empty list.
    """
    if split != 'train':
        return []
    return sorted(glob.glob(_EPISODE_GLOB))


def parse_episode(episode_path):
    """Load one episode .npy file and return (key, example_dict)."""
    data = np.load(episode_path, allow_pickle = True)
    episode = []
    for step in data:
        episode.append({
            'observation.image':  np.asarray(step['observation.image'],  dtype = np.int64),
            'observation.state':  np.asarray(step['observation.state'],  dtype = np.float32),
            'action':             np.asarray(step['action'],             dtype = np.float32),
            'discount':           np.float32(0.99),
            'reward':             np.asarray(step['reward'],             dtype = np.float32),
            'next.reward':        np.asarray(step['next.reward'],        dtype = np.float32),
            'next.done':          np.asarray(step['next.done'],          dtype = np.int64),
            'episode_index':      np.asarray(step['episode_index'],      dtype = np.int64),
            'frame_index':        np.asarray(step['frame_index'],        dtype = np.int64),
            'timestamp':          np.asarray(step['timestamp'],          dtype = np.float32),
            'task_index':         np.asarray(step['task_index'],         dtype = np.int64),
            'index':              np.asarray(step['index'],              dtype = np.int64),
            'next.success':       np.asarray(step['next.success'],       dtype = np.int64),
            'state_is_pad':       np.asarray(step['state_is_pad'],       dtype = np.int64),
            'action_is_pad':      np.asarray(step['action_is_pad'],      dtype = np.int64),
            'episode_length':     np.int64(len(data)),
        })

    sample = {
        'steps': episode,
        'episode_metadata': {
            'file_path': tf.constant(episode_path, dtype = tf.string).numpy(),
        },
    }
    return episode_path, sample

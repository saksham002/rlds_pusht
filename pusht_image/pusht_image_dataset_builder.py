from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class PushTImage(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for PushTImage dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation.image': tfds.features.Tensor(
                        shape=(2, 96, 96, 3),
                        dtype=np.uint8,
                        doc='Image of the current environment state and previous state.',
                    ),
                    'observation.state': tfds.features.Tensor(
                        shape=(2,2),
                        dtype=np.float32,
                        doc='Current and previous gripper and T-block positions.',
                    ),
                    'action': tfds.features.Tensor(
                        shape=(16,2),
                        dtype=np.float32,
                        doc='Next 16 actions in the action horizon.'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 0.99.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos that cross a reward threshold of 0.9025.'
                    ),
                    'next.reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Percentage of the goal state are covered divided by 0.95.'
                    ),
                    'next.done': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last 2 steps of the episode (last step is always unused, only for value function estimation).'
                    ),
                    'episode_index': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Index of the episode.'
                    ),
                    'frame_index': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Index of the frame within the episode.'
                    ),
                    'timestamp': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Timestamp of the frame.'
                    ),
                    'task_index': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Index of the task. Always 0 for this dataset.'
                    ),
                    'index': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Overall index of the frame within the dataset.'
                    ),
                    'next.success': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='False always, unsure of meaning.'
                    ),
                    'state_is_pad': tfds.features.Tensor(
                        shape=(2,),
                        dtype=np.bool_,
                        doc='Padding mask for state/image input.'
                    ),
                    'action_is_pad': tfds.features.Tensor(
                        shape=(16,),
                        dtype=np.bool_,
                        doc='Padding mask for action outputs.'
                    ),
                    'episode_length': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Length of the episode.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/data/group_data/rl/saksham3/pusht/episode_data_image/episode_*.npy'),
            # val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle = True)     # this is a list of dicts in our case

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                # compute Kona language embedding
                # language_embedding = self._embed([step['language_instruction']])[0].numpy()

                episode.append({
                    'observation.image': step['observation.image'],
                    'observation.state': step['observation.state'],
                    'action': step['action'],
                    'discount': 0.99,
                    'reward': step['reward'],
                    'next.reward': step['next.reward'],
                    'next.done': step['next.done'],
                    'episode_index': step['episode_index'],
                    'frame_index': step['frame_index'],
                    'timestamp': step['timestamp'],
                    'task_index': step['task_index'],
                    'index': step['index'],
                    'next.success': step['next.success'],
                    'state_is_pad': step['state_is_pad'],
                    'action_is_pad': step['action_is_pad'],
                    'episode_length': len(data),
                    # 'language_instruction': step['language_instruction'],
                    # 'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

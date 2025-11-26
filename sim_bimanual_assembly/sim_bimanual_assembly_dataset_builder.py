from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import h5py
import pdb
import tensorflow_hub as hub
import cv2

MAX_COMPRESSED_DIMENSION = 20000

VALID_KEYS = {
    'actions/global_action', 'actions/relative_action', 'dones', 'obses/images/left/top',
    'obses/images/left/wrist', 'obses/images/right/top', 'obses/images/right/wrist',
    'obses/state/left/ego_tcp_pose', 'obses/state/left/ego_tcp_vel',
    'obses/state/left/gripper_pos', 'obses/state/left/joint_qpos',
    'obses/state/left/relative2_tcp_pose', 'obses/state/left/relative2_tcp_vel',
    'obses/state/left/tcp_pose', 'obses/state/left/tcp_vel',
    'obses/state/left/wrist_tcp_pose', 'obses/state/left/wrist_tcp_vel',
    'obses/state/right/ego_tcp_pose', 'obses/state/right/ego_tcp_vel',
    'obses/state/right/gripper_pos', 'obses/state/right/joint_qpos',
    'obses/state/right/relative2_tcp_pose', 'obses/state/right/relative2_tcp_vel',
    'obses/state/right/tcp_pose', 'obses/state/right/tcp_vel',
    'obses/state/right/wrist_tcp_pose', 'obses/state/right/wrist_tcp_vel', 'rewards',
    'truncateds'
}


class SimBimanualAssembly(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for SimBimanualAssembly dataset."""

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
                    'actions/global_action': tfds.features.Tensor(shape=(14,), dtype=np.float32),
                    'actions/relative_action': tfds.features.Tensor(shape=(14,), dtype=np.float32),
                    'dones': tfds.features.Scalar(dtype=np.bool_),
                    # 'obses/images/left/top': tfds.features.Tensor(shape=(MAX_COMPRESSED_DIMENSION,), dtype=np.uint8),
                    # 'obses/images/left/wrist': tfds.features.Tensor(shape=(MAX_COMPRESSED_DIMENSION,), dtype=np.uint8),
                    # 'obses/images/right/top': tfds.features.Tensor(shape=(MAX_COMPRESSED_DIMENSION,), dtype=np.uint8),
                    # 'obses/images/right/wrist': tfds.features.Tensor(shape=(MAX_COMPRESSED_DIMENSION,), dtype=np.uint8),
                    'obses/images/left/top': tfds.features.Tensor(shape=(), dtype=tf.string),
                    'obses/images/left/wrist': tfds.features.Tensor(shape=(), dtype=tf.string),
                    'obses/images/right/top': tfds.features.Tensor(shape=(), dtype=tf.string),
                    'obses/images/right/wrist': tfds.features.Tensor(shape=(), dtype=tf.string),
                    'obses/state/left/ego_tcp_pose': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'obses/state/left/ego_tcp_vel': tfds.features.Tensor(shape=(6,), dtype=np.float32),
                    'obses/state/left/gripper_pos': tfds.features.Scalar(dtype=np.float32),
                    'obses/state/left/joint_qpos': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'obses/state/left/relative2_tcp_pose': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'obses/state/left/relative2_tcp_vel': tfds.features.Tensor(shape=(6,), dtype=np.float32),
                    'obses/state/left/tcp_pose': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'obses/state/left/tcp_vel': tfds.features.Tensor(shape=(6,), dtype=np.float32),
                    'obses/state/left/wrist_tcp_pose': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'obses/state/left/wrist_tcp_vel': tfds.features.Tensor(shape=(6,), dtype=np.float32),
                    'obses/state/right/ego_tcp_pose': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'obses/state/right/ego_tcp_vel': tfds.features.Tensor(shape=(6,), dtype=np.float32),
                    'obses/state/right/gripper_pos': tfds.features.Scalar(dtype=np.float32),
                    'obses/state/right/joint_qpos': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'obses/state/right/relative2_tcp_pose': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'obses/state/right/relative2_tcp_vel': tfds.features.Tensor(shape=(6,), dtype=np.float32),
                    'obses/state/right/tcp_pose': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'obses/state/right/tcp_vel': tfds.features.Tensor(shape=(6,), dtype=np.float32),
                    'obses/state/right/wrist_tcp_pose': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'obses/state/right/wrist_tcp_vel': tfds.features.Tensor(shape=(6,), dtype=np.float32),
                    'rewards': tfds.features.Scalar(dtype=np.float32),
                    'truncateds': tfds.features.Scalar(dtype=np.bool_),
                    'is_intervention_step': tfds.features.Scalar(dtype=np.bool_),
                    # 'padding_length/left/top': tfds.features.Scalar(dtype=np.int64),
                    # 'padding_length/left/wrist': tfds.features.Scalar(dtype=np.int64),
                    # 'padding_length/right/top': tfds.features.Scalar(dtype=np.int64),
                    # 'padding_length/right/wrist': tfds.features.Scalar(dtype=np.int64),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    # 'interventions': tfds.features.Tensor(shape=(), dtype=np.int64),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(paths=['/data/group_data/rl/dexterous_robot_data/sim_double_insert_0226_hdf5/episode_*.hdf5', '/data/group_data/rl/dexterous_robot_data/sim_double_insert_round1_0520_v2_hdf5/episode_*.hdf5', '/data/group_data/rl/dexterous_robot_data/sim_double_insert_round2_0522_hdf5/episode_*.hdf5', '/data/group_data/rl/dexterous_robot_data/sim_double_insert_round3_0605_hdf5/episode_*.hdf5', '/data/group_data/rl/dexterous_robot_data/sim_double_insert_round4_0606_hdf5/episode_*.hdf5', '/data/group_data/rl/dexterous_robot_data/sim_double_insert_full_success_r4_hdf5/episode_*.hdf5', '/data/group_data/rl/dexterous_robot_data/sim_double_insert_round5_0812_hdf5/episode_*.hdf5', '/data/group_data/rl/dexterous_robot_data/sim_double_insert_full_success_r5_hdf5/episode_*.hdf5', '/data/group_data/rl/dexterous_robot_data/sim_double_insert_round6_0820_hdf5/episode_*.hdf5']),
            # 'train': self._generate_examples(paths=['/data/group_data/rl/dexterous_robot_data/sim_double_insert_0226_hdf5/episode_0.hdf5']),
            # val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            with h5py.File(episode_path, 'r') as f:
                steps = {}
                episode_metadata = {}

                def visit_all(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        key = name
                        # rename keys like "rewards/rewards" to "rewards"
                        parts = key.split('/')
                        if len(parts) == 2 and parts[0] == parts[1]:
                            key = parts[0]
                        
                        if key.startswith('metadata/'):
                            if key == 'metadata/interventions':
                                episode_metadata['interventions'] = obj[()]
                        elif key in VALID_KEYS:
                            steps[key] = obj[()]
                        else:
                            pdb.set_trace()
                
                f.visititems(visit_all)

                steps['is_intervention_step'] = np.zeros_like(steps['rewards'], dtype = np.bool_)
                
                if 'interventions' in episode_metadata:
                    steps['is_intervention_step'][episode_metadata['interventions']] = True
                    episode_metadata.pop('interventions')

                episode_metadata['file_path'] = episode_path
            
            # Pad image observations to MAX_COMPRESSED_DIMENSION and record padding lengths
            image_keys = [
                'obses/images/left/top',
                'obses/images/left/wrist',
                'obses/images/right/top',
                'obses/images/right/wrist'
            ]

            if np.abs(np.max(steps['rewards']) - 4.0) < 1e-3:
                steps['rewards'][np.where(steps['rewards'] == 2.0)] = 1.0
                steps['rewards'][np.where(steps['rewards'] == 3.0)] = 2.0
                steps['rewards'][np.where(steps['rewards'] == 4.0)] = 3.0
                print(f"Found episode {episode_path} with max reward 4, rewards corrected. New max reward: {np.max(steps['rewards'])}.")
            
            # Pad image observations to MAX_COMPRESSED_DIMENSION and record padding lengths
           # padding_lengths = {}
            for image_key in image_keys:
                # camera_key = '/'.join(image_key.split('/')[2 : ]) 
                # current_image = steps[image_key]
                # current_compressed_dim = current_image.shape[-1]
                # padding_length = MAX_COMPRESSED_DIMENSION - current_compressed_dim
                # steps[image_key] = np.concatenate([current_image, np.zeros((current_image.shape[0], padding_length), dtype = np.uint8)], axis = -1)
                # padding_lengths[f'padding_length/{camera_key}'] = np.int64(padding_length)

                compressed_images = steps[image_key]  # Shape: (episode_length, compressed_dim)
                encoded_jpegs = []
                
                for compressed_frame in compressed_images:
                    # Decode the compressed image using cv2 (flag 1 = cv2.IMREAD_COLOR)
                    decoded_image = cv2.imdecode(compressed_frame, 1)
                    
                    # Encode as JPEG using TensorFlow
                    jpeg_encoded = tf.image.encode_jpeg(decoded_image).numpy()
                    encoded_jpegs.append(jpeg_encoded)
                
                # Store as array of byte strings
                steps[image_key] = np.array(encoded_jpegs, dtype = object)
            
            # Convert dictionary of arrays to list of dictionaries
            # Get episode length from one of the arrays (assuming all have same length)
            episode_length = len(steps['rewards'])
            steps_list = []
            
            # Define scalar keys that need .item() to extract Python scalar
            scalar_keys = {'dones', 'rewards', 'truncateds', 'obses/state/left/gripper_pos', 'obses/state/right/gripper_pos'}
            
            for i in range(episode_length):
                step = {}
                for key, value in steps.items():
                    if key in scalar_keys:
                        step[key] = value[i].item()
                    else:
                        step[key] = value[i]
                # Add padding lengths to each step (constant across episode)
                # for padding_key, padding_value in padding_lengths.items():
                #     step[padding_key] = padding_value
                steps_list.append(step)
            
            # create output data sample
            sample = {
                'steps': steps_list,
                'episode_metadata': episode_metadata
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = []
        for path in paths:
            episode_paths.extend(glob.glob(path))

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

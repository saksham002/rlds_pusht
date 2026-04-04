"""slurm_rlds config: Dexterous robot shirt-hanging HDF5 dataset.

Converts HDF5 episodes at 60fps to RLDS tfrecords at 30fps (skipping every other frame).
Produces fields compatible with the RoboCOIN data loader.

Annotations are loaded from annotations/heuristic_annotations.json, which is produced
by solve_subtask_boundaries.py. The annotation file keys episodes by dataset directory
name (e.g. "real_hang_full_success_r5_hdf5") and episode number.

Usage (single worker, local):
    cd hdf5_to_tfds/
    python -m framework.runner \
        --config dexterous_hang_config.py \
        --data_dir /tmp/real_hang/0 \
        --worker_id 0 \
        --num_workers 1
"""
import io
import json
import os

import h5py
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_NAME = 'real_hang'
DATASET_VERSION = '1.0.0'

_DATA_ROOT = '/data/group_data/rl/dexterous_robot_data'
_DATASET_DIRS = [
    'real_hang_zheyuan_0325_hdf5',
    'real_hang_robyn_0324_hdf5',
    'real_hang_riya_0327_hdf5',
    'real_hang_jasmine_0331_hdf5',
    'real_hang_round1_0623_hdf5',
    'real_hang_round2_0624_hdf5',
    'real_hang_round3_0703_hdf5',
    'real_hang_round4_0705_hdf5',
    'real_hang_round5_0709_hdf5',
    'real_hang_round6_0717_hdf5',
    'real_hang_full_success_r4_hdf5',
    'real_hang_full_success_r5_hdf5',
]
_ANNOTATIONS_DIR = os.path.join(os.path.dirname(__file__), 'annotations')
_FPS = 30.0
_MAX_CAMERAS = 3
_MAX_STATE_DIM = 16
_MAX_ACTION_DIM = 14
_MAX_SUBTASKS = 5
_VAL_FRACTION = 0.05
_DIR_TO_REPO_INDEX = {ds: i for i, ds in enumerate(_DATASET_DIRS)}


def _load_annotations():
    path = os.path.join(_ANNOTATIONS_DIR, 'heuristic_annotations.json')
    with open(path, 'r') as f:
        return json.load(f)

_ANNOTATIONS = _load_annotations()
_SUBTASK_NAMES = [d['subtask'] for d in _ANNOTATIONS['subtask_definitions']]


def _build_episode_list():
    """Enumerate episodes with valid annotations and split into train/val."""
    all_eps = []
    for ds in _DATASET_DIRS:
        ds_data = _ANNOTATIONS['datasets'].get(ds, {})
        for ep_str, info in ds_data.items():
            if info['boundaries'] is not None:
                all_eps.append((ds, int(ep_str)))
    all_eps.sort()

    rng = np.random.RandomState(86)
    num_val = max(1, int(len(all_eps) * _VAL_FRACTION))
    indices = rng.permutation(len(all_eps))
    val_set = set(indices[:num_val].tolist())

    train = [all_eps[i] for i in range(len(all_eps)) if i not in val_set]
    val = [all_eps[i] for i in range(len(all_eps)) if i in val_set]
    return train, val

_TRAIN_EPISODES, _VAL_EPISODES = _build_episode_list()


def _get_subtask_segments(dataset_key, ep_num, ep_len):
    """Build subtask segment list from heuristic boundaries.

    boundaries[i] is the first step of subtask i+1, so subtask i spans
    [boundaries[i-1], boundaries[i] - 1], with subtask 0 starting at 0 and
    subtask 5 ending at ep_len - 1.
    """
    ep_data = _ANNOTATIONS['datasets'][dataset_key][str(ep_num)]
    bounds = ep_data['boundaries']

    starts = [0] + bounds
    ends = [b - 1 for b in bounds] + [ep_len - 1]

    num_subtasks = len(bounds) + 1
    segments = []
    for i in range(num_subtasks):
        segments.append({
            'subtask': _SUBTASK_NAMES[i],
            'start_step': starts[i],
            'end_step': ends[i],
        })
    return segments


def _get_subtask_info_for_step(step_idx, segments):
    """Return the active subtask in slot 0 for a given step."""
    subtask_names = ['null'] * _MAX_SUBTASKS
    subtask_mask = np.zeros(_MAX_SUBTASKS, dtype = np.bool_)
    steps_to_end = np.zeros(_MAX_SUBTASKS, dtype = np.int32)
    subtask_len_arr = np.zeros(_MAX_SUBTASKS, dtype = np.int32)
    is_first = np.zeros(_MAX_SUBTASKS, dtype = np.bool_)
    is_last = np.zeros(_MAX_SUBTASKS, dtype = np.bool_)
    first_null_index = 0

    for seg in segments:
        start, end = seg['start_step'], seg['end_step']
        if not (start <= step_idx <= end):
            continue

        subtask_names[0] = seg['subtask']
        subtask_mask[0] = True
        steps_to_end[0] = end - step_idx
        subtask_len_arr[0] = end - start + 1
        is_first[0] = (step_idx == start)
        is_last[0] = (step_idx == end)
        first_null_index = 1
        break

    return {
        'subtask_names': subtask_names,
        'subtask_mask': subtask_mask,
        'steps_to_subtask_end': steps_to_end,
        'subtask_len': subtask_len_arr,
        'subtask_is_first': is_first,
        'subtask_is_last': is_last,
        'first_null_index': np.int32(first_null_index),
    }


def get_features():
    obs_features = {}
    for i in range(_MAX_CAMERAS):
        obs_features[f'observation/image/cam_{i}'] = tfds.features.Tensor(shape = (), dtype = tf.string)
    obs_features['observation/state'] = tfds.features.Tensor(shape = (_MAX_STATE_DIM,), dtype = np.float32)

    return tfds.features.FeaturesDict({
        'steps': tfds.features.Dataset({
            **obs_features,
            'action': tfds.features.Tensor(shape = (_MAX_ACTION_DIM,), dtype = np.float32),
            'state_diff': tfds.features.Tensor(shape = (_MAX_STATE_DIM,), dtype = np.float32),
            'action_diff': tfds.features.Tensor(shape = (_MAX_ACTION_DIM,), dtype = np.float32),
            'is_first': tfds.features.Scalar(dtype = np.bool_),
            'is_terminal': tfds.features.Scalar(dtype = np.bool_),
            'frame_index': tfds.features.Scalar(dtype = np.int64),
            'task': tfds.features.Text(),
            'episode_index': tfds.features.Scalar(dtype = np.int64),
            'index': tfds.features.Scalar(dtype = np.int64),
            'subtask_1': tfds.features.Text(),
            'subtask_2': tfds.features.Text(),
            'subtask_3': tfds.features.Text(),
            'subtask_4': tfds.features.Text(),
            'subtask_5': tfds.features.Text(),
            'subtask_mask': tfds.features.Tensor(shape = (5,), dtype = np.bool_),
            'steps_to_subtask_end': tfds.features.Tensor(shape = (5,), dtype = np.int32),
            'subtask_len': tfds.features.Tensor(shape = (5,), dtype = np.int32),
            'subtask_is_first': tfds.features.Tensor(shape = (5,), dtype = np.bool_),
            'subtask_is_last': tfds.features.Tensor(shape = (5,), dtype = np.bool_),
            'first_null_index': tfds.features.Scalar(dtype = np.int32),
            'scene_annotation': tfds.features.Scalar(dtype = np.int32),
            'eef_sim_pose_state': tfds.features.Tensor(shape = (12,), dtype = np.float32),
            'eef_sim_pose_action': tfds.features.Tensor(shape = (12,), dtype = np.float32),
            'eef_sim_pose_action_diff': tfds.features.Tensor(shape = (12,), dtype = np.float32),
            'repo_index': tfds.features.Scalar(dtype = np.int32),
        }),
        'episode_metadata': tfds.features.FeaturesDict({
            'repo_id': tfds.features.Text(),
            'robot_type': tfds.features.Text(),
            'fps': tfds.features.Scalar(dtype = np.float32),
            'camera_names': tfds.features.Sequence(tfds.features.Text()),
            'camera_shapes': tfds.features.Sequence(tfds.features.Tensor(shape = (3,), dtype = np.int32)),
            'num_cameras': tfds.features.Scalar(dtype = np.int64),
            'state_feature_names': tfds.features.Sequence(tfds.features.Text()),
            'action_feature_names': tfds.features.Sequence(tfds.features.Text()),
            'subtasks': tfds.features.Sequence(tfds.features.Text()),
            'task_description': tfds.features.Text(),
        }),
    })


def get_episodes(split):
    if split == 'train':
        ep_list = _TRAIN_EPISODES
    elif split == 'val':
        ep_list = _VAL_EPISODES
    else:
        return []
    return sorted([
        os.path.join(_DATA_ROOT, ds, f'episode_{ep}.hdf5')
        for ds, ep in ep_list
    ])


def _build_state(left_joints, left_gripper, right_joints, right_gripper):
    """16-D state: [left_joint_qpos(7), left_gripper(1), right_joint_qpos(7), right_gripper(1)]."""
    return np.concatenate([
        left_joints,
        [left_gripper],
        right_joints,
        [right_gripper],
    ]).astype(np.float32)


def _quat_to_euler(quat):
    """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)."""
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype = np.float32)


def _build_eef(left_tcp_pose, right_tcp_pose):
    """12-D EEF: left pos(3) + euler(3) + right pos(3) + euler(3)."""
    left_euler = _quat_to_euler(left_tcp_pose[3:7])
    right_euler = _quat_to_euler(right_tcp_pose[3:7])
    return np.concatenate([
        left_tcp_pose[:3], left_euler,
        right_tcp_pose[:3], right_euler,
    ]).astype(np.float32)


def _decode_and_reencode_jpeg(raw_uint8):
    """Decode JPEG from numpy uint8 array, resize to 224x224, re-encode."""
    img = Image.open(io.BytesIO(raw_uint8.tobytes()))
    img_array = np.asarray(img, dtype = np.float32) / 255.0
    img_resized = tf.image.resize(img_array, (224, 224)).numpy()
    img_uint8 = np.rint(img_resized * 255.0).astype(np.uint8)
    return tf.image.encode_jpeg(img_uint8).numpy()


def parse_episode(episode_path):
    dataset_key = os.path.basename(os.path.dirname(episode_path))
    repo_index = _DIR_TO_REPO_INDEX[dataset_key]
    ep_idx = int(os.path.basename(episode_path).replace('episode_', '').replace('.hdf5', ''))

    with h5py.File(episode_path, 'r') as f:
        task = f['metadata/task'][()].decode('utf-8') if isinstance(f['metadata/task'][()], bytes) else str(f['metadata/task'][()])
        total_frames = f['obses/state/left/gripper_pos'].shape[0]

        frame_indices = list(range(0, total_frames, 2))
        ep_len = len(frame_indices)

        segments = _get_subtask_segments(dataset_key, ep_idx, ep_len)

        left_joints = f['obses/state/left/joint_qpos'][:]
        left_gripper = f['obses/state/left/gripper_pos'][:, 0]
        right_joints = f['obses/state/right/joint_qpos'][:]
        right_gripper = f['obses/state/right/gripper_pos'][:, 0]
        left_tcp = f['obses/state/left/tcp_pose'][:]
        right_tcp = f['obses/state/right/tcp_pose'][:]
        images_right_top = f['obses/images/right/top']
        images_left_wrist = f['obses/images/left/wrist']
        images_right_wrist = f['obses/images/right/wrist']

        first_top_img = Image.open(io.BytesIO(images_right_top[0].tobytes()))
        first_left_wrist_img = Image.open(io.BytesIO(images_left_wrist[0].tobytes()))
        first_right_wrist_img = Image.open(io.BytesIO(images_right_wrist[0].tobytes()))
        camera_shapes = [
            np.array([first_top_img.size[1], first_top_img.size[0], 3], dtype = np.int32),
            np.array([first_left_wrist_img.size[1], first_left_wrist_img.size[0], 3], dtype = np.int32),
            np.array([first_right_wrist_img.size[1], first_right_wrist_img.size[0], 3], dtype = np.int32),
        ]

        steps = []
        prev_state = None
        prev_action = None
        prev_eef_action = None

        for i, fi in enumerate(frame_indices):
            state = _build_state(left_joints[fi], left_gripper[fi], right_joints[fi], right_gripper[fi])
            action = np.zeros(_MAX_ACTION_DIM, dtype = np.float32)
            action[6] = left_gripper[fi]
            action[13] = right_gripper[fi]
            eef = _build_eef(left_tcp[fi], right_tcp[fi])

            img_bytes = _decode_and_reencode_jpeg(images_right_top[fi])
            img_left_wrist = _decode_and_reencode_jpeg(images_left_wrist[fi])
            img_right_wrist = _decode_and_reencode_jpeg(images_right_wrist[fi])

            state_diff = np.zeros(_MAX_STATE_DIM, dtype = np.float32)
            action_diff = np.zeros(_MAX_ACTION_DIM, dtype = np.float32)
            eef_action_diff = np.zeros(12, dtype = np.float32)
            if prev_state is not None:
                steps[-1]['state_diff'] = state - prev_state
            if prev_action is not None:
                steps[-1]['action_diff'] = action - prev_action
            if prev_eef_action is not None:
                steps[-1]['eef_sim_pose_action_diff'] = eef - prev_eef_action

            sub = _get_subtask_info_for_step(i, segments)

            step = {
                'observation/image/cam_0': img_bytes,
                'observation/image/cam_1': img_left_wrist,
                'observation/image/cam_2': img_right_wrist,
                'observation/state': state,
                'action': action,
                'state_diff': state_diff,
                'action_diff': action_diff,
                'is_first': (i == 0),
                'is_terminal': (i == ep_len - 1),
                'frame_index': np.int64(i),
                'task': task,
                'episode_index': np.int64(ep_idx),
                'index': np.int64(i),
                'subtask_1': sub['subtask_names'][0],
                'subtask_2': sub['subtask_names'][1],
                'subtask_3': sub['subtask_names'][2],
                'subtask_4': sub['subtask_names'][3],
                'subtask_5': sub['subtask_names'][4],
                'subtask_mask': sub['subtask_mask'],
                'steps_to_subtask_end': sub['steps_to_subtask_end'],
                'subtask_len': sub['subtask_len'],
                'subtask_is_first': sub['subtask_is_first'],
                'subtask_is_last': sub['subtask_is_last'],
                'first_null_index': sub['first_null_index'],
                'scene_annotation': np.int32(0),
                'eef_sim_pose_state': eef,
                'eef_sim_pose_action': eef,
                'eef_sim_pose_action_diff': eef_action_diff,
                'repo_index': np.int32(repo_index),
            }

            prev_state = state.copy()
            prev_action = action.copy()
            prev_eef_action = eef.copy()
            steps.append(step)

    sample = {
        'steps': steps,
        'episode_metadata': {
            'repo_id': 'dexterous_hang',
            'robot_type': 'xarm',
            'fps': np.float32(_FPS),
            'camera_names': ['right_top', 'left_wrist', 'right_wrist'],
            'camera_shapes': camera_shapes,
            'num_cameras': np.int64(len(camera_shapes)),
            'state_feature_names': [
                'left_joint_0', 'left_joint_1', 'left_joint_2', 'left_joint_3',
                'left_joint_4', 'left_joint_5', 'left_joint_6', 'left_gripper_open',
                'right_joint_0', 'right_joint_1', 'right_joint_2', 'right_joint_3',
                'right_joint_4', 'right_joint_5', 'right_joint_6', 'right_gripper_open',
            ],
            'action_feature_names': [
                'dummy_1', 'dummy_2', 'dummy_3',
                'dummy_4', 'dummy_5', 'dummy_6', 'left_gripper_open',
                'dummy_7', 'dummy_8', 'dummy_9',
                'dummy_10', 'dummy_11', 'dummy_12', 'right_gripper_open',
            ],
            'subtasks': _SUBTASK_NAMES + ['null'],
            'task_description': task,
        },
    }
    return episode_path, sample

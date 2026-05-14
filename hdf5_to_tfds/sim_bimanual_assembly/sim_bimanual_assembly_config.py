"""slurm_rlds config: Sim bimanual block-insert dataset.

Converts sim_double_insert HDF5 episodes to RLDS tfrecords at the native
60 fps.

Per-frame action is the next-step state derived by composing the current
state with actions/global_action[t]:
    arm xyz : state_tcp_xyz + global_xyz
    arm rpy : state_R composed with R.from_euler('xyz', global_rpy),
              re-extracted as extrinsic xyz Euler
    gripper : 1 - (state_gripper + 0.1 * global_gripper)
relative_action is copied straight from actions/relative_action with the
gripper slots scaled by -0.1.

Annotations come from
  /data/user_data/saksham3/sim_bimanual_assembly/annotations/subtask_annotations_all.json
 (episode keys are dataset directory name + episode number). Three subtasks total, so
boundaries is one of: [b1] (only subtask 1->2 visible),
or [b1, b2] (all three subtasks visible).

Validation split: 5% of episodes PER FOLDER, seeded with RandomState(86).

Per-step reward is 0 everywhere except the terminal step, which carries 1.0
iff max(rewards/rewards) >= 3.0 in the HDF5 episode (otherwise 0.0).

Usage (single worker, local):
    cd hdf5_to_tfds/
    python -m framework.runner \
        --config sim_bimanual_assembly/sim_bimanual_assembly_config.py \
        --data_dir /tmp/sim_bimanual_assembly/0 \
        --worker_id 0 \
        --num_workers 1
"""
import io
import json
import os

import cv2
import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_NAME = 'sim_bimanual_assembly'
DATASET_VERSION = '1.0.0'

_TASK_PROMPT = (
    'Insert the white block into the pink and blue blocks and place the '
    'combination on the platform.'
)

_DATA_ROOT = '/data/group_data/rl/dexterous_robot_data'
_DATASET_DIRS = [
    'sim_double_insert_0226_hdf5',
    'sim_double_insert_round1_0520_v2_hdf5',
    'sim_double_insert_round2_0522_hdf5',
    'sim_double_insert_round3_0605_hdf5',
    'sim_double_insert_round4_0606_hdf5',
    'sim_double_insert_full_success_r4_hdf5',
    'sim_double_insert_round5_0812_hdf5',
    'sim_double_insert_full_success_r5_hdf5',
    'sim_double_insert_round6_0820_hdf5',
]
_ANNOTATIONS_PATH = (
    '/data/user_data/saksham3/sim_bimanual_assembly/annotations/'
    'subtask_annotations_all.json'
)
_FPS = 60.0
_MAX_CAMERAS = 3
_MAX_ACTION_DIM = 14
_MAX_STATE_DIM = 16
_MAX_SUBTASKS = 5
_VAL_FRACTION = 0.05
_TERMINAL_REWARD_THRESHOLD = 3.0
_DIR_TO_REPO_INDEX = {ds: i for i, ds in enumerate(_DATASET_DIRS)}

_ACTION_FEATURE_NAMES = [
    'left_x', 'left_y', 'left_z',
    'left_roll', 'left_pitch', 'left_yaw', 'left_gripper_open',
    'right_x', 'right_y', 'right_z',
    'right_roll', 'right_pitch', 'right_yaw', 'right_gripper_open',
]
_STATE_FEATURE_NAMES = [
    'left_joint_0', 'left_joint_1', 'left_joint_2', 'left_joint_3',
    'left_joint_4', 'left_joint_5', 'left_joint_6', 'left_gripper_open',
    'right_joint_0', 'right_joint_1', 'right_joint_2', 'right_joint_3',
    'right_joint_4', 'right_joint_5', 'right_joint_6', 'right_gripper_open',
]


def _load_annotations():
    with open(_ANNOTATIONS_PATH, 'r') as f:
        return json.load(f)

_ANNOTATIONS = _load_annotations()
_SUBTASK_NAMES = [d['subtask'] for d in _ANNOTATIONS['subtask_definitions']]


def _build_episode_list():
    """Per-folder 5% validation split (deterministic via seed 86)."""
    rng = np.random.RandomState(86)
    train = []
    val = []
    for ds in _DATASET_DIRS:
        ds_data = _ANNOTATIONS['datasets'].get(ds, {})
        ep_nums = sorted(int(k) for k in ds_data.keys())
        if not ep_nums:
            continue
        indices = rng.permutation(len(ep_nums))
        num_val = max(1, int(len(ep_nums) * _VAL_FRACTION))
        val_set = set(indices[:num_val].tolist())
        for i, ep in enumerate(ep_nums):
            if i in val_set:
                val.append((ds, ep))
            else:
                train.append((ds, ep))
    train.sort()
    val.sort()
    return train, val

_TRAIN_EPISODES, _VAL_EPISODES = _build_episode_list()


def _get_subtask_segments(dataset_key, ep_num, ep_len):
    """Build subtask segment list from heuristic boundaries.

    boundaries[i] is the first step of subtask i+1, so subtask i spans
    [boundaries[i-1], boundaries[i] - 1], with subtask 0 starting at 0 and
    the last present subtask ending at ep_len - 1.
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


def _get_subtask_info_for_step(step_idx, segments, ep_len):
    """Return the active subtask in slot 0 for a given step.

    Defaults (used when no segment matches, e.g. unannotated episodes):
    'dummy' for every subtask name, first_null_index = _MAX_SUBTASKS, and
    steps_to_subtask_end[0] = ep_len - 1 - step_idx so it counts down to the
    final step. A matching segment overrides slot 0 with the real values.
    """
    subtask_names = ['dummy'] * _MAX_SUBTASKS
    subtask_mask = np.zeros(_MAX_SUBTASKS, dtype = np.bool_)
    steps_to_end = np.zeros(_MAX_SUBTASKS, dtype = np.int32)
    steps_to_end[0] = np.int32(ep_len - 1 - step_idx)
    subtask_len_arr = np.zeros(_MAX_SUBTASKS, dtype = np.int32)
    is_first = np.zeros(_MAX_SUBTASKS, dtype = np.bool_)
    is_last = np.zeros(_MAX_SUBTASKS, dtype = np.bool_)
    first_null_index = _MAX_SUBTASKS

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


def _build_is_intervention(ep_len, intervention_indices):
    if intervention_indices is None:
        return np.ones(ep_len, dtype = np.bool_)
    return np.isin(np.arange(ep_len), intervention_indices).astype(np.bool_)


def _build_state(left_joints, left_gripper, right_joints, right_gripper):
    """16-D: [left_joint_qpos(7), 1 - left_gripper, right_joint_qpos(7), 1 - right_gripper]."""
    return np.concatenate([
        left_joints,
        [1.0 - left_gripper],
        right_joints,
        [1.0 - right_gripper],
    ]).astype(np.float32)


def _quat_to_euler(quat):
    """quaternion (w, x, y, z) -> (roll, pitch, yaw) extrinsic XYZ Euler."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
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


def _build_eef_state(left_tcp_pose, right_tcp_pose):
    """12-D absolute EEF state: left xyz + rpy, right xyz + rpy."""
    left_rpy = _quat_to_euler(left_tcp_pose[3:7])
    right_rpy = _quat_to_euler(right_tcp_pose[3:7])
    return np.concatenate([
        left_tcp_pose[:3], left_rpy,
        right_tcp_pose[:3], right_rpy,
    ]).astype(np.float32)


def _compose_arm_action(tcp_pose, global_arm_action):
    """Compose tcp_pose (xyz + quat wxyz) with a 6-D global action (xyz + rpy).

    Returns a 6-D float32 [xyz(3), rpy(3)] arm action where:
        xyz = state_xyz + global_xyz
        rpy = (R(state_quat) * R.from_euler('xyz', global_rpy)).as_euler('xyz')
    """
    state_R = R.from_quat(tcp_pose[3:7], scalar_first = True)
    delta_R = R.from_euler('xyz', global_arm_action[3:6])
    composed_rpy = (state_R * delta_R).as_euler('xyz').astype(np.float32)
    composed_xyz = (tcp_pose[:3] + global_arm_action[:3]).astype(np.float32)
    return np.concatenate([composed_xyz, composed_rpy]).astype(np.float32)


def _compose_gripper_action(gripper_state, gripper_global_action):
    """1 - (state + 0.1 * global)."""
    return np.float32(1.0 - (gripper_state + 0.1 * gripper_global_action))


def _decode_and_reencode_jpeg(raw_uint8):
    """Decode JPEG via cv2 (HDF5 JPEGs are stored BGR; matches the decoding in
    solve_subtask_boundaries.py / log_all_videos.py — no channel swap), then
    resize to 480x480 and re-encode with TensorFlow.
    """
    img = cv2.imdecode(np.asarray(raw_uint8), 1)
    img_array = np.asarray(img, dtype = np.float32) / 255.0
    img_resized = tf.image.resize(img_array, (480, 480)).numpy()
    img_uint8 = np.rint(img_resized * 255.0).astype(np.uint8)
    return tf.image.encode_jpeg(img_uint8).numpy()


def get_features():
    obs_features = {}
    for i in range(_MAX_CAMERAS):
        obs_features[f'observation/image/cam_{i}'] = tfds.features.Tensor(shape = (), dtype = tf.string)
    obs_features['observation/state'] = tfds.features.Tensor(shape = (_MAX_STATE_DIM,), dtype = np.float32)

    return tfds.features.FeaturesDict({
        'steps': tfds.features.Dataset({
            **obs_features,
            'action': tfds.features.Tensor(shape = (_MAX_ACTION_DIM,), dtype = np.float32),
            'relative_action': tfds.features.Tensor(shape = (_MAX_ACTION_DIM,), dtype = np.float32),
            'is_first': tfds.features.Scalar(dtype = np.bool_),
            'is_terminal': tfds.features.Scalar(dtype = np.bool_),
            'is_intervention': tfds.features.Scalar(dtype = np.bool_),
            'reward': tfds.features.Scalar(dtype = np.float32),
            'frame_index': tfds.features.Scalar(dtype = np.int64),
            'task': tfds.features.Text(),
            'episode_index': tfds.features.Scalar(dtype = np.int64),
            'index': tfds.features.Scalar(dtype = np.int64),
            'subtask_1': tfds.features.Text(),
            'subtask_2': tfds.features.Text(),
            'subtask_3': tfds.features.Text(),
            'subtask_4': tfds.features.Text(),
            'subtask_5': tfds.features.Text(),
            'subtask_mask': tfds.features.Tensor(shape = (_MAX_SUBTASKS,), dtype = np.bool_),
            'steps_to_subtask_end': tfds.features.Tensor(shape = (_MAX_SUBTASKS,), dtype = np.int32),
            'subtask_len': tfds.features.Tensor(shape = (_MAX_SUBTASKS,), dtype = np.int32),
            'subtask_is_first': tfds.features.Tensor(shape = (_MAX_SUBTASKS,), dtype = np.bool_),
            'subtask_is_last': tfds.features.Tensor(shape = (_MAX_SUBTASKS,), dtype = np.bool_),
            'first_null_index': tfds.features.Scalar(dtype = np.int32),
            'scene_annotation': tfds.features.Scalar(dtype = np.int32),
            'eef_sim_pose_state': tfds.features.Tensor(shape = (12,), dtype = np.float32),
            'eef_sim_pose_action': tfds.features.Tensor(shape = (12,), dtype = np.float32),
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
            'has_subtask_annotations': tfds.features.Scalar(dtype = np.bool_),
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


def parse_episode(episode_path):
    dataset_key = os.path.basename(os.path.dirname(episode_path))
    repo_index = _DIR_TO_REPO_INDEX[dataset_key]
    ep_idx = int(os.path.basename(episode_path).replace('episode_', '').replace('.hdf5', ''))

    with h5py.File(episode_path, 'r') as f:
        task = _TASK_PROMPT
        ep_len = f['obses/state/left/gripper_pos'].shape[0]

        ep_annotation = _ANNOTATIONS['datasets'][dataset_key][str(ep_idx)]
        if ep_annotation['boundaries'] is None:
            raise ValueError(
                f'No subtask boundary annotation for {dataset_key}/'
                f'episode_{ep_idx}; dummy subtasks are not supported.'
            )
        segments = _get_subtask_segments(dataset_key, ep_idx, ep_len)

        left_joints = f['obses/state/left/joint_qpos'][:]
        left_gripper = f['obses/state/left/gripper_pos'][:, 0]
        right_joints = f['obses/state/right/joint_qpos'][:]
        right_gripper = f['obses/state/right/gripper_pos'][:, 0]
        left_tcp = f['obses/state/left/tcp_pose'][:]
        right_tcp = f['obses/state/right/tcp_pose'][:]
        global_action = f['actions/global_action'][:]
        relative_action = f['actions/relative_action'][:]
        rewards = f['rewards/rewards'][:]
        intervention_indices = None
        if 'interventions' in f['metadata']:
            intervention_indices = f['metadata/interventions'][:]
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
        is_intervention = _build_is_intervention(ep_len, intervention_indices)
        terminal_reward = np.float32(
            1.0 if float(rewards.max()) >= _TERMINAL_REWARD_THRESHOLD else 0.0
        )

        steps = []
        for i in range(ep_len):
            state = _build_state(
                left_joints[i], left_gripper[i], right_joints[i], right_gripper[i],
            )

            left_arm = _compose_arm_action(left_tcp[i], global_action[i, 0:6])
            right_arm = _compose_arm_action(right_tcp[i], global_action[i, 7:13])
            left_grip_action = _compose_gripper_action(left_gripper[i], global_action[i, 6])
            right_grip_action = _compose_gripper_action(right_gripper[i], global_action[i, 13])
            action = np.concatenate([
                left_arm, [left_grip_action],
                right_arm, [right_grip_action],
            ]).astype(np.float32)

            eef_state = _build_eef_state(left_tcp[i], right_tcp[i])
            eef_action = np.concatenate([left_arm, right_arm]).astype(np.float32)

            rel_act = relative_action[i].astype(np.float32).copy()
            rel_act[6] *= -0.1
            rel_act[13] *= -0.1

            img_right_top = _decode_and_reencode_jpeg(images_right_top[i])
            img_left_wrist = _decode_and_reencode_jpeg(images_left_wrist[i])
            img_right_wrist = _decode_and_reencode_jpeg(images_right_wrist[i])

            sub = _get_subtask_info_for_step(i, segments, ep_len)

            reward = terminal_reward if i == ep_len - 1 else np.float32(0.0)

            step = {
                'observation/image/cam_0': img_right_top,
                'observation/image/cam_1': img_left_wrist,
                'observation/image/cam_2': img_right_wrist,
                'observation/state': state,
                'action': action,
                'relative_action': rel_act,
                'is_first': (i == 0),
                'is_terminal': (i == ep_len - 1),
                'is_intervention': is_intervention[i],
                'reward': reward,
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
                'eef_sim_pose_state': eef_state,
                'eef_sim_pose_action': eef_action,
                'repo_index': np.int32(repo_index),
            }
            steps.append(step)

    sample = {
        'steps': steps,
        'episode_metadata': {
            'repo_id': 'sim_bimanual_assembly',
            'robot_type': 'sim_xarm',
            'fps': np.float32(_FPS),
            'camera_names': ['right_top', 'left_wrist', 'right_wrist'],
            'camera_shapes': camera_shapes,
            'num_cameras': np.int64(len(camera_shapes)),
            'state_feature_names': _STATE_FEATURE_NAMES,
            'action_feature_names': _ACTION_FEATURE_NAMES,
            'subtasks': _SUBTASK_NAMES + ['dummy'],
            'task_description': task,
            'has_subtask_annotations': np.bool_(True),
        },
    }
    return episode_path, sample

"""slurm_rlds config: Dexterous robot shirt-hanging HDF5 dataset.

By default, converts HDF5 episodes from 60fps to RLDS tfrecords at 30fps (skipping
every other frame). Set REAL_HANG_SUBSAMPLE=0 to disable subsampling and emit every
raw frame at 60fps; in that mode actions and eef_sim_pose_action are read at the
current frame (no look-ahead).

B1 (state shift): in the raw HDF5, observation index k stores the state AFTER
action k. To pair (obs_t, action_t) as 'action taken FROM obs_t', every
per-step observation array (images, tcp_pose, joint_qpos, gripper_pos) is
shifted back by one — new[t] = old[t-1] for t >= 1, with new[0] = old[0].
target_tcp_pose and relative_action are action arrays and keep their original
indices.

Produces fields compatible with the RoboCOIN data loader.

Annotations are loaded from annotations/heuristic_annotations.json at 30fps, or
annotations/heuristic_annotations_60hz.json when REAL_HANG_SUBSAMPLE=0. The
annotation file is produced by solve_subtask_boundaries.py and keys episodes by
dataset directory name (e.g. "real_hang_full_success_r5_hdf5") and episode number.

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
from scipy.spatial.transform import Rotation as R
import tensorflow as tf
import tensorflow_datasets as tfds


# Toggle 60->30 Hz subsampling. When disabled, every raw frame is emitted and
# actions / eef_sim_pose_action are read at the current frame instead of fi+1.
_SUBSAMPLE = os.environ.get('REAL_HANG_SUBSAMPLE', '1') == '1'

# The 'relative_action' step field is logged directly from the HDF5 at the
# current frame (next_fi == fi when subsampling is off). If subsampling is
# re-enabled, relative_action's indexing must be revisited before this assert
# is removed.
assert not _SUBSAMPLE, (
    'REAL_HANG_SUBSAMPLE must be 0: relative_action handling needs to be '
    'updated if subsampling is on.'
)

DATASET_NAME = 'real_shirt_hang'
DATASET_VERSION = '1.0.0'

_TASK_PROMPT = 'Place the shirt on the hanger and hang it from the rod.'

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
_ANNOTATIONS_FILENAME = os.environ.get(
    'REAL_HANG_ANNOTATIONS_FILE',
    'subtask_annotations_all.json',
)
_FPS = 30.0 if _SUBSAMPLE else 60.0
_MAX_CAMERAS = 3
_MAX_ACTION_DIM = 14
_MAX_SUBTASKS = 5
_VAL_FRACTION = 0.05
_DIR_TO_REPO_INDEX = {ds: i for i, ds in enumerate(_DATASET_DIRS)}

# Env overrides: set STATE_DIM to 14 or 16, SUBSET_DIRS to a comma-separated
# list of dataset subdirs to restrict the episode pool, and TRAIN_COUNT /
# VAL_COUNT to override the default val-fraction split with fixed counts.
_STATE_DIM = int(os.environ.get('REAL_HANG_STATE_DIM', '16'))
assert _STATE_DIM in (14, 16), f'REAL_HANG_STATE_DIM must be 14 or 16, got {_STATE_DIM}'
_MAX_STATE_DIM = _STATE_DIM

_SUBSET_DIRS_ENV = os.environ.get('REAL_HANG_SUBSET_DIRS', '').strip()
_SUBSET_DIRS = set(_SUBSET_DIRS_ENV.split(',')) if _SUBSET_DIRS_ENV else None

_TRAIN_COUNT_ENV = os.environ.get('REAL_HANG_TRAIN_COUNT', '').strip()
_VAL_COUNT_ENV = os.environ.get('REAL_HANG_VAL_COUNT', '').strip()
_TRAIN_COUNT = int(_TRAIN_COUNT_ENV) if _TRAIN_COUNT_ENV else None
_VAL_COUNT = int(_VAL_COUNT_ENV) if _VAL_COUNT_ENV else None

_ACTION_FEATURE_NAMES = [
    'dummy_1', 'dummy_2', 'dummy_3',
    'dummy_4', 'dummy_5', 'dummy_6', 'left_gripper_open',
    'dummy_7', 'dummy_8', 'dummy_9',
    'dummy_10', 'dummy_11', 'dummy_12', 'right_gripper_open',
]
if _STATE_DIM == 16:
    _STATE_FEATURE_NAMES = [
        'left_joint_0', 'left_joint_1', 'left_joint_2', 'left_joint_3',
        'left_joint_4', 'left_joint_5', 'left_joint_6', 'left_gripper_open',
        'right_joint_0', 'right_joint_1', 'right_joint_2', 'right_joint_3',
        'right_joint_4', 'right_joint_5', 'right_joint_6', 'right_gripper_open',
    ]
else:
    _STATE_FEATURE_NAMES = list(_ACTION_FEATURE_NAMES)


def _load_annotations():
    path = os.path.join(_ANNOTATIONS_DIR, _ANNOTATIONS_FILENAME)
    with open(path, 'r') as f:
        return json.load(f)

_ANNOTATIONS = _load_annotations()
_SUBTASK_NAMES = [d['subtask'] for d in _ANNOTATIONS['subtask_definitions']]


def _build_episode_list():
    """Enumerate every episode on disk and split into train/val.

    All hdf5 episodes under each dataset dir are included regardless of
    whether the annotation file has subtask boundaries for them; the per-step
    subtask fields are filled with dummy values and 'has_subtask_annotations'
    in episode_metadata records whether boundaries were available.
    """
    all_eps = []
    for ds in _DATASET_DIRS:
        if _SUBSET_DIRS is not None and ds not in _SUBSET_DIRS:
            continue
        ds_dir = os.path.join(_DATA_ROOT, ds)
        for fn in os.listdir(ds_dir):
            if fn.startswith('episode_') and fn.endswith('.hdf5'):
                all_eps.append((ds, int(fn[len('episode_'):-len('.hdf5')])))
    all_eps.sort()

    rng = np.random.RandomState(86)
    indices = rng.permutation(len(all_eps))
    if _TRAIN_COUNT is not None and _VAL_COUNT is not None:
        assert _TRAIN_COUNT + _VAL_COUNT <= len(all_eps), (
            f'Requested {_TRAIN_COUNT} train + {_VAL_COUNT} val = {_TRAIN_COUNT + _VAL_COUNT} '
            f'episodes but only {len(all_eps)} available after filtering.'
        )
        val_set = set(indices[:_VAL_COUNT].tolist())
        train_set = set(indices[_VAL_COUNT : _VAL_COUNT + _TRAIN_COUNT].tolist())
        train = [all_eps[i] for i in range(len(all_eps)) if i in train_set]
        val = [all_eps[i] for i in range(len(all_eps)) if i in val_set]
    else:
        num_val = max(1, int(len(all_eps) * _VAL_FRACTION))
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


def _build_is_intervention(frame_indices, intervention_indices):
    """Build (ep_len,) intervention mask for the kept frame indices."""
    if intervention_indices is None:
        return np.ones(len(frame_indices), dtype = np.bool_)
    return np.isin(np.asarray(frame_indices), intervention_indices).astype(np.bool_)


def _build_gripper_action(curr_gripper, rel_action):
    """Convert relative gripper action to clipped absolute gripper command."""
    return np.float32(np.clip(curr_gripper - 80.0 * rel_action, 80.0, 840.0))


def _shift_state_back(arr):
    """B1 shift: new[0] = arr[0], new[t] = arr[t-1] for t >= 1. Preserves length."""
    return np.concatenate([arr[:1], arr[:-1]])


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


def _build_state(left_joints, left_gripper, right_joints, right_gripper):
    """State vector, 16-D or 14-D depending on _STATE_DIM.

    16-D: [left_joint_qpos(7), left_gripper(1), right_joint_qpos(7), right_gripper(1)]
    14-D: [zeros(6), left_gripper(1), zeros(6), right_gripper(1)] (matches action layout)
    """
    if _STATE_DIM == 16:
        return np.concatenate([
            left_joints,
            [left_gripper],
            right_joints,
            [right_gripper],
        ]).astype(np.float32)
    state = np.zeros(14, dtype = np.float32)
    state[6] = left_gripper
    state[13] = right_gripper
    return state


def _quat_to_euler(quat):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
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


def _build_eef(left_tcp_pose, right_tcp_pose):
    """12-D absolute EEF: left pos(3) + euler(3) + right pos(3) + euler(3)."""
    left_euler = _quat_to_euler(left_tcp_pose[3:7])
    right_euler = _quat_to_euler(right_tcp_pose[3:7])
    return np.concatenate([
        left_tcp_pose[:3], left_euler,
        right_tcp_pose[:3], right_euler,
    ]).astype(np.float32)

def _decode_and_reencode_jpeg(raw_uint8):
    """Decode JPEG from numpy uint8 array, resize to 480x480, re-encode."""
    img = Image.open(io.BytesIO(raw_uint8.tobytes()))
    img_array = np.asarray(img, dtype = np.float32) / 255.0
    img_resized = tf.image.resize(img_array, (480, 480)).numpy()
    img_uint8 = np.rint(img_resized * 255.0).astype(np.uint8)
    return tf.image.encode_jpeg(img_uint8).numpy()


def parse_episode(episode_path):
    dataset_key = os.path.basename(os.path.dirname(episode_path))
    repo_index = _DIR_TO_REPO_INDEX[dataset_key]
    ep_idx = int(os.path.basename(episode_path).replace('episode_', '').replace('.hdf5', ''))

    with h5py.File(episode_path, 'r') as f:
        task = _TASK_PROMPT
        total_frames = f['obses/state/left/gripper_pos'].shape[0]

        if _SUBSAMPLE:
            frame_indices = list(range(0, total_frames, 2))
            next_frame_indices = np.minimum(np.asarray(frame_indices) + 1, total_frames - 1)
        else:
            frame_indices = list(range(total_frames))
            next_frame_indices = np.asarray(frame_indices)
        ep_len = len(frame_indices)

        ep_ann = _ANNOTATIONS['datasets'].get(dataset_key, {}).get(str(ep_idx))
        has_subtask_annotations = ep_ann is not None and ep_ann['boundaries'] is not None
        if has_subtask_annotations:
            segments = _get_subtask_segments(dataset_key, ep_idx, ep_len)
        else:
            # Empty segments -> _get_subtask_info_for_step returns dummy
            # null values for every step.
            segments = []

        # B1: observation-like arrays are shifted back by one at load time
        # (new[0] = old[0], new[t] = old[t-1] for t >= 1) so that index t now
        # holds the pre-action-t state. target_tcp_pose and relative_action
        # are action arrays and stay at their original indices.
        left_joints = _shift_state_back(f['obses/state/left/joint_qpos'][:])
        left_gripper = _shift_state_back(f['obses/state/left/gripper_pos'][:, 0])
        right_joints = _shift_state_back(f['obses/state/right/joint_qpos'][:])
        right_gripper = _shift_state_back(f['obses/state/right/gripper_pos'][:, 0])
        left_tcp = _shift_state_back(f['obses/state/left/tcp_pose'][:])
        right_tcp = _shift_state_back(f['obses/state/right/tcp_pose'][:])
        left_target_tcp = f['obses/state/left/target_tcp_pose'][:]
        right_target_tcp = f['obses/state/right/target_tcp_pose'][:]
        relative_action = f['actions/relative_action'][:]
        intervention_indices = None
        if 'interventions' in f['metadata']:
            intervention_indices = f['metadata/interventions'][:]
        images_right_top = _shift_state_back(f['obses/images/right/top'][:])
        images_left_wrist = _shift_state_back(f['obses/images/left/wrist'][:])
        images_right_wrist = _shift_state_back(f['obses/images/right/wrist'][:])

        first_top_img = Image.open(io.BytesIO(images_right_top[0].tobytes()))
        first_left_wrist_img = Image.open(io.BytesIO(images_left_wrist[0].tobytes()))
        first_right_wrist_img = Image.open(io.BytesIO(images_right_wrist[0].tobytes()))
        camera_shapes = [
            np.array([first_top_img.size[1], first_top_img.size[0], 3], dtype = np.int32),
            np.array([first_left_wrist_img.size[1], first_left_wrist_img.size[0], 3], dtype = np.int32),
            np.array([first_right_wrist_img.size[1], first_right_wrist_img.size[0], 3], dtype = np.int32),
        ]
        is_intervention = _build_is_intervention(frame_indices, intervention_indices)

        steps = []

        for i, fi in enumerate(frame_indices):
            next_fi = next_frame_indices[i]
            state = _build_state(left_joints[fi], left_gripper[fi], right_joints[fi], right_gripper[fi])
            action = np.zeros(_MAX_ACTION_DIM, dtype = np.float32)
            action[6] = _build_gripper_action(left_gripper[next_fi], relative_action[next_fi, 6])
            action[13] = _build_gripper_action(right_gripper[next_fi], relative_action[next_fi, 13])
            eef_state = _build_eef(left_tcp[fi], right_tcp[fi])
            eef_action = _build_eef(left_target_tcp[next_fi], right_target_tcp[next_fi])

            img_right_top = _decode_and_reencode_jpeg(images_right_top[fi])
            img_left_wrist = _decode_and_reencode_jpeg(images_left_wrist[fi])
            img_right_wrist = _decode_and_reencode_jpeg(images_right_wrist[fi])

            sub = _get_subtask_info_for_step(i, segments, ep_len)

            step = {
                'observation/image/cam_0': img_right_top,
                'observation/image/cam_1': img_left_wrist,
                'observation/image/cam_2': img_right_wrist,
                'observation/state': state,
                'action': action,
                'relative_action': relative_action[next_fi].astype(np.float32),
                'is_first': (i == 0),
                'is_terminal': (i == ep_len - 1),
                'is_intervention': is_intervention[i],
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
            'repo_id': 'dexterous_hang',
            'robot_type': 'xarm',
            'fps': np.float32(_FPS),
            'camera_names': ['right_top', 'left_wrist', 'right_wrist'],
            'camera_shapes': camera_shapes,
            'num_cameras': np.int64(len(camera_shapes)),
            'state_feature_names': _STATE_FEATURE_NAMES,
            'action_feature_names': _ACTION_FEATURE_NAMES,
            'subtasks': _SUBTASK_NAMES + ['dummy'],
            'task_description': task,
            'has_subtask_annotations': np.bool_(has_subtask_annotations),
        },
    }
    return episode_path, sample

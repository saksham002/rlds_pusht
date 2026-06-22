"""slurm_rlds config: real-world dual-xArm packing dataset (LeRobot -> RLDS).

Each RLDS episode is a COMBINED episode (chunk): a contiguous run of short
subtask mp4s within one round, grouped by base-camera scene continuity
(see scripts/build_chunks_json.py -> chunks.json). Subtask frames are
concatenated in order with NO frame dropping.

State/action (per A1/A2 decisions):
- Source observation.state / action are 20-dim per frame, per arm
  [pos(3), 6D-rotation(6), gripper(1)], expressed in the robot BASE frame.
- The 6D rotation is the first two COLUMNS of the EEF-wrt-base rotation matrix:
  col0 = [r00,r10,r20], col1 = [r01,r11,r21], col2 = col0 x col1.
- Output is 14-dim: per arm [x, y, z, roll, pitch, yaw, gripper] with gripper
  at positions 7 and 14. Euler is scipy 'xyz' extrinsic, radians. Gripper is
  the raw encoder value (~80-842). The `action` column is used directly for
  the action (reformatted identically to state).

Per-frame subtask fields are single-valued for the ACTIVE subtask (no
length-5 slot arrays).

The LeRobot data is already (obs_t, action_t) aligned by the converter
(convert_xarm_data_to_lerobot.py: action[i] == state_pose[i+1]), so state,
images and action are read straight through at their original indices.

Usage (single worker, local debug):
    cd slurm_rlds/
    python -m framework.runner \
        --config ../lerobot/realworld_xarm_packing_config.py \
        --data_dir /tmp/xarm_packing/0 --worker_id 0 --num_workers 8
"""
import json
import os

import av
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_NAME = 'realworld_xarm_packing'
DATASET_VERSION = '1.0.0'

_DATA_ROOT = '/data/group_data/rl/saksham3/realworld_xarm_packing_lerobot'
_CHUNKS_PATH = os.path.join(_DATA_ROOT, 'chunks.json')

_CAMERAS = ['base', 'left_wrist', 'right_wrist']   # cam_0, cam_1, cam_2
_MAX_CAMERAS = 3
_IMAGE_SIZE = (480, 480)
_JPEG_QUALITY = 95
_STATE_DIM = 14
_ACTION_DIM = 14
_FPS = 60.0
_VAL_FRACTION = 0.05
_ROBOT_TYPE = 'dual_xarm'

_ARM_NAMES = ['left', 'right']
_STATE_FEATURE_NAMES = []
for _arm in _ARM_NAMES:
    _STATE_FEATURE_NAMES += [f'{_arm}_x', f'{_arm}_y', f'{_arm}_z',
                             f'{_arm}_roll', f'{_arm}_pitch', f'{_arm}_yaw',
                             f'{_arm}_gripper']
_ACTION_FEATURE_NAMES = list(_STATE_FEATURE_NAMES)


def _load_chunks():
    with open(_CHUNKS_PATH) as f:
        data = json.load(f)
    return {c['global_chunk_index']: c for c in data['chunks']}

_CHUNKS = _load_chunks()


def _split_indices():
    """Deterministic per-folder 5% val split (seed 86).

    Stratify by repo_id (folder): each folder's chunks are split 95/5 so every
    folder is represented in both train and val, rather than pooling all chunks
    and risking a folder being absent from a split.
    """
    by_folder = {}
    for ci, ch in _CHUNKS.items():
        by_folder.setdefault(ch['repo_id'], []).append(ci)

    rng = np.random.RandomState(86)
    val_set = set()
    for repo_id in sorted(by_folder):
        idx = sorted(by_folder[repo_id])
        perm = rng.permutation(len(idx))
        num_val = max(1, int(len(idx) * _VAL_FRACTION))
        val_set.update(idx[perm[i]] for i in range(num_val))

    all_idx = sorted(_CHUNKS.keys())
    train = [i for i in all_idx if i not in val_set]
    val = [i for i in all_idx if i in val_set]
    return train, val

_TRAIN_CHUNKS, _VAL_CHUNKS = _split_indices()


def get_features():
    obs = {}
    for i in range(_MAX_CAMERAS):
        obs[f'observation/image/cam_{i}'] = tfds.features.Tensor(shape = (), dtype = tf.string)
    obs['observation/state'] = tfds.features.Tensor(shape = (_STATE_DIM,), dtype = np.float32)

    return tfds.features.FeaturesDict({
        'steps': tfds.features.Dataset({
            **obs,
            'action': tfds.features.Tensor(shape = (_ACTION_DIM,), dtype = np.float32),
            'is_first': tfds.features.Scalar(dtype = np.bool_),
            'is_terminal': tfds.features.Scalar(dtype = np.bool_),
            'frame_index': tfds.features.Scalar(dtype = np.int64),
            'task': tfds.features.Text(),
            'episode_index': tfds.features.Scalar(dtype = np.int64),
            'index': tfds.features.Scalar(dtype = np.int64),
            'subtask': tfds.features.Text(),
            'subtask_index': tfds.features.Scalar(dtype = np.int32),
            'steps_to_subtask_end': tfds.features.Scalar(dtype = np.int32),
            'subtask_len': tfds.features.Scalar(dtype = np.int32),
            'subtask_is_first': tfds.features.Scalar(dtype = np.bool_),
            'subtask_is_last': tfds.features.Scalar(dtype = np.bool_),
            'repo_index': tfds.features.Scalar(dtype = np.int32),
        }),
        'episode_metadata': tfds.features.FeaturesDict({
            'repo_id': tfds.features.Text(),
            'repo_index': tfds.features.Scalar(dtype = np.int32),
            'robot_type': tfds.features.Text(),
            'fps': tfds.features.Scalar(dtype = np.float32),
            'camera_names': tfds.features.Sequence(tfds.features.Text()),
            'camera_shapes': tfds.features.Sequence(tfds.features.Tensor(shape = (3,), dtype = np.int32)),
            'num_cameras': tfds.features.Scalar(dtype = np.int64),
            'state_feature_names': tfds.features.Sequence(tfds.features.Text()),
            'action_feature_names': tfds.features.Sequence(tfds.features.Text()),
            'subtasks': tfds.features.Sequence(tfds.features.Text()),
            'task_description': tfds.features.Text(),
            'num_subtasks': tfds.features.Scalar(dtype = np.int32),
        }),
    })


def get_episodes(split):
    if split == 'train':
        ids = _TRAIN_CHUNKS
    elif split == 'val':
        ids = _VAL_CHUNKS
    else:
        return []
    return [str(i) for i in ids]


def _pose20_to_14(vec20):
    """Convert a 20-dim [pos3, rot6D(cols), grip] x{L,R} pose to 14-dim
    [x,y,z,roll,pitch,yaw,grip] x{L,R}. rot6D = first two COLUMNS of R_eef_in_base
    (R[:,0], R[:,1]), matching the converter; recover R as [col0 | col1 | col0xcol1]."""
    out = np.zeros(14, dtype = np.float32)
    for arm in range(2):
        b = arm * 10
        pos = vec20[b: b + 3]
        col0 = vec20[b + 3: b + 6]
        col1 = vec20[b + 6: b + 9]
        grip = vec20[b + 9]
        col2 = np.cross(col0, col1)
        R = np.stack([col0, col1, col2], axis = 1).astype(np.float64)
        euler = Rotation.from_matrix(R).as_euler('xyz').astype(np.float32)
        o = arm * 7
        out[o: o + 3] = pos
        out[o + 3: o + 6] = euler
        out[o + 6] = grip
    return out


def _encode_frame(frame_rgb):
    """uint8 (H,W,3) -> resize 480x480 -> JPEG q95 bytes."""
    img = tf.image.resize(frame_rgb.astype(np.float32) / 255.0, _IMAGE_SIZE).numpy()
    img = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    return tf.io.encode_jpeg(img, quality = _JPEG_QUALITY).numpy()


def _decode_video(path, expected):
    frames = []
    with av.open(path) as c:
        for fr in c.decode(video = 0):
            frames.append(fr.to_ndarray(format = 'rgb24'))
    assert len(frames) == expected, f'{path}: {len(frames)} frames, expected {expected}'
    return frames


def parse_episode(chunk_id):
    chunk = _CHUNKS[int(chunk_id)]
    rdir = os.path.join(_DATA_ROOT, chunk['repo_id'])
    total = chunk['total_frames']
    repo_index = chunk['repo_index']
    combined = chunk['combined_annotation']

    steps = []
    for sub_idx, sub in enumerate(chunk['subtasks']):
        ep = sub['episode_index']
        length = sub['length']
        pq = pd.read_parquet(
            os.path.join(rdir, 'data', 'chunk-000', f'episode_{ep:06d}.parquet'),
            columns = ['observation.state', 'action'])
        states = np.stack(pq['observation.state'].values).astype(np.float32)
        actions = np.stack(pq['action'].values).astype(np.float32)
        assert len(states) == length, f'{chunk["repo_id"]} ep{ep}: {len(states)} vs {length}'

        cam_frames = []
        for cam in _CAMERAS:
            vp = os.path.join(rdir, 'videos', 'chunk-000',
                              f'observation.images.{cam}', f'episode_{ep:06d}.mp4')
            cam_frames.append(_decode_video(vp, length))

        start = sub['start_frame']
        end = sub['end_frame']
        for t_local in range(length):
            t = start + t_local                       # frame index within combined episode
            step = {}
            for ci in range(_MAX_CAMERAS):
                step[f'observation/image/cam_{ci}'] = _encode_frame(cam_frames[ci][t_local])
            step['observation/state'] = _pose20_to_14(states[t_local])
            step['action'] = _pose20_to_14(actions[t_local])
            step['is_first'] = (t == 0)
            step['is_terminal'] = (t == total - 1)
            step['frame_index'] = np.int64(t)
            step['task'] = combined
            step['episode_index'] = np.int64(chunk['global_chunk_index'])
            step['index'] = np.int64(t)
            step['subtask'] = sub['task']
            step['subtask_index'] = np.int32(sub_idx)
            step['steps_to_subtask_end'] = np.int32(end - t)
            step['subtask_len'] = np.int32(length)
            step['subtask_is_first'] = (t == start)
            step['subtask_is_last'] = (t == end)
            step['repo_index'] = np.int32(repo_index)

            for k, v in step.items():
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
                    assert np.all(np.isfinite(v)), f'non-finite {k} in chunk {chunk_id} ep{ep} frame {t_local}'
            steps.append(step)

    assert len(steps) == total, f'chunk {chunk_id}: {len(steps)} steps vs {total}'

    camera_shapes = [np.array([_IMAGE_SIZE[0], _IMAGE_SIZE[1], 3], dtype = np.int32)
                     for _ in _CAMERAS]
    sample = {
        'steps': steps,
        'episode_metadata': {
            'repo_id': chunk['repo_id'],
            'repo_index': np.int32(repo_index),
            'robot_type': _ROBOT_TYPE,
            'fps': np.float32(_FPS),
            'camera_names': _CAMERAS,
            'camera_shapes': camera_shapes,
            'num_cameras': np.int64(_MAX_CAMERAS),
            'state_feature_names': _STATE_FEATURE_NAMES,
            'action_feature_names': _ACTION_FEATURE_NAMES,
            'subtasks': [s['task'] for s in chunk['subtasks']],
            'task_description': combined,
            'num_subtasks': np.int32(chunk['num_subtasks']),
        },
    }
    return str(chunk['global_chunk_index']), sample

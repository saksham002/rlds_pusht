"""slurm_rlds config: YAM-leader lego LeRobot dataset -> RLDS.

A fresh dataset, separate from lego 1.0.0-3.0.0 (the GELLO-leader 3lego rounds): new leader arms
(YAM teaching handles), a different follower gripper (linear_4310, stroke 0..1 instead of
0..0.85), per-block subtasks with a different wording, and its own norm stats, policy and
counterfactual store. Because nothing downstream is shared, none of packing_config.py's
yam_3lego compatibility machinery is carried over: no round-local episode indices, no
source_episode_index, no adversarial-by-folder rule, no partial-label files. What IS kept is the
per-frame / episode schema the openpi LeRobotRldsDataset reads, so the same loader, configs and
filters work unchanged:

- observation/image/cam_{0,1,2} = top, left_wrist, right_wrist (224x224 JPEG q95).
- observation/state, action: 14-dim joint space [left_joint(6), left_gripper, right_joint(6),
  right_gripper], written through from LeRobot unchanged (absolute leader targets as actions).
- eef_sim_pose_state / eef_sim_pose_action: 12-dim [x,y,z,roll,pitch,yaw] x{L,R}, wrist FLANGE
  (link6) in each arm's base frame via the pinned yam_fk.py + vendor/yam_vendor_kin.xml. Same
  convention as lego 3.0.0 and openpi's vendored copy.
- is_partial is False on every step and episode_metadata/is_adversarial is False on every
  episode: both fields are redundant for this round but are kept because the lego openpi configs
  set filter_partial=True / filter_adversarial=True, which raise when the field is absent.
- task_id = tray permutation index 0-5 (rotation_index mod 6 from annotations.json).

Episodes are chunks from scripts/build_chunks_json_lego.py: one per raw recording, in recording
order. episode_index = global_chunk_index (0..N-1) and doubles as the TFDS example key, so
episode ORDER and identity are fixed by chunks.json alone. Split: deterministic 5% val (seed 86).

Environment:
    LEGO_LEROBOT_ROOT  directory containing the converted LeRobot dataset folder
    LEGO_REPO          that folder's name (default: lego_pc0)
    LEGO_CHUNKS_PATH   chunks.json (default: <LEGO_LEROBOT_ROOT>/<LEGO_REPO>/chunks.json)

Usage (single worker, local debug):
    cd slurm_rlds/
    LEGO_LEROBOT_ROOT=/data/hf python -m framework.runner \
        --config ../lerobot/lego_config.py --data_dir /tmp/lego_pc0/0 --worker_id 0 --num_workers 8
"""
import importlib.util
import json
import os

import av
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_NAME = 'lego_pc0'
DATASET_VERSION = '1.0.0'

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get('LEGO_LEROBOT_ROOT', '/data/group_data/rl/saksham3/hf')
_REPO = os.environ.get('LEGO_REPO', 'lego_pc0')
_CHUNKS_PATH = os.environ.get('LEGO_CHUNKS_PATH', os.path.join(_DATA_ROOT, _REPO, 'chunks.json'))

_CAMERAS = ['top', 'left_wrist', 'right_wrist']   # -> cam_0, cam_1, cam_2
_IMAGE_SIZE = (224, 224)                           # native LeRobot size; the resize is a no-op
_JPEG_QUALITY = 95
_STATE_DIM = 14
_ACTION_DIM = 14
_FPS = 60.0
_VAL_FRACTION = 0.05
_SPLIT_SEED = 86
_ROBOT_TYPE = 'yam'

_STATE_FEATURE_NAMES = []
for _arm in ['left', 'right']:
    _STATE_FEATURE_NAMES += [f'{_arm}_joint_{_i}' for _i in range(6)] + [f'{_arm}_gripper']
_ACTION_FEATURE_NAMES = list(_STATE_FEATURE_NAMES)

_EEF_POSE_FEATURE_NAMES = []
for _arm in ['left', 'right']:
    _EEF_POSE_FEATURE_NAMES += [f'{_arm}_x', f'{_arm}_y', f'{_arm}_z',
                                f'{_arm}_roll', f'{_arm}_pitch', f'{_arm}_yaw']
_EEF_POSE_DIM = len(_EEF_POSE_FEATURE_NAMES)

# Loaded by path: the framework imports this config by file, so lerobot/ is not necessarily on
# sys.path. yam_fk resolves vendor/yam_vendor_kin.xml relative to its own __file__.
_spec = importlib.util.spec_from_file_location('yam_fk', os.path.join(_HERE, 'yam_fk.py'))
_yam_fk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_yam_fk)
_FK = _yam_fk.YamFK()


def _load_chunks():
    with open(_CHUNKS_PATH) as f:
        data = json.load(f)
    chunks = {chunk['global_chunk_index']: chunk for chunk in data['chunks']}
    assert sorted(chunks) == list(range(len(chunks))), 'global_chunk_index must be 0..N-1 (see build_chunks_json_lego.py)'
    return chunks

_CHUNKS = _load_chunks()


def _split_indices():
    """Deterministic 5% val split over chunks (seed 86); at least one val chunk."""
    all_idx = sorted(_CHUNKS)
    rng = np.random.RandomState(_SPLIT_SEED)
    perm = rng.permutation(len(all_idx))
    num_val = max(1, int(len(all_idx) * _VAL_FRACTION))
    val_set = {all_idx[perm[i]] for i in range(num_val)}
    train = [i for i in all_idx if i not in val_set]
    val = [i for i in all_idx if i in val_set]
    return train, val

_TRAIN_CHUNKS, _VAL_CHUNKS = _split_indices()


def get_features():
    obs = {}
    for i in range(len(_CAMERAS)):
        obs[f'observation/image/cam_{i}'] = tfds.features.Tensor(shape = (), dtype = tf.string)
    obs['observation/state'] = tfds.features.Tensor(shape = (_STATE_DIM,), dtype = np.float32)

    return tfds.features.FeaturesDict({
        'steps': tfds.features.Dataset({
            **obs,
            'eef_sim_pose_state': tfds.features.Tensor(shape = (_EEF_POSE_DIM,), dtype = np.float32),
            'eef_sim_pose_action': tfds.features.Tensor(shape = (_EEF_POSE_DIM,), dtype = np.float32),
            'is_partial': tfds.features.Scalar(dtype = np.bool_),
            'task_id': tfds.features.Scalar(dtype = np.int32),
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
            'recording': tfds.features.Text(),
            'robot_type': tfds.features.Text(),
            'fps': tfds.features.Scalar(dtype = np.float32),
            'camera_names': tfds.features.Sequence(tfds.features.Text()),
            'camera_shapes': tfds.features.Sequence(tfds.features.Tensor(shape = (3,), dtype = np.int32)),
            'num_cameras': tfds.features.Scalar(dtype = np.int64),
            'state_feature_names': tfds.features.Sequence(tfds.features.Text()),
            'action_feature_names': tfds.features.Sequence(tfds.features.Text()),
            'eef_pose_feature_names': tfds.features.Sequence(tfds.features.Text()),
            'subtasks': tfds.features.Sequence(tfds.features.Text()),
            'subtask_is_partial': tfds.features.Sequence(tfds.features.Scalar(dtype = np.bool_)),
            'task_description': tfds.features.Text(),
            'num_subtasks': tfds.features.Scalar(dtype = np.int32),
            'task_id': tfds.features.Scalar(dtype = np.int32),
            'is_adversarial': tfds.features.Scalar(dtype = np.bool_),
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


def _joints_to_eef(vec14):
    """14-dim joint vector -> 12-dim flange pose [x,y,z,roll,pitch,yaw] x{L,R}.

    The FK input skips the gripper at index 6 (left) and 13 (right). Euler is scipy 'xyz'
    (extrinsic) radians, matching lego 3.0.0 and openpi's DeltaActions/AbsoluteActions.
    """
    out = np.zeros(_EEF_POSE_DIM, dtype = np.float32)
    for arm, joint_slice in enumerate([slice(0, 6), slice(7, 13)]):
        pos, quat_xyzw = _FK.fk(np.asarray(vec14[joint_slice], dtype = np.float64))
        offset = arm * 6
        out[offset: offset + 3] = pos
        out[offset + 3: offset + 6] = Rotation.from_quat(quat_xyzw).as_euler('xyz')
    return out


def _encode_frame(frame_rgb):
    """uint8 (H,W,3) -> resize to _IMAGE_SIZE -> JPEG bytes."""
    img = tf.image.resize(frame_rgb.astype(np.float32) / 255.0, _IMAGE_SIZE).numpy()
    img = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    return tf.io.encode_jpeg(img, quality = _JPEG_QUALITY).numpy()


def _decode_video(path, expected):
    frames = []
    with av.open(path) as container:
        for frame in container.decode(video = 0):
            frames.append(frame.to_ndarray(format = 'rgb24'))
    assert len(frames) == expected, f'{path}: {len(frames)} frames, expected {expected}'
    return frames


def parse_episode(chunk_id):
    chunk = _CHUNKS[int(chunk_id)]
    repo_dir = os.path.join(_DATA_ROOT, chunk['repo_id'])
    total = chunk['total_frames']
    episode_index = np.int64(chunk['global_chunk_index'])
    task_id = np.int32(chunk['task_id'])
    combined = chunk['combined_annotation']

    steps = []
    for sub_idx, sub in enumerate(chunk['subtasks']):
        lerobot_episode = sub['episode_index']
        length = sub['length']
        lerobot_chunk = f'chunk-{lerobot_episode // 1000:03d}'   # LeRobot v2.1 stores 1000 episodes per chunk dir
        parquet = pd.read_parquet(
            os.path.join(repo_dir, 'data', lerobot_chunk, f'episode_{lerobot_episode:06d}.parquet'),
            columns = ['observation.state', 'action'])
        states = np.stack(parquet['observation.state'].values).astype(np.float32)
        actions = np.stack(parquet['action'].values).astype(np.float32)
        assert len(states) == length, f'{chunk["repo_id"]} ep{lerobot_episode}: {len(states)} vs {length}'
        assert states.shape[1] == _STATE_DIM and actions.shape[1] == _ACTION_DIM, (
            f'{chunk["repo_id"]} ep{lerobot_episode}: state {states.shape}, action {actions.shape}')

        cam_frames = []
        for cam in _CAMERAS:
            video_path = os.path.join(repo_dir, 'videos', lerobot_chunk,
                                      f'observation.images.{cam}', f'episode_{lerobot_episode:06d}.mp4')
            cam_frames.append(_decode_video(video_path, length))

        start = sub['start_frame']
        end = sub['end_frame']   # inclusive
        for t_local in range(length):
            t = start + t_local
            step = {}
            for cam_idx in range(len(_CAMERAS)):
                step[f'observation/image/cam_{cam_idx}'] = _encode_frame(cam_frames[cam_idx][t_local])
            step['observation/state'] = states[t_local].copy()
            step['action'] = actions[t_local].copy()
            step['eef_sim_pose_state'] = _joints_to_eef(states[t_local])
            step['eef_sim_pose_action'] = _joints_to_eef(actions[t_local])
            step['is_partial'] = np.bool_(False)
            step['task_id'] = task_id
            step['is_first'] = (t == 0)
            step['is_terminal'] = (t == total - 1)
            step['frame_index'] = np.int64(t)
            step['task'] = combined
            step['episode_index'] = episode_index
            step['index'] = np.int64(t)
            step['subtask'] = sub['task']
            step['subtask_index'] = np.int32(sub_idx)
            step['steps_to_subtask_end'] = np.int32(end - t)
            step['subtask_len'] = np.int32(length)
            step['subtask_is_first'] = (t == start)
            step['subtask_is_last'] = (t == end)
            step['repo_index'] = np.int32(chunk['repo_index'])

            for key, value in step.items():
                if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
                    assert np.all(np.isfinite(value)), f'non-finite {key} in chunk {chunk_id} ep{lerobot_episode} frame {t_local}'
            steps.append(step)

    assert len(steps) == total, f'chunk {chunk_id}: {len(steps)} steps vs {total}'

    camera_shapes = [np.array([_IMAGE_SIZE[0], _IMAGE_SIZE[1], 3], dtype = np.int32) for _ in _CAMERAS]
    sample = {
        'steps': steps,
        'episode_metadata': {
            'repo_id': chunk['repo_id'],
            'repo_index': np.int32(chunk['repo_index']),
            'recording': chunk['recording'],
            'robot_type': _ROBOT_TYPE,
            'fps': np.float32(_FPS),
            'camera_names': _CAMERAS,
            'camera_shapes': camera_shapes,
            'num_cameras': np.int64(len(_CAMERAS)),
            'state_feature_names': _STATE_FEATURE_NAMES,
            'action_feature_names': _ACTION_FEATURE_NAMES,
            'eef_pose_feature_names': list(_EEF_POSE_FEATURE_NAMES),
            'subtasks': [sub['task'] for sub in chunk['subtasks']],
            'subtask_is_partial': [np.bool_(False) for _ in chunk['subtasks']],
            'task_description': combined,
            'num_subtasks': np.int32(chunk['num_subtasks']),
            'task_id': task_id,
            'is_adversarial': np.bool_(False),
        },
    }
    return str(chunk['global_chunk_index']), sample

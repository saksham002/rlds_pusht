"""slurm_rlds config: AgiBot World 2026 ImitationLearning slice (13 tasks).

Converts the LeRobot v2.1 shards mirrored at _DATA_ROOT (layout:
ImitationLearning/{CommercialSpaces,Home}/task_XXXX/<start>_<end>/) into RLDS
tfrecords in the RoboCOIN loader format, using the sentence-level Task Frame
annotations from each shard's meta/info.json as subtasks.

Schema notes (verified against the metadata mirror of all 146 IL shards):
- State/action are written as individual components, preserving the original
  LeRobot field names from info.json field_descriptions. Components whose
  dimensionality varies across shards (state/end/wrench 12|24,
  state/end/velocity 12|24, state/robot/position 0|3, state/robot/orientation
  0|4, action/robot/velocity 2|6) are zero-padded to the max; the actual
  per-shard dims are recorded in episode_metadata state/action_feature_dims.
- Subtask segments come from key_frame[ep]['dual'] entries whose
  frame_type_name normalizes to 'taskframe'/'subtaskframe' (spelling varies
  across tasks). Segments are [start, end) frame intervals; max concurrent
  overlap observed is 2, filled into slots 0..4 ordered by start.
- Error Frame spans drive the per-step is_error_frame flag.
- Quaternions (state/end/arm_orientation, action/end/orientation) are xyzw,
  left arm then right (per the official AgiBot World proprio docs);
  eef_sim_pose_* converts them to [pos(3), euler_xyz(3)] x {left, right}.
- has_base_motion in episode_metadata is computed from the per-episode
  action/robot/velocity min/max in episodes_stats.jsonl.
- Images are resized to 480x480 (RoboCOIN-style tf.image.resize on [0,1]
  floats) and JPEG-encoded at quality 95; episodes whose total image bytes
  would exceed the protobuf 2 GB example limit fall back to quality 90 then
  85 (recorded in episode_metadata jpeg_quality), and raise if even q85 does
  not fit. camera_shapes records the SOURCE video resolutions from info.json.

Usage (single worker, local debug):
    cd slurm_rlds/
    python -m framework.runner \
        --config ../agibot/agibot_config.py \
        --data_dir /tmp/agibot/0 \
        --worker_id 0 \
        --num_workers 64
"""
import glob
import json
import os
import re

import av
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_NAME = 'agibot'
DATASET_VERSION = '1.0.0'

_DATA_ROOT = os.environ.get('AGIBOT_DATA_ROOT', '/data/group_data/rl/datasets/agibot_world')
_IL_ROOT = os.path.join(_DATA_ROOT, 'ImitationLearning')

# All 13 ImitationLearning tasks; repo_index = position in this list.
_TASK_IDS = [
    'task_3400', 'task_3401', 'task_3402', 'task_3404', 'task_3405',
    'task_3477', 'task_3641', 'task_3705', 'task_3777', 'task_4053',
    'task_4542', 'task_4713', 'task_4799',
]
_TASK_TO_REPO_INDEX = {t: i for i, t in enumerate(_TASK_IDS)}

# When set (e.g. 'task_3400'), restrict the build to a single task. Used by
# scripts/build.sh to run one tfds build per task so a preemption only loses
# the in-progress task, not the worker's whole slice.
_TASK_FILTER = os.environ.get('AGIBOT_TASK', '').strip() or None
if _TASK_FILTER is not None:
    assert _TASK_FILTER in _TASK_TO_REPO_INDEX, f'Unknown AGIBOT_TASK {_TASK_FILTER}'

# cam_0 = front/high, cam_1 = left wrist, cam_2 = right wrist (RoboCOIN order)
_CAMERA_NAMES = ['top_head', 'hand_left', 'hand_right']
_MAX_CAMERAS = 3
_MAX_SUBTASKS = 5
_VAL_FRACTION = 0.05
_NULL_SUBTASK = 'null'

# Canonical component layout: (lerobot field name, max dims across all IL shards).
# Order matches the field_descriptions insertion order, identical in every shard.
_STATE_FIELDS = [
    ('state/left_effector/position', 1),
    ('state/right_effector/position', 1),
    ('state/end/wrench', 24),
    ('state/end/velocity', 24),
    ('state/end/arm_orientation', 8),
    ('state/end/arm_position', 6),
    ('state/joint/position', 14),
    ('state/joint/effort', 14),
    ('state/joint/velocity', 14),
    ('state/head/position', 3),
    ('state/waist/position', 5),
    ('state/robot/position', 3),
    ('state/robot/orientation', 4),
    ('extrinsic_end_T_hand_left_rgbd_aligned/rotation_matrix', 9),
    ('extrinsic_end_T_hand_right_rgbd_aligned/rotation_matrix', 9),
    ('extrinsic_end_T_head_left_fisheye_aligned/rotation_matrix', 9),
    ('extrinsic_end_T_head_right_fisheye_aligned/rotation_matrix', 9),
    ('extrinsic_end_T_head_front_rgbd_aligned/rotation_matrix', 9),
    ('extrinsic_end_T_head_back_fisheye_aligned/rotation_matrix', 9),
    ('extrinsic_end_T_hand_left_rgbd_aligned/translation_vector', 3),
    ('extrinsic_end_T_hand_right_rgbd_aligned/translation_vector', 3),
    ('extrinsic_end_T_head_left_fisheye_aligned/translation_vector', 3),
    ('extrinsic_end_T_head_right_fisheye_aligned/translation_vector', 3),
    ('extrinsic_end_T_head_front_rgbd_aligned/translation_vector', 3),
    ('extrinsic_end_T_head_back_fisheye_aligned/translation_vector', 3),
]
_ACTION_FIELDS = [
    ('action/left_effector/position', 1),
    ('action/right_effector/position', 1),
    ('action/end/position', 6),
    ('action/end/orientation', 8),
    ('action/joint/position', 14),
    ('action/head/position', 3),
    ('action/waist/position', 5),
    ('action/robot/velocity', 6),
]
_STATE_FEATURE_NAMES = [n for n, _ in _STATE_FIELDS]
_ACTION_FEATURE_NAMES = [n for n, _ in _ACTION_FIELDS]


def _norm_frame_type(s):
    return re.sub(r'[^a-z]', '', s.lower())

_SUBTASK_FRAME_TYPES = {'taskframe', 'subtaskframe'}
_ERROR_FRAME_TYPES = {'errorframe'}


# ─── Shard metadata loading ───────────────────────────────────────────────────

_SHARD_CACHE = {}


def _field_spec(field_descriptions, canonical_fields, feat_name, shard_dir):
    """Map a shard's field_descriptions onto the canonical layout.

    Returns list of (name, start_index, dims) aligned with canonical_fields;
    dims = 0 when the field is absent (or 0-dim) in this shard.
    """
    canonical_names = [n for n, _ in canonical_fields]
    unknown = set(field_descriptions.keys()) - set(canonical_names)
    assert not unknown, f'Unknown {feat_name} fields {unknown} in {shard_dir}'

    spec = []
    for name, max_dim in canonical_fields:
        fd = field_descriptions.get(name)
        if fd is None or fd['dimensions'] == 0:
            spec.append((name, 0, 0))
            continue
        idx = fd['indices']
        assert fd['dimensions'] == len(idx) <= max_dim, (
            f'{feat_name} field {name} has {len(idx)} dims > max {max_dim} in {shard_dir}'
        )
        assert idx == list(range(idx[0], idx[0] + len(idx))), (
            f'Non-contiguous indices for {name} in {shard_dir}'
        )
        spec.append((name, idx[0], len(idx)))
    return spec


def _load_shard_meta(shard_dir):
    """Load and cache trimmed per-shard metadata (info.json annotations, episodes, stats)."""
    if shard_dir in _SHARD_CACHE:
        return _SHARD_CACHE[shard_dir]

    with open(os.path.join(shard_dir, 'meta', 'info.json'), 'r') as f:
        info = json.load(f)

    assert info['robot_type'] == 'g2a', f"Unexpected robot_type {info['robot_type']} in {shard_dir}"
    fps = float(info['fps'])

    features = info['features']
    state_spec = _field_spec(features['observation.state']['field_descriptions'],
                             _STATE_FIELDS, 'state', shard_dir)
    action_spec = _field_spec(features['action']['field_descriptions'],
                              _ACTION_FIELDS, 'action', shard_dir)

    camera_shapes = []
    for cam in _CAMERA_NAMES:
        shape = features[f'observation.images.{cam}']['shape']
        camera_shapes.append(np.array(shape, dtype = np.int32))

    # Episodes (episode_index is local 0-based within the shard)
    episodes = {}
    with open(os.path.join(shard_dir, 'meta', 'episodes.jsonl'), 'r') as f:
        for line in f:
            if line.strip():
                ep = json.loads(line)
                episodes[ep['episode_index']] = ep

    # Per-episode base-motion flag from action/robot/velocity stats
    vel_spec = dict((n, (s, d)) for n, s, d in action_spec)['action/robot/velocity']
    vel_slice = slice(vel_spec[0], vel_spec[0] + vel_spec[1])
    has_base_motion = {}
    with open(os.path.join(shard_dir, 'meta', 'episodes_stats.jsonl'), 'r') as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                a_stats = s['stats']['action']
                vmin = np.array(a_stats['min'])[vel_slice]
                vmax = np.array(a_stats['max'])[vel_slice]
                has_base_motion[s['episode_index']] = bool(
                    np.any(np.abs(vmin) > 0) or np.any(np.abs(vmax) > 0)
                )

    # Subtask segments + error spans from key_frame (dual track; single is always empty)
    subtask_segments = {}
    error_spans = {}
    for ep_str, kf in info['key_frame'].items():
        ep_idx = int(ep_str)
        segs = []
        errs = []
        for entry in kf.get('dual', []):
            ftype = _norm_frame_type(entry['frame_type_name'])
            if ftype in _SUBTASK_FRAME_TYPES:
                segs.append({
                    'start': int(entry['start']),
                    'end': int(entry['end']),
                    'text': entry['frame_detail']['comment'],
                    'succeed': bool(entry['frame_detail']['is_result_succeed']),
                })
            elif ftype in _ERROR_FRAME_TYPES:
                errs.append((int(entry['start']), int(entry['end'])))
        segs.sort(key = lambda s: (s['start'], s['end']))
        if segs:
            subtask_segments[ep_idx] = segs
        if errs:
            error_spans[ep_idx] = errs

    meta = {
        'fps': fps,
        'chunks_size': info['chunks_size'],
        'data_path': info['data_path'],
        'video_path': info['video_path'],
        'state_spec': state_spec,
        'action_spec': action_spec,
        'camera_shapes': camera_shapes,
        'episodes': episodes,
        'has_base_motion': has_base_motion,
        'subtask_segments': subtask_segments,
        'error_spans': error_spans,
    }
    _SHARD_CACHE[shard_dir] = meta
    return meta


# ─── Episode enumeration / split ──────────────────────────────────────────────

_SPLIT_CACHE = None


def _enumerate_task_episodes():
    """Return {task_id: sorted [(shard_dir, ep_idx), ...]} for the selected IL tasks."""
    selected = [_TASK_FILTER] if _TASK_FILTER else _TASK_IDS
    per_task = {t: [] for t in selected}
    for task_dir in sorted(glob.glob(os.path.join(_IL_ROOT, '*', 'task_*'))):
        task_id = os.path.basename(task_dir)
        assert task_id in _TASK_TO_REPO_INDEX, f'Unexpected task dir {task_dir}'
        if task_id not in per_task:
            continue
        for shard_dir in sorted(glob.glob(os.path.join(task_dir, '*'))):
            if not os.path.isdir(os.path.join(shard_dir, 'meta')):
                # Known release quirk: some <start>_<end> range folders are empty.
                continue
            with open(os.path.join(shard_dir, 'meta', 'episodes.jsonl'), 'r') as f:
                for line in f:
                    if line.strip():
                        ep = json.loads(line)
                        per_task[task_id].append((shard_dir, ep['episode_index']))
    missing = [t for t in per_task if not per_task[t]]
    assert not missing, f'No episodes found for tasks {missing} under {_IL_ROOT}'
    for t in per_task:
        per_task[t].sort()
    return per_task


def _get_split_lists():
    """Deterministic per-task 5% val split (seed 86). Returns (train, val) token lists."""
    global _SPLIT_CACHE
    if _SPLIT_CACHE is not None:
        return _SPLIT_CACHE

    per_task = _enumerate_task_episodes()
    train, val = [], []
    for task_id in sorted(per_task):
        eps = per_task[task_id]
        rng = np.random.RandomState(86)
        perm = rng.permutation(len(eps))
        num_val = max(1, int(len(eps) * _VAL_FRACTION))
        val_set = set(perm[:num_val].tolist())
        for i, (shard_dir, ep_idx) in enumerate(eps):
            token = f'{shard_dir}|{ep_idx}'
            (val if i in val_set else train).append(token)

    _SPLIT_CACHE = (sorted(train), sorted(val))
    return _SPLIT_CACHE


def get_episodes(split):
    if split == 'train':
        return _get_split_lists()[0]
    elif split == 'val':
        return _get_split_lists()[1]
    return []


# ─── Feature schema ───────────────────────────────────────────────────────────

def get_features():
    step_features = {}
    for i in range(_MAX_CAMERAS):
        step_features[f'observation/image/cam_{i}'] = tfds.features.Tensor(shape = (), dtype = tf.string)
    for name, max_dim in _STATE_FIELDS + _ACTION_FIELDS:
        step_features[name] = tfds.features.Tensor(shape = (max_dim,), dtype = np.float32)

    return tfds.features.FeaturesDict({
        'steps': tfds.features.Dataset({
            **step_features,
            'is_first': tfds.features.Scalar(dtype = np.bool_),
            'is_terminal': tfds.features.Scalar(dtype = np.bool_),
            'is_error_frame': tfds.features.Scalar(dtype = np.bool_),
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
            'subtask_success': tfds.features.Tensor(shape = (5,), dtype = np.bool_),
            'steps_to_subtask_end': tfds.features.Tensor(shape = (5,), dtype = np.int32),
            'subtask_len': tfds.features.Tensor(shape = (5,), dtype = np.int32),
            'subtask_is_first': tfds.features.Tensor(shape = (5,), dtype = np.bool_),
            'subtask_is_last': tfds.features.Tensor(shape = (5,), dtype = np.bool_),
            'first_null_index': tfds.features.Scalar(dtype = np.int32),
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
            'state_feature_dims': tfds.features.Sequence(tfds.features.Scalar(dtype = np.int32)),
            'action_feature_names': tfds.features.Sequence(tfds.features.Text()),
            'action_feature_dims': tfds.features.Sequence(tfds.features.Scalar(dtype = np.int32)),
            'subtasks': tfds.features.Sequence(tfds.features.Text()),
            'task_description': tfds.features.Text(),
            'has_subtask_annotations': tfds.features.Scalar(dtype = np.bool_),
            'has_base_motion': tfds.features.Scalar(dtype = np.bool_),
            'jpeg_quality': tfds.features.Scalar(dtype = np.int32),
        }),
    })


# ─── Episode parsing ──────────────────────────────────────────────────────────

# Resize target and protobuf-size guard. A serialized tf.Example must stay under
# protobuf's hard 2 GB message limit; episodes whose summed JPEG bytes exceed
# _SIZE_LIMIT are re-encoded at the next quality in _JPEG_QUALITIES (q95 -> q90
# -> q85, no further).
_IMAGE_SIZE = (480, 480)
_SIZE_LIMIT = int(1.9e9)
_JPEG_QUALITIES = (95, 90, 85)


def _decode_video_jpegs(video_path, expected_frames, quality):
    """Decode an mp4 (AV1), resize each frame to 480x480, return per-frame JPEG bytes.

    Resize follows the RoboCOIN builder: float32 [0,1] -> tf.image.resize -> rint.
    """
    jpegs = []
    with av.open(video_path) as container:
        for frame in container.decode(video = 0):
            arr = frame.to_ndarray(format = 'rgb24').astype(np.float32) / 255.0
            scaled = tf.image.resize(arr, _IMAGE_SIZE).numpy()
            img = np.rint(scaled * 255.0).astype(np.uint8)
            jpegs.append(tf.image.encode_jpeg(img, quality = quality).numpy())
    assert len(jpegs) == expected_frames, (
        f'{video_path}: decoded {len(jpegs)} frames, expected {expected_frames}'
    )
    return jpegs


def _components_from_vector(vec, spec, canonical_fields):
    out = {}
    for (name, start, dims), (cname, max_dim) in zip(spec, canonical_fields):
        assert name == cname
        comp = np.zeros(max_dim, dtype = np.float32)
        if dims > 0:
            comp[:dims] = vec[start : start + dims]
        out[name] = comp
    return out


def _eef_12d(pos6, quat8):
    """[left pos(3) + euler_xyz(3), right pos(3) + euler_xyz(3)] from xyzw quats."""
    out = np.zeros(12, dtype = np.float32)
    for arm in range(2):
        q = quat8[4 * arm : 4 * arm + 4]
        assert np.linalg.norm(q) > 1e-6, 'Zero quaternion in EEF orientation'
        eul = R.from_quat(q).as_euler('xyz')
        out[6 * arm : 6 * arm + 3] = pos6[3 * arm : 3 * arm + 3]
        out[6 * arm + 3 : 6 * arm + 6] = eul
    return out


def _clamp_segments(segments, ep_len):
    """Clamp [start, end) segments to the episode and drop empty ones."""
    out = []
    for seg in segments:
        start = max(0, seg['start'])
        end = min(ep_len, seg['end'])
        if end > start:
            out.append({**seg, 'start': start, 'end': end})
    return out


def _subtask_info_for_step(t, segments, ep_len):
    """Fill the 5-slot subtask fields with segments active at step t (start <= t < end)."""
    names = [_NULL_SUBTASK] * _MAX_SUBTASKS
    mask = np.zeros(_MAX_SUBTASKS, dtype = np.bool_)
    success = np.zeros(_MAX_SUBTASKS, dtype = np.bool_)
    steps_to_end = np.zeros(_MAX_SUBTASKS, dtype = np.int32)
    sub_len = np.zeros(_MAX_SUBTASKS, dtype = np.int32)
    is_first = np.zeros(_MAX_SUBTASKS, dtype = np.bool_)
    is_last = np.zeros(_MAX_SUBTASKS, dtype = np.bool_)

    active = [s for s in segments if s['start'] <= t < s['end']]
    assert len(active) <= _MAX_SUBTASKS
    for pos, seg in enumerate(active):
        end_step = seg['end'] - 1
        names[pos] = seg['text']
        mask[pos] = True
        success[pos] = seg['succeed']
        steps_to_end[pos] = end_step - t
        sub_len[pos] = seg['end'] - seg['start']
        is_first[pos] = (t == seg['start'])
        is_last[pos] = (t == end_step)
    if not active:
        steps_to_end[0] = np.int32(ep_len - 1 - t)

    return {
        'names': names,
        'mask': mask,
        'success': success,
        'steps_to_end': steps_to_end,
        'len': sub_len,
        'is_first': is_first,
        'is_last': is_last,
        'first_null_index': np.int32(len(active)),
    }


def parse_episode(episode_token):
    shard_dir, ep_idx_str = episode_token.rsplit('|', 1)
    ep_idx = int(ep_idx_str)
    meta = _load_shard_meta(shard_dir)

    repo_id = os.path.relpath(shard_dir, _DATA_ROOT)
    task_id = os.path.basename(os.path.dirname(shard_dir))
    repo_index = _TASK_TO_REPO_INDEX[task_id]

    ep_info = meta['episodes'][ep_idx]
    ep_len = ep_info['length']
    task_desc = ep_info['tasks'][0]
    chunk = ep_idx // meta['chunks_size']

    parquet_path = os.path.join(
        shard_dir, meta['data_path'].format(episode_chunk = chunk, episode_index = ep_idx))
    tbl = pq.read_table(parquet_path)
    assert tbl.num_rows == ep_len, (
        f'{parquet_path}: {tbl.num_rows} rows, expected {ep_len}'
    )
    state_mat = np.stack(tbl.column('observation.state').to_pylist()).astype(np.float32)
    action_mat = np.stack(tbl.column('action').to_pylist()).astype(np.float32)
    frame_index = tbl.column('frame_index').to_numpy()
    episode_index_col = tbl.column('episode_index').to_numpy()
    index_col = tbl.column('index').to_numpy()
    assert int(episode_index_col[0]) == ep_idx

    video_paths = [
        os.path.join(shard_dir, meta['video_path'].format(
            episode_chunk = chunk,
            video_key = f'observation.images.{cam}',
            episode_index = ep_idx,
        ))
        for cam in _CAMERA_NAMES
    ]
    for jpeg_quality in _JPEG_QUALITIES:
        cam_jpegs = [_decode_video_jpegs(vp, ep_len, jpeg_quality) for vp in video_paths]
        image_bytes = sum(len(b) for jl in cam_jpegs for b in jl)
        if image_bytes <= _SIZE_LIMIT:
            break
    if image_bytes > _SIZE_LIMIT:
        raise RuntimeError(
            f'[OVERSIZED-EPISODE] {repo_id}|{ep_idx} ({ep_len} frames): image payload '
            f'{image_bytes / 1e9:.2f} GB still exceeds {_SIZE_LIMIT / 1e9:.1f} GB at '
            f'quality {_JPEG_QUALITIES[-1]} — cannot fit in protobuf 2 GB example limit'
        )

    segments = _clamp_segments(meta['subtask_segments'].get(ep_idx, []), ep_len)
    has_subtask_annotations = bool(segments)
    error_spans = meta['error_spans'].get(ep_idx, [])
    is_error = np.zeros(ep_len, dtype = np.bool_)
    for start, end in error_spans:
        is_error[max(0, start) : min(ep_len, end)] = True

    state_field_index = {name: i for i, (name, _, _) in enumerate(meta['state_spec'])}
    arm_pos_spec = meta['state_spec'][state_field_index['state/end/arm_position']]
    arm_ori_spec = meta['state_spec'][state_field_index['state/end/arm_orientation']]
    action_field_index = {name: i for i, (name, _, _) in enumerate(meta['action_spec'])}
    act_pos_spec = meta['action_spec'][action_field_index['action/end/position']]
    act_ori_spec = meta['action_spec'][action_field_index['action/end/orientation']]

    steps = []
    for t in range(ep_len):
        step = {}
        for cam_idx in range(_MAX_CAMERAS):
            step[f'observation/image/cam_{cam_idx}'] = cam_jpegs[cam_idx][t]

        step.update(_components_from_vector(state_mat[t], meta['state_spec'], _STATE_FIELDS))
        step.update(_components_from_vector(action_mat[t], meta['action_spec'], _ACTION_FIELDS))

        step['eef_sim_pose_state'] = _eef_12d(
            state_mat[t, arm_pos_spec[1] : arm_pos_spec[1] + arm_pos_spec[2]],
            state_mat[t, arm_ori_spec[1] : arm_ori_spec[1] + arm_ori_spec[2]],
        )
        step['eef_sim_pose_action'] = _eef_12d(
            action_mat[t, act_pos_spec[1] : act_pos_spec[1] + act_pos_spec[2]],
            action_mat[t, act_ori_spec[1] : act_ori_spec[1] + act_ori_spec[2]],
        )

        sub = _subtask_info_for_step(t, segments, ep_len)
        for pos in range(_MAX_SUBTASKS):
            step[f'subtask_{pos + 1}'] = sub['names'][pos]
        step['subtask_mask'] = sub['mask']
        step['subtask_success'] = sub['success']
        step['steps_to_subtask_end'] = sub['steps_to_end']
        step['subtask_len'] = sub['len']
        step['subtask_is_first'] = sub['is_first']
        step['subtask_is_last'] = sub['is_last']
        step['first_null_index'] = sub['first_null_index']

        step['is_first'] = (t == 0)
        step['is_terminal'] = (t == ep_len - 1)
        step['is_error_frame'] = bool(is_error[t])
        step['frame_index'] = np.int64(frame_index[t])
        step['task'] = task_desc
        step['episode_index'] = np.int64(ep_idx)
        step['index'] = np.int64(index_col[t])
        step['repo_index'] = np.int32(repo_index)

        for key, val in step.items():
            if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating):
                assert np.all(np.isfinite(val)), (
                    f"Non-finite values in step['{key}'] at step {t} of episode {ep_idx} in {repo_id}"
                )

        steps.append(step)

    subtask_texts = []
    for seg in segments:
        if seg['text'] not in subtask_texts:
            subtask_texts.append(seg['text'])

    sample = {
        'steps': steps,
        'episode_metadata': {
            'repo_id': repo_id,
            'robot_type': 'g2a',
            'fps': np.float32(meta['fps']),
            'camera_names': _CAMERA_NAMES,
            'camera_shapes': meta['camera_shapes'],
            'num_cameras': np.int64(len(_CAMERA_NAMES)),
            'state_feature_names': _STATE_FEATURE_NAMES,
            'state_feature_dims': [np.int32(d) for _, _, d in meta['state_spec']],
            'action_feature_names': _ACTION_FEATURE_NAMES,
            'action_feature_dims': [np.int32(d) for _, _, d in meta['action_spec']],
            'subtasks': subtask_texts + [_NULL_SUBTASK],
            'task_description': task_desc,
            'has_subtask_annotations': np.bool_(has_subtask_annotations),
            'has_base_motion': np.bool_(meta['has_base_motion'][ep_idx]),
            'jpeg_quality': np.int32(jpeg_quality),
        },
    }
    return f'{repo_id}_{ep_idx}', sample

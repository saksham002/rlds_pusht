"""
RoboCOIN Bimanual Dataset Builder using Apache Beam for parallel repo processing.
Merges multiple Hugging Face RoboCOIN datasets into a single RLDS dataset.

Requires:
    apache-beam >= 2.47.0
    tensorflow-datasets >= 4.9.0
    tensorflow >= 2.12.0

Usage:
    # Build with 8 parallel workers:
    tfds build --beam_pipeline_options="direct_num_workers=8,direct_running_mode=multi_processing" \
        --data_dir=/data/group_data/rl/saksham3/robocoin/

    # After build completes, manually clean up downloaded data:
    rm -rf /data/group_data/rl/saksham3/robocoin_bimanual/RoboCOIN/
"""

from typing import Iterator, Tuple, Any, List, Dict
import os
import json
import logging
import random
import subprocess
import time
import sys
import fcntl
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
from apache_beam.metrics import Metrics
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

assert os.environ.get("HF_TOKEN"), "HF_TOKEN environment variable must be set"


# --- Configuration ---
# Local directory for downloading HF repos (must be a local/POSIX path).
# Override via ROBOCOIN_DOWNLOAD_ROOT env var.
DOWNLOAD_ROOT = os.environ.get(
    "ROBOCOIN_DOWNLOAD_ROOT",
    "/data/group_data/rl/saksham3/robocoin_bimanual/"
)
PREFIX = "RoboCOIN/"
MAX_CAMERAS = 3
MAX_STATE_DIM = 14
MAX_ACTION_DIM = 14
ALLOWED_ROBOT_TYPES = ["Split_aloha", "Cobot_Magic", "R1_Lite"]
DEFAULT_CAMERA_ORDER = ["cam_high_rgb", "cam_left_wrist_rgb", "cam_right_wrist_rgb"]
VAL_SPLIT_RATIO = 0.05

RATE_LIMIT_MARKERS = (
    "429 Too Many Requests",
)


class OnlineStats:
    """Welford's online algorithm for streaming mean, variance, min, max."""

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype = np.float64)
        self.m2 = np.zeros(dim, dtype = np.float64)
        self.min_vals = np.full(dim, np.inf, dtype = np.float64)
        self.max_vals = np.full(dim, -np.inf, dtype = np.float64)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype = np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
        np.minimum(self.min_vals, x, out = self.min_vals)
        np.maximum(self.max_vals, x, out = self.max_vals)

    def to_dict(self, names: List[str]) -> Dict:
        std = np.sqrt(self.m2 / self.n) if self.n > 0 else np.zeros_like(self.mean)
        return {
            'names': list(names),
            'mean': self.mean.tolist(),
            'std': std.tolist(),
            'min': self.min_vals.tolist(),
            'max': self.max_vals.tolist(),
        }

# 14 required joint dimensions in canonical order
REQUIRED_DIM_NAMES = []
for _i in range(1, 7):
    REQUIRED_DIM_NAMES.append(f'left_arm_joint_{_i}_rad')
REQUIRED_DIM_NAMES.append('left_gripper_open')
for _i in range(1, 7):
    REQUIRED_DIM_NAMES.append(f'right_arm_joint_{_i}_rad')
REQUIRED_DIM_NAMES.append('right_gripper_open')

# 12 required EEF dimensions in canonical order (same for state and action)
EEF_DIM_NAMES = [
    "left_eef_pos_x", "left_eef_pos_y", "left_eef_pos_z",
    "left_eef_ori_x", "left_eef_ori_y", "left_eef_ori_z",
    "right_eef_pos_x", "right_eef_pos_y", "right_eef_pos_z",
    "right_eef_ori_x", "right_eef_ori_y", "right_eef_ori_z",
]


def extract_robot_type_from_repo_id(repo_id: str) -> str:
    """Extract robot type from repo_id by taking first 2 underscore/hyphen-delimited parts.

    Args:
        repo_id: e.g. "RoboCOIN/Split_aloha_plate_storage"

    Returns:
        e.g. "Split_aloha"
    """
    import re
    if '/' in repo_id:
        repo_name = repo_id.split('/', 1)[1]
    else:
        repo_name = repo_id
    parts = re.split(r'[_-]', repo_name)
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    return parts[0] if parts else repo_name


def get_camera_names_for_repo(features: Dict, robot_type: str) -> List[str]:
    """Get ordered camera names: [front/high, left_wrist, right_wrist].

    Args:
        features: Features dictionary from info.json
        robot_type: Robot type extracted from repo_id

    Returns:
        List of 3 camera names in canonical order
    """
    all_cameras = []
    for key, val in features.items():
        if isinstance(val, dict) and val.get('dtype') == 'video':
            parts = key.split('.')
            if len(parts) >= 3:
                cam_name = ".".join(parts[2:])
                if "fisheye" not in cam_name:
                    all_cameras.append(cam_name)

    if all(cam in all_cameras for cam in DEFAULT_CAMERA_ORDER):
        return DEFAULT_CAMERA_ORDER.copy()

    front_camera = None
    left_wrist_camera = None
    right_wrist_camera = None

    for cam in all_cameras:
        cam_lower = cam.lower()
        if front_camera is None and ("high" in cam_lower or "front" in cam_lower):
            front_camera = cam
        elif left_wrist_camera is None and "left" in cam_lower and "wrist" in cam_lower:
            left_wrist_camera = cam
        elif right_wrist_camera is None and "right" in cam_lower and "wrist" in cam_lower:
            right_wrist_camera = cam

    if front_camera is None or left_wrist_camera is None or right_wrist_camera is None:
        raise ValueError(
            f"Could not find required cameras for robot_type '{robot_type}'. "
            f"Available cameras: {all_cameras}. "
            f"Found: front={front_camera}, left_wrist={left_wrist_camera}, right_wrist={right_wrist_camera}"
        )
    return [front_camera, left_wrist_camera, right_wrist_camera]


def should_include_repo(repo_id: str, info_data: Dict) -> Tuple[bool, str, int]:
    """Determine if a repo should be included based on robot_type and state_dim.

    Returns:
        (should_include, robot_type, state_dim)
    """
    robot_type = extract_robot_type_from_repo_id(repo_id)
    if robot_type not in ALLOWED_ROBOT_TYPES:
        return False, robot_type, 0
    state_dim = info_data['features']['observation.state']['shape'][0]
    if robot_type == "R1_Lite" and state_dim != 14:
        return False, robot_type, state_dim
    return True, robot_type, state_dim


def run_with_rate_limit_retry(
    cmd: List[str],
    sleep_seconds: int = 300,
    max_retries: int = None,
) -> None:
    """Run a command, retrying on Hugging Face Hub rate limit errors."""
    attempt = 0
    while True:
        attempt += 1
        p = subprocess.run(cmd, capture_output = True, text = True)
        out = (p.stdout or "") + "\n" + (p.stderr or "")

        if any(m in out for m in RATE_LIMIT_MARKERS):
            Metrics.counter("robocoin", "api_rate_limit_retries").inc()
            logger.warning(
                f"[rate-limit] Sleeping {sleep_seconds}s then retrying. Attempt={attempt}"
            )
            if max_retries is not None and attempt > max_retries:
                raise RuntimeError(f"Exceeded max_retries={max_retries} for command: {cmd}")
            time.sleep(sleep_seconds)
            continue

        if p.returncode == 0:
            if p.stdout:
                logger.info(p.stdout.rstrip())
            if p.stderr:
                logger.warning(p.stderr.rstrip())
            return

        logger.error(out)
        raise subprocess.CalledProcessError(p.returncode, cmd, output = p.stdout, stderr = p.stderr)


VAL_ONLY_REPOS = {
    "Split_aloha_plate_storage",
    "Cobot_Magic_cut_banana",
    "R1_Lite_tableware_cleaning",
}


def _get_episode_indices_for_split(total_episodes: int, split: str, repo_id: str = "") -> List[int]:
    """Deterministically compute episode indices for train or val.

    Uses a fixed seed so train/val are always disjoint regardless of which
    Beam worker computes them. Repos in VAL_ONLY_REPOS get all episodes
    assigned to val and none to train.

    Args:
        total_episodes: Total number of episodes in the repo.
        split: 'train' or 'val'.
        repo_id: Full repo ID (e.g. "RoboCOIN/Split_aloha_plate_storage").

    Returns:
        Sorted list of episode indices for this split.
    """
    repo_suffix = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id
    if repo_suffix in VAL_ONLY_REPOS:
        if split == 'train':
            return []
        else:
            return list(range(total_episodes))

    val_count = max(1, int(total_episodes * VAL_SPLIT_RATIO))
    train_count = total_episodes - val_count
    all_indices = list(range(total_episodes))
    rng = random.Random(86)
    rng.shuffle(all_indices)
    if split == 'train':
        return sorted(all_indices[ : train_count])
    else:
        return sorted(all_indices[train_count : ])


def _process_repo_for_split(element, split):
    """Process a single repo for a given split.

    This is a module-level function so Beam can serialize it across workers.
    Each invocation downloads (with locking), loads metadata, iterates over
    the episodes assigned to the requested split, and yields (key, example)
    tuples.

    Args:
        element: (repo_index, repo_id) tuple from beam.Create
        split: 'train' or 'val'

    Yields:
        (key, example_dict) tuples for tfds
    """
    # Resolve config at call time so Dataflow workers use their own env vars
    download_root = os.environ.get(
        "ROBOCOIN_DOWNLOAD_ROOT",
        "/data/group_data/rl/saksham3/robocoin_bimanual/"
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_index, repo_id = element
    logger.info(f"[{split}] Processing {repo_id} (index {repo_index})...")

    dataset_path = os.path.join(download_root, repo_id)
    meta_path = os.path.join(dataset_path, "meta")
    repo_suffix = repo_id[len(PREFIX) : ]

    # Download with file lock so parallel workers don't duplicate downloads
    os.makedirs(download_root, exist_ok = True)
    lock_path = os.path.join(download_root, f".lock_{repo_suffix.replace('/', '_')}")
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            cmd = [
                "robocoin-download", "--hub", "huggingface",
                "--target-dir", download_root, "--ds_lists", repo_suffix
            ]
            run_with_rate_limit_retry(cmd, sleep_seconds = 90)
        except Exception as e:
            logger.info(f"[{split}] Failed to download {repo_id}: {e}")
            return
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    progress_marker = os.path.join(dataset_path, f"{split}.txt")
    open(progress_marker, 'w').close()

    try:
        # --- Metadata ---
        with open(os.path.join(meta_path, "info.json"), 'r') as f:
            info_data = json.load(f)

        robot_type = extract_robot_type_from_repo_id(repo_id)
        fps = float(info_data['fps'])
        features = info_data['features']

        # Cameras
        all_camera_shapes = {}
        for key, val in features.items():
            if isinstance(val, dict) and val.get('dtype') == 'video':
                parts = key.split('.')
                cam_name = "".join(parts[2 : ])
                if "fisheye" in cam_name:
                    continue
                vi = val['info']
                all_camera_shapes[cam_name] = np.array(
                    [vi['video.height'], vi['video.width'], vi['video.channels']], dtype = np.int32
                )

        camera_names = get_camera_names_for_repo(features, robot_type)
        camera_shapes = [all_camera_shapes[cam] for cam in camera_names]
        num_cameras = len(camera_names)

        # State / action dimension indices
        state_feat_names = features['observation.state']['names']
        action_feat_names = features['action']['names'] if 'action' in features else []
        state_indices = np.array([state_feat_names.index(n) for n in REQUIRED_DIM_NAMES], dtype = np.int32)
        action_indices = np.array([action_feat_names.index(n) for n in REQUIRED_DIM_NAMES], dtype = np.int32)

        # EEF dimension indices
        assert 'eef_sim_pose_state' in features, f"eef_sim_pose_state not in features for {repo_id}"
        assert 'eef_sim_pose_action' in features, f"eef_sim_pose_action not in features for {repo_id}"
        eef_state_feat_names = features['eef_sim_pose_state']['names']
        eef_action_feat_names = features['eef_sim_pose_action']['names']
        eef_state_indices = np.array([eef_state_feat_names.index(n) for n in EEF_DIM_NAMES], dtype = np.int32)
        eef_action_indices = np.array([eef_action_feat_names.index(n) for n in EEF_DIM_NAMES], dtype = np.int32)

        # Subtask annotations
        subtask_ann_path = os.path.join(dataset_path, "annotations", "subtask_annotations.jsonl")
        subtasks_list = []
        subtask_index_to_text = {}
        null_subtask_index = None
        with open(subtask_ann_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    subtasks_list.append(data['subtask'])
                    subtask_index_to_text[data['subtask_index']] = data['subtask']
                    if data['subtask'] == 'null':
                        null_subtask_index = data['subtask_index']
        if null_subtask_index is None:
            raise ValueError(f"No null subtask found in {subtask_ann_path}")

        # Scene annotations
        scene_annotations = []
        with open(os.path.join(dataset_path, "annotations", "scene_annotations.jsonl"), 'r') as f:
            for line in f:
                if line.strip():
                    scene_annotations.append(json.loads(line))

        # Tasks
        tasks_list = []
        with open(os.path.join(meta_path, "tasks.jsonl"), 'r') as f:
            count = 0
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    assert data['task_index'] == count
                    tasks_list.append(data)
                    count += 1

        # Episodes metadata
        episodes_meta = []
        with open(os.path.join(meta_path, "episodes.jsonl"), 'r') as f:
            for line in f:
                if line.strip():
                    episodes_meta.append(json.loads(line))

        try:
            ds = LeRobotDataset(root = download_root + repo_id, repo_id = repo_id, video_backend = "pyav")
        except Exception:
            stats_path = os.path.join(meta_path, "episodes_stats.jsonl")
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    content = f.read()
                content = content.replace('NaN', '0')
                with open(stats_path, 'w') as f:
                    f.write(content)
                logger.info(f"[{split}] Fixed NaN values in {stats_path}, retrying...")
                ds = LeRobotDataset(root = download_root + repo_id, repo_id = repo_id, video_backend = "pyav")
            else:
                raise

        # --- Episode iteration ---
        total_episodes = len(episodes_meta)
        episode_indices = _get_episode_indices_for_split(total_episodes, split, repo_id)
        logger.info(f"[{split}] {repo_id}: processing {len(episode_indices)}/{total_episodes} episodes")

        # Norm stats trackers (train split only)
        compute_norm_stats = (split == 'train')
        if compute_norm_stats:
            norm_state = OnlineStats(MAX_STATE_DIM)
            norm_action = OnlineStats(MAX_ACTION_DIM)
            norm_action_diff = OnlineStats(MAX_ACTION_DIM)
            norm_eef_state = OnlineStats(12)
            norm_eef_action = OnlineStats(12)
            norm_total_steps = 0
            norm_total_episodes = 0

        test_mode = os.environ.get("TEST_MODE", "0") == "1"

        for ep_count, ep_idx in enumerate(episode_indices):
            if ep_count >= 50 and test_mode:
                break

            ep_info = episodes_meta[ep_idx]
            cumulative_idx = sum(episodes_meta[j]['length'] for j in range(ep_idx))
            ep_len = ep_info['length']
            ep_id = ep_info['episode_index']

            # Episode metadata
            scene_desc = scene_annotations[ep_idx]['scene']
            first_frame = ds[cumulative_idx]
            assert ep_id == first_frame['episode_index']
            task_desc = tasks_list[int(first_frame['task_index'])]['task']

            episode_metadata = {
                'repo_id': repo_id,
                'robot_type': robot_type,
                'fps': fps,
                'camera_names': camera_names,
                'camera_shapes': camera_shapes,
                'num_cameras': num_cameras,
                'state_feature_names': state_feat_names,
                'action_feature_names': action_feat_names,
                'subtasks': subtasks_list,
                'scene_description': scene_desc,
                'task_description': task_desc,
            }

            # --- Step-level processing (direct indexing, no DataLoader) ---
            steps_list = []
            all_subtask_annotations = []
            prev_action = None

            for i in range(ep_len):
                global_idx = cumulative_idx + i
                item = ds[global_idx]

                step = {}

                # Images
                for cam_idx in range(MAX_CAMERAS):
                    cam_key = f'observation/image/cam_{cam_idx}'
                    if cam_idx < num_cameras:
                        img_data = item[f"observation.images.{camera_names[cam_idx]}"].numpy()
                        if img_data.shape[0] == 3 and img_data.ndim == 3:
                            img_data = np.transpose(img_data, (1, 2, 0))
                        if img_data.dtype == np.uint8:
                            img_data = img_data.astype(np.float32) / 255.0
                        elif img_data.dtype == np.float64:
                            img_data = img_data.astype(np.float32)
                        img_data = tf.image.resize(img_data, (224, 224)).numpy()
                        img_data = (img_data * 255.0).astype(np.uint8)
                        step[cam_key] = tf.image.encode_jpeg(img_data).numpy()
                    else:
                        step[cam_key] = b''

                # State
                state_val = item['observation.state'].numpy().astype(np.float32)[state_indices]
                if state_val.shape[0] != MAX_STATE_DIM:
                    raise ValueError(
                        f"State dim {state_val.shape[0]} != {MAX_STATE_DIM} in {repo_id}"
                    )
                step['observation/state'] = state_val

                # Action
                act_val = item['action'].numpy().astype(np.float32)[action_indices]
                if act_val.shape[0] != MAX_ACTION_DIM:
                    raise ValueError(
                        f"Action dim {act_val.shape[0]} != {MAX_ACTION_DIM} in {repo_id}"
                    )
                step['action'] = act_val

                # Action diff
                step['action_diff'] = np.zeros(MAX_ACTION_DIM, dtype = np.float32)
                if prev_action is not None:
                    steps_list[-1]['action_diff'] = act_val - prev_action
                prev_action = act_val.copy()

                # Subtask annotations
                subtask_ann = item['subtask_annotation'].numpy().astype(np.int32)
                is_null = np.array(
                    [subtask_ann[pos] == null_subtask_index for pos in range(5)], dtype = np.bool_
                )
                non_null_indices = np.where(~is_null)[0]
                null_indices = np.where(is_null)[0]
                if len(non_null_indices) > 0 and len(null_indices) > 0:
                    assert np.max(non_null_indices) < np.min(null_indices), (
                        f"Subtask annotation not in [non_null]+[null] form at step {i} "
                        f"of episode {ep_idx} in {repo_id}"
                    )
                first_null_idx = np.min(null_indices) if is_null.any() else 5
                all_subtask_annotations.append(subtask_ann)

                for pos in range(5):
                    step[f'subtask_{pos + 1}'] = subtask_index_to_text.get(
                        int(subtask_ann[pos]), ''
                    )
                step['subtask_mask'] = np.array(
                    [int(subtask_ann[pos]) != null_subtask_index for pos in range(5)],
                    dtype = np.bool_
                )
                step['first_null_index'] = first_null_idx
                step['scene_annotation'] = item['scene_annotation'].numpy().astype(np.int32).item()

                # RLDS fields
                step['is_first'] = (i == 0)
                step['is_terminal'] = (i == ep_len - 1)
                step['frame_index'] = int(item['frame_index'])
                task_idx = int(item['task_index'])
                step['task'] = tasks_list[task_idx]['task']
                assert step['task'] == task_desc
                step['episode_index'] = int(item['episode_index'])
                step['index'] = int(item['index'])

                step['eef_sim_pose_state'] = item['eef_sim_pose_state'].numpy().astype(np.float32)[eef_state_indices]
                step['eef_sim_pose_action'] = item['eef_sim_pose_action'].numpy().astype(np.float32)[eef_action_indices]

                step['repo_index'] = repo_index

                # Update norm stats trackers
                if compute_norm_stats:
                    norm_state.update(state_val)
                    norm_action.update(act_val)
                    norm_eef_state.update(step['eef_sim_pose_state'])
                    norm_eef_action.update(step['eef_sim_pose_action'])
                    # action_diff for the previous step (just computed on steps_list[-1])
                    if i > 0:
                        norm_action_diff.update(steps_list[-1]['action_diff'])
                    norm_total_steps += 1

                steps_list.append(step)

            # --- Subtask timing features (set-difference logic) ---
            steps_to_subtask_end_list = [np.zeros(5, dtype = np.int32) for _ in range(ep_len)]
            subtask_len_list = [np.zeros(5, dtype = np.int32) for _ in range(ep_len)]
            subtask_is_first_list = [np.zeros(5, dtype = np.bool_) for _ in range(ep_len)]
            subtask_is_last_list = [np.zeros(5, dtype = np.bool_) for _ in range(ep_len)]

            subtask_start_steps = {}
            current_subtasks = set()

            def fill_subtask_interval(subtask_id, start_step, end_step):
                interval_len = end_step - start_step + 1
                for t in range(start_step, end_step + 1):
                    ann = all_subtask_annotations[t]
                    for pos in range(5):
                        if int(ann[pos]) == subtask_id:
                            steps_to_subtask_end_list[t][pos] = end_step - t
                            subtask_len_list[t][pos] = interval_len
                            subtask_is_first_list[t][pos] = (t == start_step)
                            subtask_is_last_list[t][pos] = (t == end_step)
                            break

            for step_idx, ann in enumerate(all_subtask_annotations):
                valid_subtasks = set(int(s) for s in ann if s != null_subtask_index)
                for sid in (current_subtasks - valid_subtasks):
                    fill_subtask_interval(sid, subtask_start_steps.pop(sid), step_idx - 1)
                for sid in (valid_subtasks - current_subtasks):
                    subtask_start_steps[sid] = step_idx
                current_subtasks = valid_subtasks

            for sid in current_subtasks:
                fill_subtask_interval(sid, subtask_start_steps.pop(sid), ep_len - 1)

            for i, step in enumerate(steps_list):
                step['steps_to_subtask_end'] = steps_to_subtask_end_list[i]
                step['subtask_len'] = subtask_len_list[i]
                step['subtask_is_first'] = subtask_is_first_list[i]
                step['subtask_is_last'] = subtask_is_last_list[i]

            yield f"{repo_id}_{ep_idx}", {
                'steps': steps_list,
                'episode_metadata': episode_metadata,
            }
            Metrics.counter("robocoin", f"episodes_{split}").inc()

            if compute_norm_stats:
                norm_total_episodes += 1

        Metrics.counter("robocoin", f"repos_completed_{split}").inc()

        # Write per-repo norm stats to GCS (train split only)
        if compute_norm_stats and norm_total_steps > 0:
            norm_stats_output = {
                'repo_id': repo_id,
                'num_steps': norm_total_steps,
                'num_episodes': norm_total_episodes,
                'observation.state': norm_state.to_dict(REQUIRED_DIM_NAMES),
                'action': norm_action.to_dict(REQUIRED_DIM_NAMES),
                'action_diff': norm_action_diff.to_dict(REQUIRED_DIM_NAMES),
                'eef_sim_pose_state': norm_eef_state.to_dict(EEF_DIM_NAMES),
                'eef_sim_pose_action': norm_eef_action.to_dict(EEF_DIM_NAMES),
            }
            norm_stats_gcs_root = os.environ.get(
                "NORM_STATS_GCS_ROOT",
                "gs://saksham-euw4/robocoin/norm_stats/"
            )
            norm_stats_dir = os.path.join(norm_stats_gcs_root, repo_suffix)
            tf.io.gfile.makedirs(norm_stats_dir)
            norm_stats_path = os.path.join(norm_stats_dir, "norm_stats.json")
            with tf.io.gfile.GFile(norm_stats_path, 'w') as f:
                json.dump(norm_stats_output, f, indent = 2)
            logger.info(f"[{split}] Wrote norm stats for {repo_id} → {norm_stats_path}")

        if os.path.exists(progress_marker):
            os.remove(progress_marker)
        other_split = 'val' if split == 'train' else 'train'
        other_marker = os.path.join(dataset_path, f"{other_split}.txt")
        if not os.path.exists(other_marker) and os.path.exists(dataset_path):
            logger.info(f"[{split}] Deleting {dataset_path}...")
            subprocess.run(['rm', '-rf', dataset_path])

        logger.info(f"[{split}] Finished {repo_id} (index {repo_index})")

    except Exception as e:
        if os.path.exists(progress_marker):
            os.remove(progress_marker)
        logger.info(f"[{split}] Error processing {repo_id}: {e}")
        import traceback
        traceback.print_exc()


class RobocoinBimanual(tfds.core.BeamBasedBuilder):
    """DatasetBuilder for RoboCOIN bimanual datasets using Apache Beam."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release covering all RoboCOIN bimanual datasets.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (identical schema to the sequential builder)."""
        obs_features = {}
        for i in range(MAX_CAMERAS):
            obs_features[f'observation/image/cam_{i}'] = tfds.features.Tensor(
                shape = (), dtype = tf.string
            )
        obs_features['observation/state'] = tfds.features.Tensor(
            shape = (MAX_STATE_DIM,), dtype = np.float32
        )

        return self.dataset_info_from_configs(
            features = tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    **obs_features,
                    'action': tfds.features.Tensor(shape = (MAX_ACTION_DIM,), dtype = np.float32),
                    'action_diff': tfds.features.Tensor(shape = (MAX_ACTION_DIM,), dtype = np.float32),
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
                    'repo_index': tfds.features.Scalar(dtype = np.int32),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'repo_id': tfds.features.Text(doc = 'Hugging Face Repository ID.'),
                    'robot_type': tfds.features.Text(),
                    'fps': tfds.features.Scalar(dtype = np.float32),
                    'camera_names': tfds.features.Sequence(tfds.features.Text()),
                    'camera_shapes': tfds.features.Sequence(
                        tfds.features.Tensor(shape = (3,), dtype = np.int32)
                    ),
                    'num_cameras': tfds.features.Scalar(dtype = np.int64),
                    'state_feature_names': tfds.features.Sequence(tfds.features.Text()),
                    'action_feature_names': tfds.features.Sequence(tfds.features.Text()),
                    'subtasks': tfds.features.Sequence(tfds.features.Text()),
                    'scene_description': tfds.features.Text(),
                    'task_description': tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Discover and filter repos, then return SplitGenerators for Beam."""
        from huggingface_hub import hf_hub_download

        logger.info("Starting dataset build...")

        test_mode = os.environ.get("TEST_MODE", "0") == "1"
        test_num_repos = int(os.environ.get("TEST_NUM_REPOS", "8"))

        # Always discover and filter all valid repos
        api = HfApi()
        logger.info(f"Listing datasets with prefix '{PREFIX}' from Hugging Face...")
        infos = api.list_datasets(search = PREFIX)
        all_repo_ids = sorted([d.id for d in infos if d.id.startswith(PREFIX)])
        logger.info(f"Found {len(all_repo_ids)} total repos. Filtering...")

        repo_ids = []
        repo_total_episodes: Dict[str, int] = {}
        repo_total_frames: Dict[str, int] = {}
        for repo_id in all_repo_ids:
            try:
                info_path = None
                retry_delay_seconds = 5
                attempt = 1
                while True:
                    try:
                        info_path = hf_hub_download(
                            repo_id = repo_id,
                            filename = "meta/info.json",
                            repo_type = "dataset"
                        )
                        if info_path and os.path.exists(info_path):
                            break
                        else:
                            time.sleep(retry_delay_seconds)
                            attempt += 1
                    except Exception as download_err:
                        logger.info(f"  Retry {attempt} for {repo_id}: {download_err}")
                        time.sleep(retry_delay_seconds)
                        attempt += 1

                with open(info_path, 'r') as f:
                    info_data = json.load(f)

                should_include, robot_type, state_dim = should_include_repo(repo_id, info_data)
                if should_include:
                    repo_ids.append(repo_id)
                    total_ep = info_data.get('total_episodes', 0)
                    total_fr = info_data.get('total_frames', 0)
                    if total_ep > 0:
                        repo_total_episodes[repo_id] = total_ep
                    repo_total_frames[repo_id] = total_fr
                    logger.info(f"  Including {repo_id} (robot_type={robot_type}, state_dim={state_dim}, episodes={total_ep}, frames={total_fr})")
                else:
                    logger.info(f"  Skipping {repo_id} (robot_type={robot_type}, state_dim={state_dim})")
            except Exception as e:
                logger.info(f"  Skipping {repo_id}: {e}")

        logger.info(f"\nFiltered to {len(repo_ids)} repos")

        if test_mode:
            n = min(test_num_repos, len(repo_ids))
            repo_ids = sorted(repo_ids, key = lambda r: repo_total_frames.get(r, 0))[ : n]
            logger.info(f"[TEST MODE] Selected {n} smallest repos by total_frames: {repo_ids}")

        random.seed(86)
        random.shuffle(repo_ids)
        os.makedirs(DOWNLOAD_ROOT, exist_ok = True)

        # Handle GCS paths (gs://...).
        build_metadata = {
            'repo_ids': repo_ids,
            'total_repos': len(repo_ids),
            'val_split_ratio': VAL_SPLIT_RATIO,
            'allowed_robot_types': ALLOWED_ROBOT_TYPES,
            'test_mode': test_mode,
        }

        # Compute deterministic train/val split per repo from cached data.
        split_map: Dict[str, Dict[str, list]] = {}
        for repo_id in repo_ids:
            total_episodes = repo_total_episodes.get(repo_id, 0)
            if total_episodes > 0:
                split_map[repo_id] = {
                    'train': _get_episode_indices_for_split(total_episodes, 'train', repo_id),
                    'val': _get_episode_indices_for_split(total_episodes, 'val', repo_id),
                }

        build_metadata['split_map'] = split_map

        # Write to data_dir (works for local and gs:// paths)
        tf.io.gfile.makedirs(self.data_dir)
        metadata_path = os.path.join(self.data_dir, "build_metadata.json")
        with tf.io.gfile.GFile(metadata_path, 'w') as f:
            json.dump(build_metadata, f, indent = 2)
        logger.info(f"Wrote build metadata ({len(split_map)} repos) → {metadata_path}")

        # Only pass repos that actually have episodes for each split, so
        # workers never produce 0 examples for a split.
        train_repo_ids = [
            r for r in repo_ids if split_map.get(r, {}).get('train', [])
        ]
        val_repo_ids = [
            r for r in repo_ids if split_map.get(r, {}).get('val', [])
        ]
        logger.info(f"Split assignment: {len(train_repo_ids)} repos → train, "
              f"{len(val_repo_ids)} repos → val")

        return [
            tfds.core.SplitGenerator(
                name = 'train',
                gen_kwargs = {'repo_ids': train_repo_ids, 'split': 'train'},
            ),
            tfds.core.SplitGenerator(
                name = 'val',
                gen_kwargs = {'repo_ids': val_repo_ids, 'split': 'val'},
            ),
        ]

    def _build_pcollection(self, pipeline, repo_ids, split):
        """Build a Beam PCollection of (key, example) tuples.

        Each repo is processed as an independent Beam element, enabling
        parallel downloads and episode processing across workers.
        """
        beam = tfds.core.lazy_imports.apache_beam
        return (
            pipeline
            | f'CreateRepos_{split}' >> beam.Create(list(enumerate(repo_ids)))
            | f'Reshuffle_{split}' >> beam.Reshuffle()
            | f'ProcessRepos_{split}' >> beam.FlatMap(
                _process_repo_for_split, split = split
            )
        )

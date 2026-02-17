"""
RoboCOIN Dataset Builder for RLDS/TensorFlow Datasets.
Merges multiple Hugging Face RoboCOIN datasets into a single RLDS dataset.
"""

from typing import Iterator, Tuple, Any, List, Dict
import os
import json
import fcntl
import logging
import random
import subprocess
import time
import traceback
import numpy as np

logger = logging.getLogger(__name__)
from torch.utils.data import DataLoader, Subset

import tensorflow as tf
import tensorflow_datasets as tfds
from huggingface_hub import HfApi

# Ensure HF token is set
assert os.environ.get("HF_TOKEN"), "HF_TOKEN environment variable must be set"


# --- Configuration ---
# Directory where data will be temporarily downloaded
DOWNLOAD_ROOT = "/data/group_data/rl/saksham3/robocoin/"
PREFIX = "RoboCOIN/"
MAX_CAMERAS = 3
MAX_STATE_DIM = 14
MAX_ACTION_DIM = 14  # Same as state - filtered to required 14 dimensions

# Allowed robot types (extracted from first 2 parts of repo_id)
# R1_Lite is only allowed if state_dim == 14
ALLOWED_ROBOT_TYPES = ["Split_aloha", "Cobot_Magic", "R1_Lite"]

# Default required cameras order: front/high, left wrist, right wrist
DEFAULT_CAMERA_ORDER = ["cam_high_rgb", "cam_left_wrist_rgb", "cam_right_wrist_rgb"]

WORKER_ID = int(os.environ.get("WORKER_ID", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "1"))
MIN_STEPS_PER_WORKER = 5000

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
    """Extract robot type from repo_id by splitting on _ or - and taking first 2 elements.
    
    Args:
        repo_id: Repository ID, e.g., "RoboCOIN/Split_aloha_plate_storage"
    
    Returns:
        Robot type as a string with first 2 parts joined by underscore, e.g., "Split_aloha"
    """
    import re
    # Remove the prefix (everything before and including the first /)
    if '/' in repo_id:
        repo_name = repo_id.split('/', 1)[1]
    else:
        repo_name = repo_id
    
    # Split on both _ and -
    parts = re.split(r'[_-]', repo_name)
    
    # Take first 2 elements and join with underscore
    if len(parts) >= 2:
        robot_type = '_'.join(parts[:2])
    else:
        robot_type = parts[0] if parts else repo_name
    
    return robot_type


def get_camera_names_for_repo(features: Dict, robot_type: str) -> List[str]:
    """Get ordered camera names for a repo based on robot type.
    
    For Cobot_Magic datasets, cameras might have different names but should follow:
    front/high camera, left wrist camera, right wrist camera.
    
    Args:
        features: Features dictionary from info.json
        robot_type: Robot type extracted from repo_id
    
    Returns:
        List of camera names in order: [front/high, left_wrist, right_wrist]
    """
    # Extract all camera names from features
    all_cameras = []
    for key, val in features.items():
        if isinstance(val, dict) and val.get('dtype') == 'video':
            # Key format is "observation.images.<name>"
            parts = key.split('.')
            if len(parts) >= 3:
                cam_name = ".".join(parts[2:])
                if "fisheye" not in cam_name:
                    all_cameras.append(cam_name)
    
    # Try default camera order first
    if all(cam in all_cameras for cam in DEFAULT_CAMERA_ORDER):
        return DEFAULT_CAMERA_ORDER.copy()
    
    # For Cobot_Magic or other robots with different camera names,
    # find cameras matching patterns: front/high, left wrist, right wrist
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
    
    Args:
        repo_id: Repository ID
        info_data: Parsed info.json data
    
    Returns:
        Tuple of (should_include, robot_type, state_dim)
    """
    robot_type = extract_robot_type_from_repo_id(repo_id)
    
    if robot_type not in ALLOWED_ROBOT_TYPES:
        return False, robot_type, 0
    
    # Get state dimension from features
    state_dim = info_data['features']['observation.state']['shape'][0]
    
    # R1_Lite is only allowed if state_dim == 14
    if robot_type == "R1_Lite" and state_dim != 14:
        return False, robot_type, state_dim
    
    return True, robot_type, state_dim

# Train/val split ratio
VAL_SPLIT_RATIO = 0.05

VAL_ONLY_REPOS = {
    "Split_aloha_plate_storage",
    "Cobot_Magic_cut_banana",
    "R1_Lite_tableware_cleaning",
}


def _get_episode_indices_for_split(total_episodes: int, split: str, repo_id: str = "") -> List[int]:
    """Deterministically compute episode indices for train or val.

    Uses a fixed seed so train/val are always disjoint regardless of call order.
    Repos in VAL_ONLY_REPOS get all episodes assigned to val and none to train.

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


# Rate limit handling constants
RATE_LIMIT_MARKERS = (
    "429 Too Many Requests",
)

def run_with_rate_limit_retry(
    cmd: List[str],
    sleep_seconds: int = 300,
    max_retries: int = None,
) -> None:
    """
    Run a command, retrying on Hugging Face Hub rate limit errors.
    """
    attempt = 0
    while True:
        attempt += 1
        # Capture output to check for rate limits
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stdout or "") + "\n" + (p.stderr or "")

        if any(m in out for m in RATE_LIMIT_MARKERS):
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
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)

class Robocoin(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for RoboCOIN datasets."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release covering all RoboCOIN datasets.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata."""
        
        # Define observation dictionary dynamically for max cameras
        obs_features = {}
        
        # 1. Images (up to MAX_CAMERAS)
        for i in range(MAX_CAMERAS):
            obs_features[f'observation/image/cam_{i}'] = tfds.features.Tensor(shape=(), dtype=tf.string)

        # 2. State
        # Fixed shape with zero padding for states shorter than MAX_STATE_DIM
        obs_features['observation/state'] = tfds.features.Tensor(shape=(MAX_STATE_DIM,), dtype=np.float32)
        
        # 3. Action
        # Fixed shape with zero padding for actions shorter than MAX_ACTION_DIM
        action_feature = tfds.features.Tensor(shape=(MAX_ACTION_DIM,), dtype=np.float32)
        
        # 4. Action Diff (a[t+1] - a[t], zeros at terminal state)
        action_diff_feature = tfds.features.Tensor(shape=(MAX_ACTION_DIM,), dtype=np.float32)

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    **obs_features,
                    'action': action_feature,
                    'state_diff': tfds.features.Tensor(shape=(MAX_STATE_DIM,), dtype=np.float32),
                    'action_diff': action_diff_feature,
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                    'frame_index': tfds.features.Scalar(dtype=np.int64),
                    'task': tfds.features.Text(),
                    'episode_index': tfds.features.Scalar(dtype=np.int64),
                    'index': tfds.features.Scalar(dtype=np.int64),
                    'subtask_1': tfds.features.Text(),
                    'subtask_2': tfds.features.Text(),
                    'subtask_3': tfds.features.Text(),
                    'subtask_4': tfds.features.Text(),
                    'subtask_5': tfds.features.Text(),
                    'subtask_mask': tfds.features.Tensor(shape=(5,), dtype=np.bool_),
                    'steps_to_subtask_end': tfds.features.Tensor(shape=(5,), dtype=np.int32),
                    'subtask_len': tfds.features.Tensor(shape=(5,), dtype=np.int32),
                    'subtask_is_first': tfds.features.Tensor(shape=(5,), dtype=np.bool_),
                    'subtask_is_last': tfds.features.Tensor(shape=(5,), dtype=np.bool_),
                    'first_null_index': tfds.features.Scalar(dtype=np.int32),
                    'scene_annotation': tfds.features.Scalar(dtype=np.int32),
                    'eef_sim_pose_state': tfds.features.Tensor(shape=(12,), dtype=np.float32),
                    'eef_sim_pose_action': tfds.features.Tensor(shape=(12,), dtype=np.float32),
                    'eef_sim_pose_action_diff': tfds.features.Tensor(shape=(12,), dtype=np.float32),
                    'repo_index': tfds.features.Scalar(dtype=np.int32),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'repo_id': tfds.features.Text(doc='Hugging Face Repository ID.'),
                    
                    # Metadata req 1 & 2
                    'robot_type': tfds.features.Text(),
                    'fps': tfds.features.Scalar(dtype=np.float32),
                    
                    # Metadata req 3
                    'camera_names': tfds.features.Sequence(tfds.features.Text()),
                    # Height, Width, Channels
                    'camera_shapes': tfds.features.Sequence(tfds.features.Tensor(shape=(3,), dtype=np.int32)),
                    'num_cameras': tfds.features.Scalar(dtype=np.int64),
                    
                    # Metadata req 4
                    'state_feature_names': tfds.features.Sequence(tfds.features.Text()),
                    'action_feature_names': tfds.features.Sequence(tfds.features.Text()),
                    
                    # Metadata req 5
                    'subtasks': tfds.features.Sequence(tfds.features.Text()),
                    
                    # Episode specific metadata
                    # 'scene_description': tfds.features.Text(),
                    'task_description': tfds.features.Text(),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        from huggingface_hub import hf_hub_download
        
        logger.info("Starting dataset build...")
        
        test_mode = os.environ.get("TEST_MODE", "0") == "1"
        repo_ids_file = os.environ.get("REPO_IDS_FILE")

        def _download_hf_meta(repo_id, filename):
            """Download a meta file from HF hub with retries, return local path."""
            retry_delay_seconds = 5
            attempt = 1
            while True:
                try:
                    local_path = hf_hub_download(
                        repo_id = repo_id,
                        filename = filename,
                        repo_type = "dataset"
                    )
                    if local_path and os.path.exists(local_path):
                        return local_path
                    time.sleep(retry_delay_seconds)
                    attempt += 1
                except Exception as download_err:
                    logger.info(
                        f"  Retry {attempt} for {repo_id} "
                        f"{filename} failed: {download_err}"
                    )
                    time.sleep(retry_delay_seconds)
                    attempt += 1

        def _load_episode_lengths(repo_id):
            """Download meta/episodes.jsonl and return list of episode lengths."""
            ep_path = _download_hf_meta(repo_id, "meta/episodes.jsonl")
            lengths = []
            with open(ep_path, 'r') as f:
                for line in f:
                    if line.strip():
                        lengths.append(json.loads(line)['length'])
            return lengths

        repo_episode_lengths: Dict[str, List[int]] = {}

        if repo_ids_file:
            with open(repo_ids_file, 'r') as f:
                repo_ids = [line.strip() for line in f if line.strip()]
            repo_ids = [r if r.startswith(PREFIX) else PREFIX + r for r in repo_ids]
            logger.info(f"[REPO_IDS_FILE] Loaded {len(repo_ids)} repos from {repo_ids_file}")
            for repo_id in repo_ids:
                try:
                    repo_episode_lengths[repo_id] = _load_episode_lengths(repo_id)
                except Exception as e:
                    logger.info(f"  Warning: could not get episode lengths for {repo_id}: {e}")
        elif test_mode:
            repo_ids = ["RoboCOIN/Split_aloha_plate_storage"]
            logger.info(f"[TEST MODE] Building only {repo_ids[0]}")
            try:
                repo_episode_lengths[repo_ids[0]] = _load_episode_lengths(repo_ids[0])
            except Exception as e:
                logger.info(f"  Warning: could not get episode lengths for {repo_ids[0]}: {e}")
        else:
            api = HfApi()
            logger.info(f"Listing datasets with prefix '{PREFIX}' from Hugging Face...")
            infos = api.list_datasets(search = PREFIX)
            all_repo_ids = sorted([d.id for d in infos if d.id.startswith(PREFIX)])

            logger.info(f"Found {len(all_repo_ids)} total repos. Filtering based on robot_type and state_dim...")

            repo_ids = []
            for repo_id in all_repo_ids:
                try:
                    info_path = _download_hf_meta(repo_id, "meta/info.json")
                    with open(info_path, 'r') as f:
                        info_data = json.load(f)

                    should_include, robot_type, state_dim = should_include_repo(repo_id, info_data)
                    if should_include:
                        ep_lengths = _load_episode_lengths(repo_id)
                        repo_ids.append(repo_id)
                        repo_episode_lengths[repo_id] = ep_lengths
                        logger.info(f"  Including {repo_id} (robot_type={robot_type}, state_dim={state_dim}, episodes={len(ep_lengths)})")
                    else:
                        logger.info(f"  Skipping {repo_id} (robot_type={robot_type}, state_dim={state_dim})")
                except Exception as e:
                    logger.info(f"  Skipping {repo_id}: Could not download meta files: {e}")
                    continue

        logger.info(f"\nFiltered to {len(repo_ids)} repos")

        # Shuffle with fixed seed for reproducibility
        random.seed(86)
        random.shuffle(repo_ids)

        # Ensure download root exists
        os.makedirs(DOWNLOAD_ROOT, exist_ok = True)

        # Pre-compute episode indices per repo per split (including worker slicing)
        # Each dict maps repo_id -> (episode_indices, effective_workers)
        train_repo_ids = []
        val_repo_ids = []
        train_episode_indices: Dict[str, Tuple[List[int], int]] = {}
        val_episode_indices: Dict[str, Tuple[List[int], int]] = {}

        for repo_id in repo_ids:
            ep_lengths = repo_episode_lengths.get(repo_id)
            if not ep_lengths:
                raise ValueError(f"No episode lengths available for {repo_id}")
            total_ep = len(ep_lengths)

            for split_name, repo_list, indices_dict in [
                ('train', train_repo_ids, train_episode_indices),
                ('val', val_repo_ids, val_episode_indices),
            ]:
                indices = _get_episode_indices_for_split(total_ep, split_name, repo_id)
                if not indices:
                    continue

                # Worker-level episode slicing
                eff_workers = 1
                if NUM_WORKERS > 1:
                    total_steps = sum(ep_lengths[idx] for idx in indices)
                    eff_workers = min(NUM_WORKERS, (total_steps + MIN_STEPS_PER_WORKER - 1) // MIN_STEPS_PER_WORKER)
                    if WORKER_ID >= eff_workers:
                        logger.info(f"  Worker {WORKER_ID}: skipping {repo_id} [{split_name}] ({total_steps} steps, {eff_workers} effective workers)")
                        continue
                    indices = indices[WORKER_ID :: eff_workers]
                    logger.info(f"  Worker {WORKER_ID}/{NUM_WORKERS}: {len(indices)} episodes for {repo_id} [{split_name}] ({eff_workers} effective workers)")

                if indices:
                    repo_list.append(repo_id)
                    indices_dict[repo_id] = (indices, eff_workers)

        logger.info(f"Split assignment: {len(train_repo_ids)} repos -> train, {len(val_repo_ids)} repos -> val")

        splits = {}
        if train_repo_ids:
            splits['train'] = self._generate_examples(
                repo_ids = train_repo_ids, split = 'train',
                repo_episode_indices = train_episode_indices,
                test_mode = test_mode,
            )
        if val_repo_ids:
            splits['val'] = self._generate_examples(
                repo_ids = val_repo_ids, split = 'val',
                repo_episode_indices = val_episode_indices,
                test_mode = test_mode,
            )
        return splits

    def _process_step(
        self, item, i, ep_len, ep_idx, repo_id, repo_index,
        camera_names, num_cameras, state_indices, action_indices,
        eef_state_indices, eef_action_indices,
        null_subtask_index, subtask_index_to_text, tasks_list,
        task_desc, prev_state, prev_action, prev_eef_action,
        steps_list, all_subtask_annotations,
        preresized_images,
    ):
        """Process a single step from a DataLoader batch item.

        Handles image encoding, state/action filtering, subtask annotations,
        and RLDS field population. Mutates steps_list[-1] diffs for the
        previous step when prev values are not None.

        Args:
            preresized_images: Dict mapping cam_key -> uint8 (224,224,3) array.

        Returns:
            step dict ready for appending to steps_list
        """
        step = {}

        # Images
        for cam_idx in range(MAX_CAMERAS):
            cam_key = f'observation/image/cam_{cam_idx}'
            if cam_idx < num_cameras:
                step[cam_key] = tf.image.encode_jpeg(preresized_images[cam_key]).numpy()
            else:
                step[cam_key] = b''

        # State
        state_val = item['observation.state'].numpy().astype(np.float32)
        state_val = state_val[state_indices]
        if state_val.shape[0] != MAX_STATE_DIM:
            raise ValueError(
                f"After filtering, state dimension is {state_val.shape[0]}, "
                f"expected {MAX_STATE_DIM} in {repo_id}"
            )
        step['observation/state'] = state_val

        # State diff
        step['state_diff'] = np.zeros(MAX_STATE_DIM, dtype = np.float32)
        if prev_state is not None:
            steps_list[-1]['state_diff'] = state_val - prev_state

        # Action
        act_val = item['action'].numpy().astype(np.float32)
        act_val = act_val[action_indices]
        if act_val.shape[0] != MAX_ACTION_DIM:
            raise ValueError(
                f"After filtering, action dimension is {act_val.shape[0]}, "
                f"expected {MAX_ACTION_DIM} in {repo_id}"
            )
        step['action'] = act_val

        # Action diff
        step['action_diff'] = np.zeros(MAX_ACTION_DIM, dtype = np.float32)
        if prev_action is not None:
            steps_list[-1]['action_diff'] = act_val - prev_action

        # Subtask annotations
        subtask_ann = item['subtask_annotation'].numpy().astype(np.int32)
        is_null = np.array([subtask_ann[pos] == null_subtask_index for pos in range(5)], dtype = np.bool_)
        non_null_indices = np.where(~is_null)[0]
        null_indices = np.where(is_null)[0]
        if len(non_null_indices) > 0 and len(null_indices) > 0:
            assert np.max(non_null_indices) < np.min(null_indices), (
                f"Subtask annotation not in [non_null] + [null] form at step {i} of episode {ep_idx} in {repo_id}. "
                f"subtask_ann: {subtask_ann.tolist()}, is_null: {is_null.tolist()}"
            )
        first_null_idx = np.min(null_indices) if is_null.any() else 5
        all_subtask_annotations.append(subtask_ann)

        for pos in range(5):
            subtask_idx = int(subtask_ann[pos])
            subtask_text = subtask_index_to_text.get(subtask_idx, '')
            step[f'subtask_{pos + 1}'] = subtask_text

        subtask_mask = np.array([int(subtask_ann[pos]) != null_subtask_index for pos in range(5)], dtype = np.bool_)
        step['subtask_mask'] = subtask_mask
        step['first_null_index'] = first_null_idx
        step['scene_annotation'] = item['scene_annotation'].numpy().astype(np.int32).item()

        # Standard RLDS fields
        step['is_first'] = (i == 0)
        step['is_terminal'] = (i == ep_len - 1)
        step['frame_index'] = item['frame_index']
        task_idx = int(item['task_index'])
        step['task'] = tasks_list[task_idx]['task']
        assert step['task'] == task_desc
        step['episode_index'] = item['episode_index']
        step['index'] = item['index']

        step['eef_sim_pose_state'] = item['eef_sim_pose_state'].numpy().astype(np.float32)[eef_state_indices]
        eef_act_val = item['eef_sim_pose_action'].numpy().astype(np.float32)[eef_action_indices]
        step['eef_sim_pose_action'] = eef_act_val

        # EEF action diff
        step['eef_sim_pose_action_diff'] = np.zeros(12, dtype = np.float32)
        if prev_eef_action is not None:
            steps_list[-1]['eef_sim_pose_action_diff'] = eef_act_val - prev_eef_action

        step['repo_index'] = repo_index
        return step

    def _generate_examples(self, repo_ids: List[str], split: str, repo_episode_indices: Dict[str, Tuple[List[int], int]], test_mode: bool = False) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        for repo_index, repo_id in enumerate(repo_ids):
            logger.info(f"Processing {repo_id}...")

            dataset_path = os.path.join(DOWNLOAD_ROOT, repo_id)
            meta_path = os.path.join(dataset_path, "meta")
            repo_suffix = repo_id[len(PREFIX) : ]

            # Acquire per-repo file lock so concurrent workers don't race
            # on download + NaN fix + LeRobotDataset creation.
            lock_path = os.path.join(DOWNLOAD_ROOT, f".lock_{repo_suffix}")
            lock_fd = open(lock_path, 'w')
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            logger.info(f"  Acquired lock for {repo_id}")

            try:
                cmd = [
                    "robocoin-download",
                    "--hub", "huggingface",
                    "--target-dir", DOWNLOAD_ROOT,
                    "--ds_lists", repo_suffix
                ]

                try:
                    run_with_rate_limit_retry(cmd, sleep_seconds = 90)
                except Exception as e:
                    logger.info(f"Failed to download {repo_id}: {e}")
                    traceback.print_exc()
                    continue

                info_json_path = os.path.join(meta_path, "info.json")
                with open(info_json_path, 'r') as f:
                    info_data = json.load(f)

                robot_type = extract_robot_type_from_repo_id(repo_id)
                fps = float(info_data['fps'])

                features = info_data['features']
                all_camera_shapes = {}

                for key, val in features.items():
                    if isinstance(val, dict) and val.get('dtype') == 'video':
                        parts = key.split('.')
                        cam_name = "".join(parts[2 : ])
                        if "fisheye" in cam_name:
                            continue

                        video_info = val['info']
                        h = video_info['video.height']
                        w = video_info['video.width']
                        c = video_info['video.channels']
                        v_fps = video_info['video.fps']

                        if v_fps is not None and abs(v_fps - fps) > 1e-4:
                            raise ValueError(f"Video FPS {v_fps} != global FPS {fps} in {repo_id}")

                        all_camera_shapes[cam_name] = np.array([h, w, c], dtype = np.int32)

                camera_names = get_camera_names_for_repo(features, robot_type)
                camera_shapes = [all_camera_shapes[cam] for cam in camera_names]
                num_cameras = len(camera_names)

                state_feat_names = features['observation.state']['names']
                state_indices = np.array([state_feat_names.index(n) for n in REQUIRED_DIM_NAMES], dtype = np.int32)
                logger.info(f"  State dimension indices for {repo_id}: {state_indices.tolist()}")

                action_feat_names = features['action']['names'] if 'action' in features else []
                action_indices = np.array([action_feat_names.index(n) for n in REQUIRED_DIM_NAMES], dtype = np.int32)
                logger.info(f"  Action dimension indices for {repo_id}: {action_indices.tolist()}")

                # EEF dimension indices
                assert 'eef_sim_pose_state' in features, f"eef_sim_pose_state not in features for {repo_id}"
                assert 'eef_sim_pose_action' in features, f"eef_sim_pose_action not in features for {repo_id}"
                eef_state_feat_names = features['eef_sim_pose_state']['names']
                eef_action_feat_names = features['eef_sim_pose_action']['names']
                eef_state_indices = np.array([eef_state_feat_names.index(n) for n in EEF_DIM_NAMES], dtype = np.int32)
                eef_action_indices = np.array([eef_action_feat_names.index(n) for n in EEF_DIM_NAMES], dtype = np.int32)

                # Subtasks
                subtasks_path = os.path.join(dataset_path, "annotations", "subtask_annotations.jsonl")
                subtasks_list = []
                subtask_index_to_text = {}
                null_subtask_index = None
                with open(subtasks_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            subtasks_list.append(data['subtask'])
                            subtask_index_to_text[data['subtask_index']] = data['subtask']
                            if data['subtask'] == 'null':
                                null_subtask_index = data['subtask_index']
                if null_subtask_index is None:
                    raise ValueError(f"No null subtask found in {subtasks_path}")

                # Tasks
                tasks_path = os.path.join(meta_path, "tasks.jsonl")
                tasks_list = []
                with open(tasks_path, 'r') as f:
                    count = 0
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            assert data['task_index'] == count
                            tasks_list.append(data)
                            count += 1

                # Episodes metadata
                episodes_path = os.path.join(meta_path, "episodes.jsonl")
                episodes_meta = []
                with open(episodes_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            episodes_meta.append(json.loads(line))

                # Fix NaN values in episodes_stats.jsonl if needed, then create dataset
                stats_path = os.path.join(meta_path, "episodes_stats.jsonl")
                if os.path.exists(stats_path):
                    with open(stats_path, 'r') as f:
                        content = f.read()
                    if 'NaN' in content:
                        content = content.replace('NaN', '0')
                        with open(stats_path, 'w') as f:
                            f.write(content)
                        logger.info(f"  Fixed NaN values in {stats_path}")

                try:
                    ds = LeRobotDataset(root = DOWNLOAD_ROOT + repo_id, repo_id = repo_id, video_backend = "pyav")
                except Exception as e:
                    logger.info(f"Failed to create LeRobotDataset for {repo_id}: {e}")
                    traceback.print_exc()
                    continue
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()

            # Precompute cumulative episode offsets (avoids O(n^2))
            ep_lengths = [em['length'] for em in episodes_meta]
            cumulative_offsets = np.zeros(len(ep_lengths) + 1, dtype = np.int64)
            np.cumsum(ep_lengths, out = cumulative_offsets[1 : ])

            episode_indices, effective_workers = repo_episode_indices[repo_id]
            logger.info(f"  Processing {len(episode_indices)} episodes (total: {len(episodes_meta)})")

            for ep_count, ep_idx in enumerate(episode_indices):
                if ep_count >= 10 and test_mode:
                    break

                ep_info = episodes_meta[ep_idx]
                cumulative_idx = int(cumulative_offsets[ep_idx])
                ep_len = ep_info['length']
                ep_id = ep_info['episode_index']
                logger.info(f"Processing episode {ep_idx} of length {ep_len}")

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
                    'task_description': task_desc
                }

                steps_list = []
                all_subtask_annotations = []
                prev_state = None
                prev_action = None
                prev_eef_action = None

                episode_frame_indices = list(range(cumulative_idx, cumulative_idx + ep_len))
                episode_loader = DataLoader(Subset(ds, episode_frame_indices), batch_size = 8, num_workers = 4, prefetch_factor = 2, shuffle = False, drop_last = False)

                step_counter = 0
                for batch_idx, batch in enumerate(episode_loader):
                    batch_size_actual = next(iter(batch.values())).shape[0]

                    # Batch image resize
                    batch_preresized = [{} for _ in range(batch_size_actual)]
                    for cam_idx in range(num_cameras):
                        cam_key = f'observation/image/cam_{cam_idx}'
                        data_key = f"observation.images.{camera_names[cam_idx]}"
                        raw_imgs = batch[data_key].numpy()
                        if raw_imgs.shape[1] == 3 and raw_imgs.ndim == 4:
                            raw_imgs = np.transpose(raw_imgs, (0, 2, 3, 1))
                        if raw_imgs.dtype == np.uint8:
                            raw_imgs = raw_imgs.astype(np.float32) / 255.0
                        elif raw_imgs.dtype == np.float64:
                            raw_imgs = raw_imgs.astype(np.float32)
                        scaled_imgs = tf.image.resize(raw_imgs, (224, 224)).numpy()
                        scaled_imgs = np.rint(scaled_imgs * 255.0).astype(np.uint8)
                        for b in range(batch_size_actual):
                            batch_preresized[b][cam_key] = scaled_imgs[b]

                    for b in range(batch_size_actual):
                        i = step_counter
                        item = {k: v[b] for k, v in batch.items()}

                        step = self._process_step(
                            item, i, ep_len, ep_idx, repo_id, repo_index,
                            camera_names, num_cameras, state_indices, action_indices,
                            eef_state_indices, eef_action_indices,
                            null_subtask_index, subtask_index_to_text, tasks_list,
                            task_desc, prev_state, prev_action, prev_eef_action,
                            steps_list, all_subtask_annotations,
                            preresized_images = batch_preresized[b],
                        )
                        prev_state = step['observation/state'].copy()
                        prev_action = step['action'].copy()
                        prev_eef_action = step['eef_sim_pose_action'].copy()
                        steps_list.append(step)
                        step_counter += 1

                # Subtask timing features (set-difference logic)
                steps_to_subtask_end_list = [np.zeros(5, dtype = np.int32) for _ in range(ep_len)]
                subtask_len_list = [np.zeros(5, dtype = np.int32) for _ in range(ep_len)]
                subtask_is_first_list = [np.zeros(5, dtype = np.bool_) for _ in range(ep_len)]
                subtask_is_last_list = [np.zeros(5, dtype = np.bool_) for _ in range(ep_len)]

                subtask_start_steps = {}
                current_subtasks = set()

                def fill_subtask_interval(subtask_id: int, start_step: int, end_step: int):
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
                    'episode_metadata': episode_metadata
                }

            # Deletion of local downloads is handled by the shell script (build_local.sh)
            # after tfds build completes, since dataset_info.json (the completion marker)
            # is only written after this generator is fully exhausted.

            logger.info(f"Processed {repo_id} successfully (index {repo_index})")

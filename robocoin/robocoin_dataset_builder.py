"""
RoboCOIN Dataset Builder for RLDS/TensorFlow Datasets.
Merges multiple Hugging Face RoboCOIN datasets into a single RLDS dataset.
"""

from typing import Iterator, Tuple, Any, List, Dict
import os
import json
import random
import subprocess
import time
import sys
import numpy as np
import torch
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

# ALLOWED_REPO_IDS = ["RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning"]

# Default required cameras order: front/high, left wrist, right wrist
DEFAULT_CAMERA_ORDER = ["cam_high_rgb", "cam_left_wrist_rgb", "cam_right_wrist_rgb"]

PROFILE_BUILD = os.environ.get("PROFILE_BUILD", "0") == "1"
USE_DATALOADER = os.environ.get("USE_DATALOADER", "1") == "1"


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
            sys.stderr.write(
                f"[rate-limit] Command failed with rate limit markers. "
                f"Sleeping {sleep_seconds}s then retrying. Attempt={attempt}\n"
            )
            if max_retries is not None and attempt > max_retries:
                raise RuntimeError(f"Exceeded max_retries={max_retries} for command: {cmd}")
            time.sleep(sleep_seconds)
            continue

        if p.returncode == 0:
            if p.stdout:
                sys.stdout.write(p.stdout)
            if p.stderr:
                sys.stderr.write(p.stderr)
            return

        sys.stderr.write(out)
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
                    'scene_description': tfds.features.Text(),
                    'task_description': tfds.features.Text(),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        from huggingface_hub import hf_hub_download
        
        print("Starting dataset build...")
        
        test_mode = os.environ.get("TEST_MODE", "0") == "1"

        if test_mode:
            repo_ids = ["RoboCOIN/Cobot_Magic_take_out_a_pen_from_the_pen_holder"]
            print(f"[TEST MODE] Building only {repo_ids[0]}")
        else:
            api = HfApi()
            print(f"Listing datasets with prefix '{PREFIX}' from Hugging Face...")
            infos = api.list_datasets(search = PREFIX)
            all_repo_ids = sorted([d.id for d in infos if d.id.startswith(PREFIX)])

            print(f"Found {len(all_repo_ids)} total repos. Filtering based on robot_type and state_dim...")

            repo_ids = []
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
                                continue
                        except Exception as download_err:
                            print(
                                f"  Retry {attempt} for {repo_id} "
                                f"meta/info.json failed: {download_err}"
                            )
                            time.sleep(retry_delay_seconds)
                            attempt += 1

                    with open(info_path, 'r') as f:
                        info_data = json.load(f)

                    should_include, robot_type, state_dim = should_include_repo(repo_id, info_data)
                    if should_include:
                        repo_ids.append(repo_id)
                        print(f"  Including {repo_id} (robot_type={robot_type}, state_dim={state_dim})")
                    else:
                        print(f"  Skipping {repo_id} (robot_type={robot_type}, state_dim={state_dim})")
                except Exception as e:
                    print(f"  Skipping {repo_id}: Could not download info.json: {e}")
                    continue

        print(f"\nFiltered to {len(repo_ids)} repos")

        # Shuffle with fixed seed for reproducibility
        random.seed(86)
        random.shuffle(repo_ids)
        
        # Ensure download root exists
        os.makedirs(DOWNLOAD_ROOT, exist_ok = True)
        
        # Global dictionary to track train episode indices per repo_id
        # This ensures train and val splits are disjoint with randomized assignment
        self._train_indices_per_repo: Dict[str, set] = {}

        return {
            'train': self._generate_examples(repo_ids = repo_ids, split = 'train', test_mode = test_mode),
            'val': self._generate_examples(repo_ids = repo_ids, split = 'val', test_mode = test_mode),
        }

    def _process_step(
        self, item, i, ep_len, ep_idx, repo_id, repo_index,
        camera_names, num_cameras, state_indices, action_indices,
        null_subtask_index, subtask_index_to_text, tasks_list,
        task_desc, prev_action, steps_list, all_subtask_annotations,
    ):
        """Process a single step from either DataLoader batch item or direct ds[idx].

        Handles image encoding, state/action filtering, subtask annotations,
        and RLDS field population. Mutates steps_list[-1]['action_diff'] for the
        previous step when prev_action is not None.

        Returns:
            step dict ready for appending to steps_list
        """
        step = {}

        # Images
        for cam_idx in range(MAX_CAMERAS):
            cam_key = f'observation/image/cam_{cam_idx}'
            if cam_idx < num_cameras:
                original_name = camera_names[cam_idx]
                data_key = f"observation.images.{original_name}"
                img_data = item[data_key]
                img_data = img_data.numpy()
                if img_data.shape[0] == 3 and img_data.ndim == 3:
                    img_data = np.transpose(img_data, (1, 2, 0))
                if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                    img_data = (img_data * 255).astype(np.uint8)
                step[cam_key] = tf.image.encode_jpeg(img_data).numpy()
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

        if 'eef_sim_pose_state' in item:
            step['eef_sim_pose_state'] = item['eef_sim_pose_state'].numpy().astype(np.float32)
        else:
            step['eef_sim_pose_state'] = np.zeros(12, dtype = np.float32)
        if 'eef_sim_pose_action' in item:
            step['eef_sim_pose_action'] = item['eef_sim_pose_action'].numpy().astype(np.float32)
        else:
            step['eef_sim_pose_action'] = np.zeros(12, dtype = np.float32)

        step['repo_index'] = repo_index
        return step

    def _generate_examples(self, repo_ids: List[str], split: str = 'train', test_mode: bool = False) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if PROFILE_BUILD:
            profile = {
                'batch_fetch': {'total': 0.0, 'count': 0},
                'batch_compute': {'total': 0.0, 'count': 0},
                'repo_init': {'total': 0.0, 'count': 0},
                'episode_init': {'total': 0.0, 'count': 0},
            }

        for repo_index, repo_id in enumerate(repo_ids):
            if PROFILE_BUILD:
                t_repo_init_start = time.perf_counter()

            print(f"Processing {repo_id}...")
            
            # Paths
            dataset_path = os.path.join(DOWNLOAD_ROOT, repo_id)
            meta_path = os.path.join(dataset_path, "meta")
            
            # 1. Download
            # Remove "RoboCOIN/" prefix for the download command argument if needed, 
            # but robocoin-download usually takes the full string or suffix. 
            repo_suffix = repo_id[len(PREFIX) : ]
            
            # Use robocoin-download command
            cmd = [
                "robocoin-download", 
                "--hub", "huggingface", 
                "--target-dir", DOWNLOAD_ROOT, 
                "--ds_lists", repo_suffix
            ]
            
            try:
                run_with_rate_limit_retry(cmd, sleep_seconds = 90)
            except Exception as e:
                print(f"Failed to download {repo_id}: {e}")
                continue

            # 2. Extract General Metadata (Common across episodes)
            info_json_path = os.path.join(meta_path, "info.json")
            with open(info_json_path, 'r') as f:
                info_data = json.load(f)

            # 2.1 robot_type & fps
            robot_type = extract_robot_type_from_repo_id(repo_id)
            fps = float(info_data['fps'])

            # 2.2 Cameras and Shapes - only include required cameras
            features = info_data['features']
            all_camera_names = []
            all_camera_shapes = {}
            
            for key, val in features.items():
                if isinstance(val, dict) and val.get('dtype') == 'video':
                    # Key format is "observation.images.<name>"
                    parts = key.split('.')
                    cam_name = "".join(parts[2 : ])
                    if "fisheye" in cam_name:
                        continue
                    
                    video_info = val['info']
                    h = video_info['video.height']
                    w = video_info['video.width']
                    c = video_info['video.channels']
                    v_fps = video_info['video.fps']
                    
                    # Assert FPS match
                    if v_fps is not None and abs(v_fps - fps) > 1e-4:
                        raise ValueError(f"Video FPS {v_fps} != global FPS {fps} in {repo_id}")

                    all_camera_names.append(cam_name)
                    all_camera_shapes[cam_name] = np.array([h, w, c], dtype=np.int32)
            
            # Get camera names based on robot type (handles different naming conventions)
            camera_names = get_camera_names_for_repo(features, robot_type)
            camera_shapes = [all_camera_shapes[cam] for cam in camera_names]
            num_cameras = len(camera_names)

            # 2.3 State/Action Feature Names
            state_feat_names = []
            state_feat_names = features['observation.state']['names']
            
            # 2.3.1 Find indices for required state dimensions (14 total)
            # Order: left_arm_joint_1-6_rad, left_gripper_open, right_arm_joint_1-6_rad, right_gripper_open
            required_state_names = []
            for i in range(1, 7):
                required_state_names.append(f'left_arm_joint_{i}_rad')
            required_state_names.append('left_gripper_open')
            for i in range(1, 7):
                required_state_names.append(f'right_arm_joint_{i}_rad')
            required_state_names.append('right_gripper_open')
            
            state_indices = []
            for req_name in required_state_names:
                try:
                    idx = state_feat_names.index(req_name)
                    state_indices.append(idx)
                except ValueError:
                    raise ValueError(
                        f"Required state dimension '{req_name}' not found in {repo_id}. "
                        f"Available state names: {state_feat_names}"
                    )
            
            state_indices = np.array(state_indices, dtype=np.int32)
            print(f"  State dimension indices for {repo_id}: {state_indices.tolist()}")
            print(f"  Selected state dimensions: {[state_feat_names[i] for i in state_indices]}")
            
            action_feat_names = []
            if 'action' in features:
                action_feat_names = features['action']['names']
                        
            # 2.3.2 Find indices for required action dimensions (14 total, same names as state)
            # Order: left_arm_joint_1-6_rad, left_gripper_open, right_arm_joint_1-6_rad, right_gripper_open
            action_indices = []
            for req_name in required_state_names:
                try:
                    idx = action_feat_names.index(req_name)
                    action_indices.append(idx)
                except ValueError:
                    raise ValueError(
                        f"Required action dimension '{req_name}' not found in {repo_id}. "
                        f"Available action names: {action_feat_names}"
                    )
            
            action_indices = np.array(action_indices, dtype=np.int32)
            print(f"  Action dimension indices for {repo_id}: {action_indices.tolist()}")
            print(f"  Selected action dimensions: {[action_feat_names[i] for i in action_indices]}")

            # 2.4 Subtasks
            subtasks_path = os.path.join(dataset_path, "annotations", "subtask_annotations.jsonl")
            subtasks_list = []
            with open(subtasks_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        subtasks_list.append(data['subtask'])

            # 3. Load Episode Specific Data
            
            # Load Scene Annotations
            scene_path = os.path.join(dataset_path, "annotations", "scene_annotations.jsonl")
            scene_annotations = []
            with open(scene_path, 'r') as f:
                for line in f:
                    if line.strip():
                        scene_annotations.append(json.loads(line))

            # Load Tasks
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

            # Load Episodes Metadata (for lengths)
            episodes_path = os.path.join(meta_path, "episodes.jsonl")
            episodes_meta = []
            with open(episodes_path, 'r') as f:
                for line in f:
                    if line.strip():
                        episodes_meta.append(json.loads(line))
            
            try:
                ds = LeRobotDataset(root = DOWNLOAD_ROOT + repo_id, repo_id = repo_id, video_backend = "pyav")
            except Exception:
                stats_path = os.path.join(meta_path, "episodes_stats.jsonl")
                if os.path.exists(stats_path):
                    with open(stats_path, 'r') as f:
                        content = f.read()
                    content = content.replace('NaN', '0')
                    with open(stats_path, 'w') as f:
                        f.write(content)
                    print(f"Fixed NaN values in {stats_path}, retrying...")
                    try:
                        ds = LeRobotDataset(root = DOWNLOAD_ROOT + repo_id, repo_id = repo_id, video_backend = "pyav")
                    except Exception as e:
                        print(f"Failed to create LeRobotDataset for {repo_id}: {e}")
                        continue
                else:
                    print(f"Failed to create LeRobotDataset for {repo_id}")
                    continue

            # # OLD LOGIC: Create DataLoader for efficient iteration (outside episode loop)
            # ds_loader = DataLoader(
            #     ds,
            #     batch_size=32,
            #     shuffle=False,
            #     num_workers=4,
            #     prefetch_factor=2,
            # )
            # ds_iter = iter(ds_loader)

            # 4. Iterate Episodes (split into train/val with randomized assignment)
            total_episodes = len(episodes_meta)
            val_count = max(1, int(total_episodes * VAL_SPLIT_RATIO))  # At least 1 val episode
            train_count = total_episodes - val_count
            
            # Determine episode indices based on split
            if split == 'train':
                # Randomly select train_count indices and store them
                all_indices = list(range(total_episodes))
                random.seed(86)  # Fixed seed for reproducibility
                random.shuffle(all_indices)
                train_indices = set(all_indices[:train_count])
                self._train_indices_per_repo[repo_id] = train_indices
                episode_indices = sorted(train_indices)
            else:  # val
                # Val indices are the complement of train indices
                train_indices = self._train_indices_per_repo[repo_id]
                episode_indices = sorted(set(range(total_episodes)) - train_indices)
            
            print(f"  Split '{split}': {len(episode_indices)} episodes (total: {total_episodes}, train: {train_count}, val: {val_count})")

            if PROFILE_BUILD:
                profile['repo_init']['total'] += time.perf_counter() - t_repo_init_start
                profile['repo_init']['count'] += 1

            for ep_idx in episode_indices:
                if episode_indices.index(ep_idx) >= 50 and test_mode:
                    break

                if PROFILE_BUILD:
                    t_ep_init_start = time.perf_counter()

                ep_info = episodes_meta[ep_idx]
                
                # Calculate cumulative_idx for this episode
                cumulative_idx = sum(episodes_meta[i]['length'] for i in range(ep_idx))                
                # 4.1 Get Episode Specific Metadata
                ep_len = ep_info['length']
                ep_id = ep_info['episode_index']
                print(f"Processing episode {ep_idx} of length {ep_len}")
                
                # Scene Description
                scene_desc = scene_annotations[ep_idx]['scene']
                
                # Task Description via LeRobotDataset access
                first_frame = ds[cumulative_idx]
                assert ep_id == first_frame['episode_index']
                task_index = first_frame['task_index']
                    
                task_desc = tasks_list[int(task_index)]['task']

                # Build Metadata Dictionary
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
                    'task_description': task_desc
                }

                # 4.2 Build subtask index to text mapping from annotations/subtask_annotations.jsonl
                subtask_ann_path = os.path.join(dataset_path, "annotations", "subtask_annotations.jsonl")
                subtask_index_to_text = {}  # subtask_index -> subtask text
                null_subtask_index = None
                with open(subtask_ann_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            subtask_idx = data['subtask_index']
                            subtask_text = data['subtask']
                            subtask_index_to_text[subtask_idx] = subtask_text
                            if subtask_text == 'null':
                                null_subtask_index = subtask_idx
                
                if null_subtask_index is None:
                    raise ValueError(f"Could not find null subtask index in {subtask_ann_path}")
                
                # 4.3 Single pass: build steps_list and collect subtask annotations
                steps_list = []
                all_subtask_annotations = []
                prev_action = None  # Track previous action for action_diff computation
                
                episode_frame_indices = list(range(cumulative_idx, cumulative_idx + ep_len))

                if USE_DATALOADER:
                    episode_loader = DataLoader(Subset(ds, episode_frame_indices), batch_size = 8, num_workers = 16, prefetch_factor = 4, shuffle = False)

                if PROFILE_BUILD:
                    profile['episode_init']['total'] += time.perf_counter() - t_ep_init_start
                    profile['episode_init']['count'] += 1
                    t_batch_fetch_start = time.perf_counter()

                if USE_DATALOADER:
                    step_counter = 0
                    for batch_idx, batch in enumerate(episode_loader):
                        if PROFILE_BUILD:
                            profile['batch_fetch']['total'] += time.perf_counter() - t_batch_fetch_start
                            profile['batch_fetch']['count'] += 1
                            t_compute_start = time.perf_counter()

                        batch_size_actual = next(iter(batch.values())).shape[0]
                        for b in range(batch_size_actual):
                            i = step_counter
                            item = {k: v[b] for k, v in batch.items()}

                            step = self._process_step(
                                item, i, ep_len, ep_idx, repo_id, repo_index,
                                camera_names, num_cameras, state_indices, action_indices,
                                null_subtask_index, subtask_index_to_text, tasks_list,
                                task_desc, prev_action, steps_list, all_subtask_annotations,
                            )
                            prev_action = step['action'].copy()
                            steps_list.append(step)
                            step_counter += 1

                        if PROFILE_BUILD:
                            profile['batch_compute']['total'] += time.perf_counter() - t_compute_start
                            profile['batch_compute']['count'] += 1
                            t_batch_fetch_start = time.perf_counter()
                else:
                    for i in range(ep_len):
                        if PROFILE_BUILD:
                            profile['batch_fetch']['total'] += time.perf_counter() - t_batch_fetch_start
                            profile['batch_fetch']['count'] += 1
                            t_compute_start = time.perf_counter()

                        global_idx = cumulative_idx + i
                        item = ds[global_idx]

                        step = self._process_step(
                            item, i, ep_len, ep_idx, repo_id, repo_index,
                            camera_names, num_cameras, state_indices, action_indices,
                            null_subtask_index, subtask_index_to_text, tasks_list,
                            task_desc, prev_action, steps_list, all_subtask_annotations,
                        )
                        prev_action = step['action'].copy()
                        steps_list.append(step)

                        if PROFILE_BUILD:
                            profile['batch_compute']['total'] += time.perf_counter() - t_compute_start
                            profile['batch_compute']['count'] += 1
                            t_batch_fetch_start = time.perf_counter()
                
                # 4.4 Compute subtask timing features using set difference logic
                # A subtask ends when it disappears from the active set (not when it changes position)
                # Subtasks can appear multiple times in different intervals
                
                # Initialize lists of arrays for each feature
                steps_to_subtask_end_list = [np.zeros(5, dtype=np.int32) for _ in range(ep_len)]
                subtask_len_list = [np.zeros(5, dtype=np.int32) for _ in range(ep_len)]
                subtask_is_first_list = [np.zeros(5, dtype=np.bool_) for _ in range(ep_len)]
                subtask_is_last_list = [np.zeros(5, dtype=np.bool_) for _ in range(ep_len)]
                
                subtask_start_steps = {}  # subtask_id -> start_step (for active subtasks)
                current_subtasks = set()
                
                def fill_subtask_interval(subtask_id: int, start_step: int, end_step: int):
                    """Fill subtask timing features for an interval [start_step, end_step]."""
                    interval_len = end_step - start_step + 1
                    for t in range(start_step, end_step + 1):
                        ann = all_subtask_annotations[t]
                        # Find the position where this subtask_id appears at timestep t
                        for pos in range(5):
                            if int(ann[pos]) == subtask_id:
                                steps_to_subtask_end_list[t][pos] = end_step - t
                                subtask_len_list[t][pos] = interval_len
                                subtask_is_first_list[t][pos] = (t == start_step)
                                subtask_is_last_list[t][pos] = (t == end_step)
                                break  # Found the position, move to next timestep
                
                for step_idx, ann in enumerate(all_subtask_annotations):
                    valid_subtasks = set(int(s) for s in ann if s != null_subtask_index)
                    
                    # Subtasks that ended (were in current but not in new)
                    ended_subtasks = current_subtasks - valid_subtasks
                    for subtask_id in ended_subtasks:
                        start_step = subtask_start_steps.pop(subtask_id)
                        end_step = step_idx - 1
                        fill_subtask_interval(subtask_id, start_step, end_step)
                    
                    # Subtasks that started (are in new but not in current)
                    new_subtasks = valid_subtasks - current_subtasks
                    for subtask_id in new_subtasks:
                        subtask_start_steps[subtask_id] = step_idx
                    
                    current_subtasks = valid_subtasks
                
                # Close subtasks still active at the end
                end_step = ep_len - 1
                for subtask_id in current_subtasks:
                    start_step = subtask_start_steps.pop(subtask_id)
                    fill_subtask_interval(subtask_id, start_step, end_step)
                
                # 4.6 Assign subtask timing fields to steps_list
                for i, step in enumerate(steps_list):
                    step['steps_to_subtask_end'] = steps_to_subtask_end_list[i]
                    step['subtask_len'] = subtask_len_list[i]
                    step['subtask_is_first'] = subtask_is_first_list[i]
                    step['subtask_is_last'] = subtask_is_last_list[i]
                
                # Yield Episode
                # Unique key: repo_id + episode_index (ep_idx is disjoint between train/val)
                yield f"{repo_id}_{ep_idx}", {
                    'steps': steps_list,
                    'episode_metadata': episode_metadata
                }

            if os.path.exists(dataset_path):
                print(f"Deleting {dataset_path}...")
                subprocess.run(['rm', '-rf', dataset_path])

            print(f"Processed {repo_id} successfully (index {repo_index})")

        if PROFILE_BUILD:
            print("\n" + "=" * 80)
            print(f"[PROFILE] Build Profiling Summary (split={split})")
            print("=" * 80)
            for key, stats in profile.items():
                avg = stats['total'] / stats['count'] if stats['count'] > 0 else 0.0
                print(f"  {key}: total={stats['total']:.2f}s, avg={avg:.4f}s, count={stats['count']}")
            print("=" * 80)

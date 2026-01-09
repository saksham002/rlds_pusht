"""
RoboCOIN Dataset Builder for RLDS/TensorFlow Datasets.
Merges multiple Hugging Face RoboCOIN datasets into a single RLDS dataset.
"""

from typing import Iterator, Tuple, Any, List, Dict, Optional
from dataclasses import dataclass
import glob
import os
import json
import shutil
import subprocess
import time
import sys
import numpy as np
import pdb
import torch
from torch.utils.data import DataLoader

# LeRobot import for task index extraction
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
from huggingface_hub import HfApi

# ---------- Robot-wise statistics tracking ----------
# Keys to track from episodes_stats.jsonl
STAT_KEYS_TO_TRACK = ["observation.state", "action", "eef_sim_pose_state", "eef_sim_pose_action"]


@dataclass
class RunningStats:
    """Holds running statistics for a single feature (e.g., observation.state)."""
    count: int = 0  # total count across all episodes
    min_vals: Optional[List[float]] = None  # element-wise min
    max_vals: Optional[List[float]] = None  # element-wise max
    mean_vals: Optional[List[float]] = None  # element-wise mean
    m2_vals: Optional[List[float]] = None  # for Welford's algorithm: sum of squared differences from mean
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        result = {"count": self.count}
        if self.min_vals is not None:
            result["min"] = self.min_vals
        if self.max_vals is not None:
            result["max"] = self.max_vals
        if self.mean_vals is not None:
            result["mean"] = self.mean_vals
        if self.m2_vals is not None and self.count > 0:
            # Compute std from m2
            result["std"] = [np.sqrt(m2 / self.count) if self.count > 0 else 0.0 for m2 in self.m2_vals]
        return result


def update_running_stats(running: RunningStats, episode_stats: Dict) -> None:
    """
    Update running statistics with a single episode's statistics.
    episode_stats has keys: count, min, max, mean, std
    count is a list of length 1, others have length = feature dimension.
    """    
    ep_count = episode_stats["count"][0]
    
    ep_min = episode_stats["min"]
    ep_max = episode_stats["max"]
    ep_mean = episode_stats["mean"]
    ep_std = episode_stats["std"]
    
    dim = len(ep_mean)
    
    # Initialize if first update
    if running.count == 0:
        running.min_vals = list(ep_min)
        running.max_vals = list(ep_max)
        running.mean_vals = list(ep_mean)
        # m2 = count * var = count * std^2
        running.m2_vals = [ep_count * (s ** 2) for s in ep_std]
        running.count = ep_count
        return
    
    # Update min/max element-wise
    running.min_vals = [min(r, e) for r, e in zip(running.min_vals, ep_min)]
    running.max_vals = [max(r, e) for r, e in zip(running.max_vals, ep_max)]
    
    # Parallel algorithm for combining mean and variance
    n_a = running.count
    n_b = ep_count
    n_ab = n_a + n_b
    
    new_mean_vals = []
    new_m2_vals = []
    
    for i in range(dim):
        mean_a = running.mean_vals[i]
        mean_b = ep_mean[i]
        
        # Combined mean
        new_mean = (n_a * mean_a + n_b * mean_b) / n_ab
        new_mean_vals.append(new_mean)
        
        # Combined M2 (sum of squared differences from mean)
        m2_a = running.m2_vals[i]
        var_b = ep_std[i] ** 2
        m2_b = n_b * var_b
        
        delta = mean_b - mean_a
        new_m2 = m2_a + m2_b + (delta ** 2) * n_a * n_b / n_ab
        new_m2_vals.append(new_m2)
    
    running.mean_vals = new_mean_vals
    running.m2_vals = new_m2_vals
    running.count = n_ab


def read_episodes_stats(dataset_path: str) -> List[Dict]:
    """Read all episode statistics from meta/episodes_stats.jsonl."""
    stats_path = os.path.join(dataset_path, "meta", "episodes_stats.jsonl")
    
    episodes = []
    with open(stats_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def update_robot_stats(
    robot_stats: Dict[str, Dict[str, RunningStats]],
    robot_type: str,
    episodes_stats: List[Dict],
) -> None:
    """
    Update robot-wise statistics from episode stats.
    robot_stats[robot_type][stat_key] = RunningStats
    """
    if robot_type not in robot_stats:
        robot_stats[robot_type] = {key: RunningStats() for key in STAT_KEYS_TO_TRACK}
    
    for ep in episodes_stats:
        stats_dict = ep["stats"]
        for key in STAT_KEYS_TO_TRACK:
            update_running_stats(robot_stats[robot_type][key], stats_dict[key])


def robot_stats_to_dict(robot_stats: Dict[str, Dict[str, RunningStats]]) -> Dict:
    """Convert robot_stats to JSON-serializable dict."""
    result = {}
    for robot_type, stats_by_key in robot_stats.items():
        result[robot_type] = {}
        for key, running in stats_by_key.items():
            if running.count > 0:
                result[robot_type][key] = running.to_dict()
    return result


def save_robot_stats(robot_stats: Dict[str, Dict[str, RunningStats]], output_path: str) -> None:
    """Save robot statistics to a JSON file."""
    stats_dict = robot_stats_to_dict(robot_stats)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved robot statistics to {output_path}")

# --- Configuration ---
# Directory where data will be temporarily downloaded
DOWNLOAD_ROOT = "/data/group_data/rl/saksham3/robocoin/"
PREFIX = "RoboCOIN/"
MAX_CAMERAS = 8

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
        # We use dynamic shapes or None dimensions because state dim varies across RoboCOIN datasets
        obs_features['observation/state'] = tfds.features.Tensor(shape=(None,), dtype=np.float32)
        
        # 3. Action
        action_feature = tfds.features.Tensor(shape=(None,), dtype=np.float32)

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    **obs_features,
                    'action': action_feature,
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                    'frame_index': tfds.features.Scalar(dtype=np.int64),
                    'task_index': tfds.features.Scalar(dtype=np.int64),
                    'episode_index': tfds.features.Scalar(dtype=np.int64),
                    'index': tfds.features.Scalar(dtype=np.int64),
                    'subtask_annotation': tfds.features.Tensor(shape=(5,), dtype=np.int32),
                    'scene_annotation': tfds.features.Scalar(dtype=np.int32),
                    'eef_sim_pose_state': tfds.features.Tensor(shape=(12,), dtype=np.float32),
                    'eef_sim_pose_action': tfds.features.Tensor(shape=(12,), dtype=np.float32),
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
        # 1. List all datasets starting with RoboCOIN/
        api = HfApi()
        print(f"Listing datasets with prefix '{PREFIX}' from Hugging Face...")
        infos = api.list_datasets(search = PREFIX)
        repo_ids = sorted([d.id for d in infos if d.id.startswith(PREFIX)])
        
        # Ensure download root exists
        os.makedirs(DOWNLOAD_ROOT, exist_ok = True)

        # Initialize robot-wise statistics tracking
        self._robot_stats: Dict[str, Dict[str, RunningStats]] = {}

        return {
            'train': self._generate_examples(repo_ids = repo_ids),
        }

    def _generate_examples(self, repo_ids: List[str]) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        import datasets
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        for repo_id in repo_ids:
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
            robot_type = info_data['robot_type']
            fps = float(info_data['fps'])

            # 2.1.1 Update robot-wise statistics from episodes_stats.jsonl
            episodes_stats = read_episodes_stats(dataset_path)
            update_robot_stats(self._robot_stats, robot_type, episodes_stats)
            print(f"  Updated robot stats for {robot_type} from {len(episodes_stats)} episodes")

            # 2.2 Cameras and Shapes
            features = info_data['features']
            camera_names = []
            camera_shapes = []
            
            for key, val in features.items():
                if isinstance(val, dict) and val.get('dtype') == 'video':
                    # Key format is "observation.images.<name>"
                    parts = key.split('.')
                    cam_name = "".join(parts[2 : ])
                    if "fisheye" in cam_name:
                        continue
                    camera_names.append(cam_name)
                    
                    video_info = val['info']
                    h = video_info['video.height']
                    w = video_info['video.width']
                    c = video_info['video.channels']
                    v_fps = video_info['video.fps']
                    
                    # Assert FPS match
                    if v_fps is not None and abs(v_fps - fps) > 1e-4:
                        raise ValueError(f"Video FPS {v_fps} != global FPS {fps} in {repo_id}")

                    camera_shapes.append(np.array([h, w, c], dtype=np.int32))

            num_cameras = len(camera_names)

            # 2.3 State/Action Feature Names
            state_feat_names = []
            state_feat_names = features['observation.state']['names']
            
            action_feat_names = []
            if 'action' in features:
                action_feat_names = features['action']['names']
            
            # Fallback: if action names missing, use state names
            if not action_feat_names:
                action_feat_names = state_feat_names

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
                for line in f:
                    if line.strip():
                        tasks_list.append(json.loads(line))

            # Load Episodes Metadata (for lengths)
            episodes_path = os.path.join(meta_path, "episodes.jsonl")
            episodes_meta = []
            with open(episodes_path, 'r') as f:
                for line in f:
                    if line.strip():
                        episodes_meta.append(json.loads(line))
            
            # Initialize LeRobotDataset for data access
            # Note: LeRobotDataset might print logs
            ds = LeRobotDataset(root = DOWNLOAD_ROOT + repo_id, repo_id = repo_id)

            # # OLD LOGIC: Create DataLoader for efficient iteration (outside episode loop)
            # ds_loader = DataLoader(
            #     ds,
            #     batch_size=32,
            #     shuffle=False,
            #     num_workers=4,
            #     prefetch_factor=2,
            # )
            # ds_iter = iter(ds_loader)

            # 4. Iterate Episodes
            cumulative_idx = 0
            
            for ep_idx, ep_info in enumerate(episodes_meta):                
                # 4.1 Get Episode Specific Metadata
                ep_len = ep_info['length']
                ep_id = ep_info['episode_index']
                print(f"Processing episode {ep_idx} of length {ep_len}")
                
                # Scene Description
                scene_desc = scene_annotations[ep_idx]['scene']
                
                if len(ep_info['tasks']) > 0:
                    task_desc = ep_info['tasks'][0]
                else:
                    # Task Description via LeRobotDataset access
                    # Access first frame of the episode
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

                # 4.2 Iterate Steps in Episode using direct dataset indexing
                steps_list = []
                
                # Process all frames in this episode using direct indexing
                for i in range(ep_len):
                    global_idx = cumulative_idx + i
                    # if global_idx % 32 == 0:
                    #     item = next(ds_iter)
                    
                    # idx = global_idx % 32
                    # Directly access the dataset using indexing
                    item = ds[global_idx]
                    
                    step = {}
                    
                    # Process Images
                    # Map actual keys (observation.images.X) to cam_Y based on camera_names order
                    for cam_idx in range(MAX_CAMERAS):
                        cam_key = f'observation/image/cam_{cam_idx}'
                        
                        if cam_idx < num_cameras:
                            original_name = camera_names[cam_idx]
                            # Reconstruct key: "observation.images." + name
                            # Note: features key was "observation.images.phone", name is "phone"
                            data_key = f"observation.images.{original_name}"
                            
                            # img_data = item[data_key][idx]
                            img_data = item[data_key]
                            
                            # LeRobot usually returns (C, H, W) float [0,1].
                            # Check type and shape.
                            img_data = img_data.numpy()
                            
                            # Convert (C, H, W) -> (H, W, C) if needed
                            if img_data.shape[0] == 3 and img_data.ndim == 3:
                                img_data = np.transpose(img_data, (1, 2, 0))
                            
                            # Convert Float [0,1] -> Uint8 [0,255] if needed
                            if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                                img_data = (img_data * 255).astype(np.uint8)
                            
                            # Encode JPEG
                            # tf.image.encode_jpeg requires input (H, W, C) uint8
                            encoded_jpeg = tf.image.encode_jpeg(img_data).numpy()
                            step[cam_key] = encoded_jpeg
                        else:
                            # Padding for cameras > num_cameras
                            step[cam_key] = b''

                    # Process State
                    # state_val = item['observation.state'][idx]
                    state_val = item['observation.state']
                    state_val = state_val.numpy()
                    step['observation/state'] = state_val.astype(np.float32)

                    # Process Action
                    # act_val = item['action'][idx]
                    act_val = item['action']
                    act_val = act_val.numpy()
                    step['action'] = act_val.astype(np.float32)

                    # step['subtask_annotation'] = item['subtask_annotation'][idx].numpy().astype(np.int32)
                    # step['scene_annotation'] = item['scene_annotation'][idx].numpy().astype(np.int32).item()
                    step['subtask_annotation'] = item['subtask_annotation'].numpy().astype(np.int32)
                    step['scene_annotation'] = item['scene_annotation'].numpy().astype(np.int32).item()

                    # Standard RLDS fields
                    step['is_first'] = (i == 0)
                    step['is_terminal'] = (i == ep_len - 1) # Assuming terminal at end of episode
                    # step['frame_index'] = item['frame_index'][idx]
                    # step['task_index'] = item['task_index'][idx]
                    # step['episode_index'] = item['episode_index'][idx]
                    # step['index'] = item['index'][idx]
                    step['frame_index'] = item['frame_index']
                    step['task_index'] = item['task_index']
                    step['episode_index'] = item['episode_index']
                    step['index'] = item['index']

                    if 'eef_sim_pose_state' in item:
                        # step['eef_sim_pose_action'] = item['eef_sim_pose_action'][idx].numpy().astype(np.float32)
                        step['eef_sim_pose_state'] = item['eef_sim_pose_state'].numpy().astype(np.float32)
                    else:
                        step['eef_sim_pose_state'] = np.zeros(12, dtype=np.float32)
                    if 'eef_sim_pose_action' in item:
                        # step['eef_sim_pose_action'] = item['eef_sim_pose_action'][idx].numpy().astype(np.float32)
                        step['eef_sim_pose_action'] = item['eef_sim_pose_action'].numpy().astype(np.float32)
                    else:
                        step['eef_sim_pose_action'] = np.zeros(12, dtype=np.float32)

                    steps_list.append(step)
                
                # Update cumulative index
                cumulative_idx += ep_len
                
                # Yield Episode
                # Unique key: repo_id + episode_index
                yield f"{repo_id}_{ep_idx}", {
                    'steps': steps_list,
                    'episode_metadata': episode_metadata
                }

            os.makedirs(os.path.join(DOWNLOAD_ROOT, "norm_stats/"), exist_ok = True)
            robot_stats_path = os.path.join(DOWNLOAD_ROOT, "norm_stats/", "robot_stats.json")
            save_robot_stats(self._robot_stats, robot_stats_path)
            
            if os.path.exists(dataset_path):
                print(f"Deleting {dataset_path}...")
                subprocess.run(['rm', '-rf', dataset_path])

            print(f"Processed {repo_id} successfully")
"""
Compute statistics (min, max, mean, sd) for observation.state, action,
and their diffs across episodes.
Usage: python compute_stats.py --repo_ids RoboCOIN/Split_aloha_plate_storage RoboCOIN/Cobot_Magic_cut_banana
"""

import argparse
import os
import json
import shutil
import subprocess
import sys
import time
import re
import numpy as np
from dataclasses import dataclass
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from typing import Dict, List, Optional
from pathlib import Path

DOWNLOAD_ROOT = "/data/group_data/rl/saksham3/robocoin/"
PREFIX = "RoboCOIN/"
NORM_STATS_DIR = "/data/group_data/rl/saksham3/robocoin/norm_stats/"

# Ensure HF token is set
assert os.environ.get("HF_TOKEN"), "HF_TOKEN environment variable must be set"

RATE_LIMIT_MARKERS = (
    "429 Too Many Requests",
)

# Required state names (14-D, same as in robocoin_dataset_builder.py)
REQUIRED_STATE_NAMES = []
for i in range(1, 7):
    REQUIRED_STATE_NAMES.append(f'left_arm_joint_{i}_rad')
REQUIRED_STATE_NAMES.append('left_gripper_open')
for i in range(1, 7):
    REQUIRED_STATE_NAMES.append(f'right_arm_joint_{i}_rad')
REQUIRED_STATE_NAMES.append('right_gripper_open')

FEATURE_KEYS = [
    'observation.state', 'action',
    'state_diff', 'action_diff',
    'eef_sim_pose_state', 'eef_sim_pose_action',
    'eef_sim_pose_state_diff', 'eef_sim_pose_action_diff',
]

DIFF_FEATURE_KEYS = {'state_diff', 'action_diff', 'eef_sim_pose_state_diff', 'eef_sim_pose_action_diff'}


def extract_robot_type_from_repo_id(repo_id: str) -> str:
    """Extract robot type from repo_id by splitting on _ or - and taking first 2 elements."""
    if '/' in repo_id:
        repo_name = repo_id.split('/', 1)[1]
    else:
        repo_name = repo_id

    parts = re.split(r'[_-]', repo_name)

    if len(parts) >= 2:
        robot_type = '_'.join(parts[:2])
    else:
        robot_type = parts[0] if parts else repo_name

    return robot_type


# Allowed robot types (extracted from first 2 parts of repo_id)
# R1_Lite is only allowed if state_dim == 14
ALLOWED_ROBOT_TYPES = ["Split_aloha", "Cobot_Magic", "R1_Lite"]

# Default required cameras order: front/high, left wrist, right wrist
DEFAULT_CAMERA_ORDER = ["cam_high_rgb", "cam_left_wrist_rgb", "cam_right_wrist_rgb"]


def get_camera_names_for_repo(features: Dict, robot_type: str) -> List[str]:
    """Get ordered camera names for a repo based on robot type."""
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


def should_include_repo(repo_id: str, info_data: Dict) -> tuple:
    """Determine if a repo should be included based on robot_type and state_dim."""
    robot_type = extract_robot_type_from_repo_id(repo_id)

    if robot_type not in ALLOWED_ROBOT_TYPES:
        return False, robot_type, 0

    state_dim = info_data['features']['observation.state']['shape'][0]

    if robot_type == "R1_Lite" and state_dim != 14:
        return False, robot_type, state_dim

    return True, robot_type, state_dim


def get_filtered_repo_ids() -> List[str]:
    """Get all repo_ids that match the filtering criteria."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    print(f"Listing datasets with prefix '{PREFIX}' from Hugging Face...")
    infos = api.list_datasets(search = PREFIX)
    all_repo_ids = sorted([d.id for d in infos if d.id.startswith(PREFIX)])

    print(f"Found {len(all_repo_ids)} total repos. Filtering based on robot_type and state_dim...")

    filtered_repos = []
    for repo_id in all_repo_ids:
        try:
            info_path = hf_hub_download(
                repo_id = repo_id,
                filename = "meta/info.json",
                repo_type = "dataset"
            )
            with open(info_path, 'r') as f:
                info_data = json.load(f)

            should_include, robot_type, state_dim = should_include_repo(repo_id, info_data)
            if should_include:
                filtered_repos.append(repo_id)
                print(f"  Including {repo_id} (robot_type={robot_type}, state_dim={state_dim})")
            else:
                print(f"  Skipping {repo_id} (robot_type={robot_type}, state_dim={state_dim})")
        except Exception as e:
            print(f"  Skipping {repo_id}: Could not download info.json: {e}")
            continue

    print(f"\nFiltered to {len(filtered_repos)} repos")
    return filtered_repos

@dataclass
class RunningStats:
    """Holds running statistics for a single feature using numpy arrays."""
    count: int = 0
    min_vals: Optional[np.ndarray] = None
    max_vals: Optional[np.ndarray] = None
    mean_vals: Optional[np.ndarray] = None
    m2_vals: Optional[np.ndarray] = None  # For Welford's algorithm

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict (converts numpy arrays to lists)."""
        result = {"count": self.count}
        if self.min_vals is not None:
            result["min"] = self.min_vals.tolist()
        if self.max_vals is not None:
            result["max"] = self.max_vals.tolist()
        if self.mean_vals is not None:
            result["mean"] = self.mean_vals.tolist()
        if self.m2_vals is not None and self.count > 0:
            result["std"] = np.sqrt(self.m2_vals / self.count).tolist()
        return result


def update_running_stats(running: RunningStats, episode_stats: Dict) -> None:
    """Update running statistics with a single episode's statistics using numpy."""

    ep_count = episode_stats["count"][0] if isinstance(episode_stats["count"], list) else episode_stats["count"]

    ep_min = np.array(episode_stats["min"], dtype = np.float64)
    ep_max = np.array(episode_stats["max"], dtype = np.float64)
    ep_mean = np.array(episode_stats["mean"], dtype = np.float64)
    ep_std = np.array(episode_stats["std"], dtype = np.float64)

    if np.isnan(ep_min).any() or np.isnan(ep_max).any() or np.isnan(ep_mean).any() or np.isnan(ep_std).any():
        return

    # Initialize if first update
    if running.count == 0:
        running.min_vals = ep_min.copy()
        running.max_vals = ep_max.copy()
        running.mean_vals = ep_mean.copy()
        running.m2_vals = ep_count * (ep_std ** 2)
        running.count = ep_count
        return

    # Update min/max element-wise
    np.minimum(running.min_vals, ep_min, out = running.min_vals)
    np.maximum(running.max_vals, ep_max, out = running.max_vals)

    # Parallel algorithm for combining mean and variance
    n_a = running.count
    n_b = ep_count
    n_ab = n_a + n_b

    delta = ep_mean - running.mean_vals
    new_mean = (n_a * running.mean_vals + n_b * ep_mean) / n_ab

    m2_b = n_b * (ep_std ** 2)
    new_m2 = running.m2_vals + m2_b + (delta ** 2) * n_a * n_b / n_ab

    running.mean_vals = new_mean
    running.m2_vals = new_m2
    running.count = n_ab


def update_running_from_array(running: RunningStats, values: np.ndarray) -> None:
    """Compute episode-level stats from a raw array and merge into running stats."""
    n = values.shape[0]
    if n == 0:
        return
    update_running_stats(running, {
        'count': n,
        'min': np.min(values, axis = 0),
        'max': np.max(values, axis = 0),
        'mean': np.mean(values, axis = 0),
        'std': np.std(values, axis = 0),
    })


def running_to_entry(running: RunningStats, names: List[str]) -> Dict:
    """Convert RunningStats into the flat-array output format."""
    d = running.to_dict()
    return {
        'names': names,
        'mean': d['mean'],
        'std': d['std'],
        'min': d['min'],
        'max': d['max'],
    }


def run_with_rate_limit_retry(
    cmd: List[str],
    sleep_seconds: int = 90,
    max_retries: int = None,
) -> None:
    """Run a command, retrying on Hugging Face Hub rate limit errors."""
    attempt = 0
    while True:
        attempt += 1
        p = subprocess.run(cmd, capture_output = True, text = True)
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
        raise subprocess.CalledProcessError(p.returncode, cmd, output = p.stdout, stderr = p.stderr)


def compute_stats_for_repo(repo_id: str, download: bool = True) -> Dict[str, any]:
    """Compute statistics for a single repository using episode-level aggregation.

    Iterates through episodes, computes per-episode stats, and merges them
    via RunningStats (Welford's parallel algorithm). Never stores full data.
    """
    print(f"\n{'=' * 80}")
    print(f"Processing {repo_id}")
    print(f"{'=' * 80}")

    dataset_path = os.path.join(DOWNLOAD_ROOT, repo_id)
    meta_path = os.path.join(dataset_path, "meta")

    if download:
        print(f"Downloading {repo_id}...")
        repo_suffix = repo_id[len(PREFIX):] if repo_id.startswith(PREFIX) else repo_id
        cmd = [
            "robocoin-download",
            "--hub", "huggingface",
            "--target-dir", DOWNLOAD_ROOT,
            "--ds_lists", repo_suffix
        ]
        try:
            run_with_rate_limit_retry(cmd, sleep_seconds = 90)
            print(f"Download complete.")
        except Exception as e:
            print(f"Failed to download {repo_id}: {e}")
            raise e

    info_json_path = os.path.join(meta_path, "info.json")
    with open(info_json_path, 'r') as f:
        info_data = json.load(f)

    state_names = info_data['features']['observation.state']['names']
    action_names = info_data['features']['action']['names']

    eef_state_names = info_data['features']['eef_sim_pose_state'].get('names',
                                                                          [f'eef_sim_pose_state_{i}' for i in range(12)])

    eef_action_names = info_data['features']['eef_sim_pose_action'].get('names',
                                                                             [f'eef_sim_pose_action_{i}' for i in range(12)])

    print(f"State dims: {len(state_names)}, Action dims: {len(action_names)}, "
          f"EEF state dims: {len(eef_state_names)}, EEF action dims: {len(eef_action_names)}")

    episodes_path = os.path.join(meta_path, "episodes.jsonl")
    if not os.path.exists(episodes_path):
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")

    episodes_meta = []
    with open(episodes_path, 'r') as f:
        for line in f:
            if line.strip():
                episodes_meta.append(json.loads(line))

    print(f"Found {len(episodes_meta)} episodes")

    try:
        ds = LeRobotDataset(root = DOWNLOAD_ROOT + repo_id, repo_id = repo_id)
    except Exception:
        stats_path = os.path.join(meta_path, "episodes_stats.jsonl")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                content = f.read()
            content = content.replace('NaN', '0')
            with open(stats_path, 'w') as f:
                f.write(content)
            print(f"Fixed NaN values in {stats_path}, retrying...")
            ds = LeRobotDataset(root = DOWNLOAD_ROOT + repo_id, repo_id = repo_id)
        else:
            raise

    # RunningStats for each feature
    state_running = RunningStats()
    action_running = RunningStats()
    state_diff_running = RunningStats()
    action_diff_running = RunningStats()
    eef_state_running = RunningStats()
    eef_action_running = RunningStats()
    eef_state_diff_running = RunningStats()
    eef_action_diff_running = RunningStats()
    total_steps = 0

    for ep_idx, ep_info in enumerate(episodes_meta):
        ep_len = ep_info['length']
        cumulative_idx = sum(episodes_meta[i]['length'] for i in range(ep_idx))

        if (ep_idx + 1) % 10 == 0:
            print(f"  Processing episode {ep_idx + 1}/{len(episodes_meta)} (length={ep_len})")

        ep_states = []
        ep_actions = []
        ep_eef_states = []
        ep_eef_actions = []

        for i in range(ep_len):
            global_idx = cumulative_idx + i
            item = ds[global_idx]

            episode_index = item['episode_index']
            if i == 0:
                assert episode_index == ep_idx, (
                    f"First datapoint of episode {ep_idx} has episode_index {episode_index}, expected {ep_idx}"
                )
            elif i == ep_len - 1:
                assert episode_index == ep_idx, (
                    f"Last datapoint of episode {ep_idx} has episode_index {episode_index}, expected {ep_idx}"
                )

            ep_states.append(item['observation.state'].numpy().astype(np.float64))
            ep_actions.append(item['action'].numpy().astype(np.float64))
            ep_eef_states.append(item['eef_sim_pose_state'].numpy().astype(np.float64))
            ep_eef_actions.append(item['eef_sim_pose_action'].numpy().astype(np.float64))

        ep_states = np.array(ep_states)
        ep_actions = np.array(ep_actions)
        ep_eef_states = np.array(ep_eef_states)
        ep_eef_actions = np.array(ep_eef_actions)
        total_steps += ep_len

        # Update non-diff running stats
        update_running_from_array(state_running, ep_states)
        update_running_from_array(action_running, ep_actions)
        update_running_from_array(eef_state_running, ep_eef_states)
        update_running_from_array(eef_action_running, ep_eef_actions)

        # Diff features: np.diff gives (ep_len - 1) rows, naturally excluding terminal
        if ep_len >= 2:
            update_running_from_array(state_diff_running, np.diff(ep_states, axis = 0))
            update_running_from_array(action_diff_running, np.diff(ep_actions, axis = 0))
            update_running_from_array(eef_state_diff_running, np.diff(ep_eef_states, axis = 0))
            update_running_from_array(eef_action_diff_running, np.diff(ep_eef_actions, axis = 0))

    print(f"  Processed {total_steps} steps across {len(episodes_meta)} episodes")

    return {
        'repo_id': repo_id,
        'num_episodes': len(episodes_meta),
        'num_steps': total_steps,
        'observation.state': running_to_entry(state_running, state_names),
        'action': running_to_entry(action_running, action_names),
        'state_diff': running_to_entry(state_diff_running, state_names),
        'action_diff': running_to_entry(action_diff_running, action_names),
        'eef_sim_pose_state': running_to_entry(eef_state_running, eef_state_names),
        'eef_sim_pose_action': running_to_entry(eef_action_running, eef_action_names),
        'eef_sim_pose_state_diff': running_to_entry(eef_state_diff_running, eef_state_names),
        'eef_sim_pose_action_diff': running_to_entry(eef_action_diff_running, eef_action_names),
    }


def aggregate_feature_stats_with_count(
    all_stats: Dict[str, Dict],
    feature_key: str,
    required_names: List[str],
) -> RunningStats:
    """Aggregate stats for a feature across repositories.

    For diff features, count = num_steps - num_episodes (excludes terminal
    dummy-zero steps). For non-diff features, count = num_steps.
    """
    is_diff = feature_key in DIFF_FEATURE_KEYS
    running = RunningStats()
    for _, repo_stats in all_stats.items():
        feat = repo_stats[feature_key]
        names = feat['names']
        indices = [names.index(n) for n in required_names]
        count = repo_stats['num_steps'] - repo_stats['num_episodes'] if is_diff else repo_stats['num_steps']
        per_repo_stats = {
            'count': count,
            'min': [feat['min'][i] for i in indices],
            'max': [feat['max'][i] for i in indices],
            'mean': [feat['mean'][i] for i in indices],
            'std': [feat['std'][i] for i in indices],
        }
        update_running_stats(running, per_repo_stats)
    return running


def compute_global_stats(all_stats: Dict[str, Dict]) -> Dict:
    """Compute global aggregated norm_stats from per-repo stats dicts.

    Works with both in-memory stats (from compute_stats_for_repo) and
    deserialized repo-wise JSON files (same flat-array structure).
    """
    total_episodes = sum(s['num_episodes'] for s in all_stats.values())
    total_steps = sum(s['num_steps'] for s in all_stats.values())

    # Aggregate 14-D state/action and their diffs
    state_running = aggregate_feature_stats_with_count(all_stats, 'observation.state', REQUIRED_STATE_NAMES)
    action_running = aggregate_feature_stats_with_count(all_stats, 'action', REQUIRED_STATE_NAMES)
    state_diff_running = aggregate_feature_stats_with_count(all_stats, 'state_diff', REQUIRED_STATE_NAMES)
    action_diff_running = aggregate_feature_stats_with_count(all_stats, 'action_diff', REQUIRED_STATE_NAMES)

    # Aggregate 12-D eef_sim_pose features (names must be consistent across repos)
    first_repo = next(iter(all_stats.values()))
    eef_state_names = first_repo['eef_sim_pose_state']['names']
    eef_action_names = first_repo['eef_sim_pose_action']['names']
    for repo_id, stats in all_stats.items():
        assert stats['eef_sim_pose_state']['names'] == eef_state_names, (
            f"eef_sim_pose_state names mismatch for {repo_id}"
        )
        assert stats['eef_sim_pose_action']['names'] == eef_action_names, (
            f"eef_sim_pose_action names mismatch for {repo_id}"
        )
    eef_state_running = aggregate_feature_stats_with_count(all_stats, 'eef_sim_pose_state', eef_state_names)
    eef_action_running = aggregate_feature_stats_with_count(all_stats, 'eef_sim_pose_action', eef_action_names)
    eef_state_diff_running = aggregate_feature_stats_with_count(all_stats, 'eef_sim_pose_state_diff', eef_state_names)
    eef_action_diff_running = aggregate_feature_stats_with_count(all_stats, 'eef_sim_pose_action_diff', eef_action_names)

    return {
        'num_episodes': total_episodes,
        'num_steps': total_steps,
        'observation.state': running_to_entry(state_running, REQUIRED_STATE_NAMES),
        'action': running_to_entry(action_running, REQUIRED_STATE_NAMES),
        'state_diff': running_to_entry(state_diff_running, REQUIRED_STATE_NAMES),
        'action_diff': running_to_entry(action_diff_running, REQUIRED_STATE_NAMES),
        'eef_sim_pose_state': running_to_entry(eef_state_running, eef_state_names),
        'eef_sim_pose_action': running_to_entry(eef_action_running, eef_action_names),
        'eef_sim_pose_state_diff': running_to_entry(eef_state_diff_running, eef_state_names),
        'eef_sim_pose_action_diff': running_to_entry(eef_action_diff_running, eef_action_names),
    }


def main():
    parser = argparse.ArgumentParser(
        description = "Compute statistics for observation.state, action, and diffs across episodes"
    )
    parser.add_argument(
        "--repo_ids",
        nargs = '*',
        default = None,
        help = "List of repo IDs (e.g., RoboCOIN/Split_aloha_plate_storage). "
             "If not provided, auto-discovers all repos matching filtering criteria "
             "(Split_aloha, Cobot_Magic, R1_Lite with state_dim=14)."
    )
    parser.add_argument(
        "--no_download",
        action = "store_true",
        help = "Skip downloading if dataset doesn't exist"
    )
    parser.add_argument(
        "--disable_global",
        action = "store_true",
        help = "Disable computation and writing of global norm_stats"
    )
    parser.add_argument(
        "--only_global",
        action = "store_true",
        help = "Only compute global norm_stats from existing repo-wise stats files. "
             "Skips per-repo computation entirely."
    )
    parser.add_argument(
        "--calc_indices",
        type = str,
        default = None,
        help = "Comma-separated pair 'a,b' (0-indexed, a < b) to process only repos[a:b] "
             "from the filtered list."
    )
    args = parser.parse_args()

    assert not (args.disable_global and args.only_global), (
        "--disable_global and --only_global are mutually exclusive"
    )

    norm_stats_dir = Path(NORM_STATS_DIR)
    norm_stats_dir.mkdir(parents = True, exist_ok = True)
    repo_wise_dir = norm_stats_dir / "repo_wise"
    repo_wise_dir.mkdir(parents = True, exist_ok = True)

    # ── only_global: aggregate from existing repo-wise JSONs ──
    if args.only_global:
        use_gcs = NORM_STATS_DIR.startswith("gs://")
        if use_gcs:
            import tensorflow as tf
            gfile = tf.io.gfile

        print("only_global mode: loading existing repo-wise stats...")
        all_stats = {}

        if use_gcs:
            stats_dir = NORM_STATS_DIR.rstrip('/')
            subdirs = gfile.listdir(stats_dir)
            for subdir in sorted(subdirs):
                candidate = os.path.join(stats_dir, subdir, "norm_stats.json")
                if gfile.exists(candidate):
                    with gfile.GFile(candidate, 'r') as f:
                        repo_data = json.load(f)
                    repo_id = repo_data['repo_id']
                    all_stats[repo_id] = repo_data
                    print(f"  Loaded {repo_id} ({repo_data['num_episodes']} episodes, {repo_data['num_steps']} steps)")
        else:
            for json_file in sorted(repo_wise_dir.glob("*_stats.json")):
                with open(json_file, 'r') as f:
                    repo_data = json.load(f)
                repo_id = repo_data['repo_id']
                all_stats[repo_id] = repo_data
                print(f"  Loaded {repo_id} ({repo_data['num_episodes']} episodes, {repo_data['num_steps']} steps)")

        if len(all_stats) == 0:
            print("No repo-wise stats found. Exiting.")
            return

        print(f"\nLoaded {len(all_stats)} repos. Computing global stats...")
        global_stats = compute_global_stats(all_stats)

        if use_gcs:
            norm_stats_file = os.path.join(NORM_STATS_DIR.rstrip('/'), "norm_stats.json")
            with gfile.GFile(norm_stats_file, 'w') as f:
                json.dump(global_stats, f, indent = 2)
        else:
            norm_stats_file = norm_stats_dir / "norm_stats.json"
            with open(norm_stats_file, 'w') as f:
                json.dump(global_stats, f, indent = 2)
        print(f"\nnorm_stats.json saved to {norm_stats_file}")
        return

    # ── Normal mode: compute per-repo stats ──
    if args.repo_ids is None or len(args.repo_ids) == 0:
        print("No repo_ids provided, auto-discovering based on filtering criteria...")
        repo_ids = get_filtered_repo_ids()
        if len(repo_ids) == 0:
            print("No repos found matching filtering criteria. Exiting.")
            return
    else:
        repo_ids = args.repo_ids

    # Apply calc_indices slicing
    if args.calc_indices is not None:
        parts = args.calc_indices.split(',')
        assert len(parts) == 2, f"--calc_indices must be 'a,b', got '{args.calc_indices}'"
        a, b = int(parts[0]), int(parts[1])
        assert 0 <= a < b, f"--calc_indices requires 0 <= a < b, got a={a}, b={b}"
        print(f"calc_indices: slicing repo list [{a}:{b}] from {len(repo_ids)} repos")
        repo_ids = repo_ids[a : b]
        print(f"Repos to process:")
        for idx, repo_id in enumerate(repo_ids):
            print(f"  [{a + idx}] {repo_id}")

    all_stats = {}

    for repo_id in repo_ids:
        repo_safe_name = repo_id.replace('/', '_').replace(' ', '_')
        if repo_safe_name.startswith('RoboCOIN_'):
            repo_safe_name = repo_safe_name[len('RoboCOIN_'):]
        stats_file = repo_wise_dir / f"{repo_safe_name}_stats.json"
        if stats_file.exists():
            print(f"\nSkipping {repo_id}: {stats_file} already exists")
            with open(stats_file, 'r') as f:
                all_stats[repo_id] = json.load(f)
            continue

        try:
            stats = compute_stats_for_repo(repo_id, download = not args.no_download)
            all_stats[repo_id] = stats
        except Exception as e:
            print(f"Error processing {repo_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Remove downloaded data to free disk space
        dataset_path = os.path.join(DOWNLOAD_ROOT, repo_id)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
            print(f"Removed downloaded data: {dataset_path}")

    # Save repo-wise stats
    for repo_id, stats in all_stats.items():
        repo_safe_name = repo_id.replace('/', '_').replace(' ', '_')
        if repo_safe_name.startswith('RoboCOIN_'):
            repo_safe_name = repo_safe_name[len('RoboCOIN_'):]
        stats_file = repo_wise_dir / f"{repo_safe_name}_stats.json"

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent = 2)

        print(f"\nSaved statistics to {stats_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Repo ID':<40} {'Episodes':<10} {'Steps':<12} {'State Dim':<12} {'Action Dim':<12}")
    print("-" * 80)

    for repo_id, stats in all_stats.items():
        state_dim = len(stats['observation.state']['names'])
        action_dim = len(stats['action']['names'])
        print(f"{repo_id:<40} {stats['num_episodes']:<10} {stats['num_steps']:<12} {state_dim:<12} {action_dim:<12}")

    # Global stats
    if not args.disable_global:
        print("\n" + "=" * 80)
        print("Computing global aggregated norm_stats")
        print("=" * 80)

        global_stats = compute_global_stats(all_stats)

        norm_stats_file = norm_stats_dir / "norm_stats.json"
        with open(norm_stats_file, 'w') as f:
            json.dump(global_stats, f, indent = 2)
        print(f"\nnorm_stats.json saved to {norm_stats_file}")


if __name__ == "__main__":
    main()

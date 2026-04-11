"""Compute real_hang normalization stats using the generic accumulator.

Reads HDF5 episodes from the dexterous robot data directories, constructs
state/action/eef arrays with gripper normalization and quat-to-euler
conversion, and feeds them to NormStatsAccumulator.

Usage:
    cd hdf5_to_tfds/
    python -m stats.real_hang_stats [--output PATH]
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dexterous_hang_config as cfg

from stats.generic_stats import FeatureSpec, NormStatsAccumulator


EEF_NAMES = [
    'left_eef_pos_x', 'left_eef_pos_y', 'left_eef_pos_z',
    'left_eef_ori_x', 'left_eef_ori_y', 'left_eef_ori_z',
    'right_eef_pos_x', 'right_eef_pos_y', 'right_eef_pos_z',
    'right_eef_ori_x', 'right_eef_ori_y', 'right_eef_ori_z',
]

STATE_NAMES = cfg._STATE_FEATURE_NAMES
ACTION_NAMES = cfg._ACTION_FEATURE_NAMES

SPECS = [
    FeatureSpec(key = 'observation.state', dim = cfg._MAX_STATE_DIM, names = STATE_NAMES),
    FeatureSpec(key = 'action', dim = cfg._MAX_ACTION_DIM, names = ACTION_NAMES),
    FeatureSpec(key = 'eef_sim_pose_state', dim = 12, names = EEF_NAMES),
    FeatureSpec(key = 'eef_sim_pose_action', dim = 12, names = EEF_NAMES),
]


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

    return np.array([roll, pitch, yaw], dtype = np.float64)


def _build_state_batch(left_joints, left_gripper, right_joints, right_gripper):
    """Build (N, D) state matching dexterous_hang_config._build_state."""
    n = len(left_gripper)
    if cfg._STATE_DIM == 16:
        state = np.zeros((n, 16), dtype = np.float64)
        state[:, :7] = left_joints
        state[:, 7] = left_gripper
        state[:, 8:15] = right_joints
        state[:, 15] = right_gripper
        return state
    state = np.zeros((n, 14), dtype = np.float64)
    state[:, 6] = left_gripper
    state[:, 13] = right_gripper
    return state


def _build_eef_batch(left_tcp, right_tcp):
    """Build (N, 12) EEF: left pos(3) + euler(3) + right pos(3) + euler(3)."""
    n = left_tcp.shape[0]
    eef = np.zeros((n, 12), dtype = np.float64)
    for i in range(n):
        left_euler = _quat_to_euler(left_tcp[i, 3 : 7])
        right_euler = _quat_to_euler(right_tcp[i, 3 : 7])
        eef[i, : 3] = left_tcp[i, : 3]
        eef[i, 3 : 6] = left_euler
        eef[i, 6 : 9] = right_tcp[i, : 3]
        eef[i, 9 : 12] = right_euler
    return eef


def extract_episode(filepath):
    """Read one HDF5 episode and return arrays for the accumulator."""
    with h5py.File(filepath, 'r') as f:
        total_frames = f['obses/state/left/gripper_pos'].shape[0]
        idx = np.arange(0, total_frames, 2)

        left_joints = f['obses/state/left/joint_qpos'][idx]
        left_tcp = f['obses/state/left/tcp_pose'][idx]
        right_joints = f['obses/state/right/joint_qpos'][idx]
        right_tcp = f['obses/state/right/tcp_pose'][idx]
        left_gripper = f['obses/state/left/gripper_pos'][idx, 0]
        right_gripper = f['obses/state/right/gripper_pos'][idx, 0]

    state = _build_state_batch(left_joints, left_gripper, right_joints, right_gripper)
    eef = _build_eef_batch(left_tcp, right_tcp)

    n = len(left_gripper)
    action = np.zeros((n, 14), dtype = np.float64)
    action[:, 6] = left_gripper
    action[:, 13] = right_gripper

    return {
        'observation.state': state,
        'action': action,
        'eef_sim_pose_state': eef,
        'eef_sim_pose_action': eef.copy(),
    }


def compute_total_steps(episodes):
    total_steps = 0
    for filepath in episodes:
        with h5py.File(filepath, 'r') as f:
            total_frames = f['obses/state/left/gripper_pos'].shape[0]
        total_steps += len(np.arange(0, total_frames, 2))
    return total_steps


def main():
    parser = argparse.ArgumentParser(description = 'Compute real_hang norm stats')
    parser.add_argument(
        '--output',
        default = '/data/group_data/rl/saksham3/hdf5/real_hang/norm_stats.json',
    )
    parser.add_argument('--quantile_keep', type = int, default = 2000)
    parser.add_argument('--diff_horizon', type = int, default = 50)
    parser.add_argument('--compute_buffer_max_size', action = 'store_true')
    args = parser.parse_args()

    episodes = cfg.get_episodes('train') + cfg.get_episodes('val')
    print(f'Processing {len(episodes)} episodes')

    quantile_keep = args.quantile_keep
    if args.compute_buffer_max_size:
        total_steps = compute_total_steps(episodes)
        quantile_keep = max(1, total_steps // 100)
        print(f'Computed quantile_keep = {quantile_keep} from total_steps = {total_steps}')

    accumulator = NormStatsAccumulator(SPECS, quantile_keep = quantile_keep, diff_horizon = args.diff_horizon)

    for i, ep_path in enumerate(episodes, start = 1):
        arrays = extract_episode(ep_path)
        accumulator.update(arrays)
        if i % 10 == 0 or i == len(episodes):
            print(f'  {i}/{len(episodes)} episodes ({accumulator.num_steps} steps)')

    result = accumulator.finalize()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok = True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent = 2)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()

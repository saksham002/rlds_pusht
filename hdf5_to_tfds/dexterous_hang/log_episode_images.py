"""Log subsampled episode images for subtask annotation.

For each episode: subsamples alternate frames (60fps -> 30fps),
then saves the right/top camera image every 5 steps.
"""
import io
import os
import sys

import h5py
import numpy as np
from PIL import Image

DATA_DIR = '/data/group_data/rl/dexterous_robot_data/real_hang_full_success_r5_hdf5'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'images', 'real_hang_full_success_r5_hdf5')

TRAIN_EPISODES = [3, 7, 8, 10, 11, 14, 16, 17, 18, 21, 22, 30, 34, 36, 37, 38, 41, 42, 44, 46]
VAL_EPISODES = [20, 23, 25, 33, 49]
ALL_EPISODES = sorted(set(TRAIN_EPISODES + VAL_EPISODES))

EVERY_N = 5


def log_episode(ep_idx):
    ep_path = os.path.join(DATA_DIR, f'episode_{ep_idx}.hdf5')
    ep_dir = os.path.join(OUTPUT_DIR, f'episode_{ep_idx}')
    os.makedirs(ep_dir, exist_ok = True)

    with h5py.File(ep_path, 'r') as f:
        total_frames = f['obses/state/left/gripper_pos'].shape[0]
        # Subsample alternate frames (60fps -> 30fps)
        frame_indices = list(range(0, total_frames, 2))
        images_top = f['obses/images/right/top']

        saved = 0
        for step_idx, fi in enumerate(frame_indices):
            if step_idx % EVERY_N != 0:
                continue
            raw = images_top[fi]
            img = Image.open(io.BytesIO(raw.tobytes()))
            out_path = os.path.join(ep_dir, f'step_{step_idx:04d}.jpg')
            img.save(out_path)
            saved += 1

    print(f'Episode {ep_idx}: {len(frame_indices)} steps, saved {saved} images to {ep_dir}')


if __name__ == '__main__':
    for ep in ALL_EPISODES:
        log_episode(ep)
    print(f'\nDone. Images saved to {OUTPUT_DIR}')

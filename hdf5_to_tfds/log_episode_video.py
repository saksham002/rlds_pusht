"""Log a video of an episode with subtask overlay from a selected camera.

Usage:
    cd hdf5_to_tfds/
    python log_episode_video.py [--output PATH] [--seed SEED]
"""
import argparse
import os

import cv2
import h5py
import imageio
import numpy as np

import dexterous_hang_config as cfg


def _get_subtask_at_step(step_idx, segments):
    for seg in segments:
        if seg['start_step'] <= step_idx <= seg['end_step']:
            return seg['subtask']
    return 'unknown'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default = 'episode_video.mp4')
    parser.add_argument('--seed', type = int, default = 86)
    parser.add_argument('--dataset', default = None)
    parser.add_argument('--episode', type = int, default = None)
    parser.add_argument('--camera', default = 'right/top')
    args = parser.parse_args()

    if args.dataset is not None or args.episode is not None:
        if args.dataset is None or args.episode is None:
            raise ValueError('Both --dataset and --episode must be provided together')
        ds_name = args.dataset
        ep_num = args.episode
    else:
        all_eps = cfg._TRAIN_EPISODES + cfg._VAL_EPISODES
        rng = np.random.RandomState(args.seed)
        ds_name, ep_num = all_eps[rng.randint(len(all_eps))]
    ep_path = os.path.join(cfg._DATA_ROOT, ds_name, f'episode_{ep_num}.hdf5')
    print(f'Episode: {ds_name}/episode_{ep_num}')

    ep_data = cfg._ANNOTATIONS['datasets'][ds_name][str(ep_num)]
    if ep_data['boundaries'] is None:
        raise ValueError(f'No boundaries available for {ds_name}/episode_{ep_num}')

    with h5py.File(ep_path, 'r') as f:
        total_frames = f[f'obses/images/{args.camera}'].shape[0]
        frame_indices = np.arange(0, total_frames, 2)
        ep_len = len(frame_indices)

        segments = cfg._get_subtask_segments(ds_name, ep_num, ep_len)

        frames = []
        for step_idx, fi in enumerate(frame_indices):
            jpg_bytes = f[f'obses/images/{args.camera}'][fi]
            img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype = np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            subtask = _get_subtask_at_step(step_idx, segments)
            label = f'[{step_idx}/{ep_len}] {subtask}'

            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(img)

    h, w = frames[0].shape[:2]
    h = h - (h % 16)
    w = w - (w % 16)
    frames = [f[:h, :w] for f in frames]

    imageio.mimsave(args.output, frames, format = "mp4", fps = 30, codec = "libx264", quality = 8)
    print(f'Wrote {args.output} ({len(frames)} frames, {len(frames)/30:.1f}s)')


if __name__ == '__main__':
    main()

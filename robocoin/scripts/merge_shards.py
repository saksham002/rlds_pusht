"""Merge per-(repo, worker) TFDS datasets into a single dataset.

Each worker writes its episodes to:
  <root>/<repo_suffix>/<worker_id>/robocoin/1.0.0/{dataset_info.json, features.json, *.tfrecord}

This script copies all TFRecord shards into a single flat output:
  <output>/robocoin/1.0.0/{dataset_info.json, features.json, *.tfrecord}

Usage:
    python scripts/merge_shards.py \
        --root gs://saksham-euw4/robocoin_local \
        --output gs://saksham-euw4/robocoin_merged \
        --num_workers 32 \
        --repo_list repos.txt
"""
import argparse
import json
import os
import sys

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required = True, help = 'Root directory with per-(repo, worker) outputs')
    parser.add_argument('--output', required = True, help = 'Output directory for merged dataset')
    parser.add_argument('--num_workers', type = int, required = True)
    parser.add_argument('--repo_list', default = 'repos.txt')
    parser.add_argument('--splits', nargs = '+', default = ['train', 'val'], help = 'Split names to merge')
    parser.add_argument('--overwrite', action = 'store_true', help = 'Overwrite existing shards')
    args = parser.parse_args()

    with open(args.repo_list, 'r') as f:
        repo_suffixes = [line.strip() for line in f if line.strip()]

    dataset_name = 'robocoin'
    version = '1.0.0'
    tfds_subdir = f'{dataset_name}/{version}'

    out_dir = os.path.join(args.output, tfds_subdir)
    tf.io.gfile.makedirs(out_dir)

    features_json_src = None
    description = None
    all_split_infos = []

    for split in args.splits:
        print(f"\n=== Merging split: {split} ===")
        all_shard_paths = []
        all_shard_lengths = []
        total_bytes = 0
        repos_found = 0

        for repo_suffix in repo_suffixes:
            repo_shards = []
            for worker_id in range(args.num_workers):
                worker_dir = os.path.join(args.root, repo_suffix, str(worker_id), tfds_subdir)
                info_path = os.path.join(worker_dir, 'dataset_info.json')

                if not tf.io.gfile.exists(info_path):
                    continue

                with tf.io.gfile.GFile(info_path, 'r') as fh:
                    info = json.loads(fh.read())

                if features_json_src is None:
                    feat_path = os.path.join(worker_dir, 'features.json')
                    if tf.io.gfile.exists(feat_path):
                        features_json_src = feat_path

                if description is None:
                    description = info.get('description', '')

                for split_info in info['splits']:
                    if split_info['name'] != split:
                        continue
                    total_bytes += int(split_info['numBytes'])
                    n_shards = len(split_info['shardLengths'])
                    for shard_idx in range(n_shards):
                        shard_file = f'{dataset_name}-{split}.tfrecord-{shard_idx:05d}-of-{n_shards:05d}'
                        shard_path = os.path.join(worker_dir, shard_file)
                        repo_shards.append(shard_path)
                        all_shard_lengths.append(split_info['shardLengths'][shard_idx])

            if repo_shards:
                repos_found += 1
                all_shard_paths.extend(repo_shards)

        total_shards = len(all_shard_paths)
        total_examples = sum(int(s) for s in all_shard_lengths)
        print(f"Found {repos_found} repos, {total_shards} shards, {total_examples} examples, {total_bytes / 1e9:.1f} GB")

        print(f"Copying {total_shards} shards to {out_dir}...")
        for new_idx, src_path in enumerate(all_shard_paths):
            dst_file = f'{dataset_name}-{split}.tfrecord-{new_idx:05d}-of-{total_shards:05d}'
            dst_path = os.path.join(out_dir, dst_file)
            if not args.overwrite and tf.io.gfile.exists(dst_path):
                print(f"  [SKIP] {dst_file} (already exists)")
                continue
            tf.io.gfile.copy(src_path, dst_path, overwrite = args.overwrite)
            if (new_idx + 1) % 100 == 0:
                print(f"  Copied {new_idx + 1}/{total_shards}")

        print(f"Copied all {total_shards} shards.")

        all_split_infos.append({
            'filepathTemplate': '{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}',
            'name': split,
            'numBytes': str(total_bytes),
            'shardLengths': all_shard_lengths,
        })

    # Copy features.json
    if features_json_src:
        tf.io.gfile.copy(features_json_src, os.path.join(out_dir, 'features.json'), overwrite = True)
        print("\nCopied features.json")

    # Write merged dataset_info.json with all splits
    merged_info = {
        'description': description or '',
        'fileFormat': 'tfrecord',
        'moduleName': 'robocoin.robocoin_dataset_builder',
        'name': dataset_name,
        'releaseNotes': {'1.0.0': 'Initial release covering all RoboCOIN datasets.'},
        'splits': all_split_infos,
        'version': version,
    }

    info_out = os.path.join(out_dir, 'dataset_info.json')
    with tf.io.gfile.GFile(info_out, 'w') as fh:
        fh.write(json.dumps(merged_info, indent = 2))

    for si in all_split_infos:
        n_ex = sum(int(s) for s in si['shardLengths'])
        print(f"  {si['name']}: {n_ex} examples, {len(si['shardLengths'])} shards")
    print("Wrote merged dataset_info.json")

    print("\nDone.")


if __name__ == '__main__':
    main()

"""Merge per-worker TFDS datasets into a single dataset.

Each worker writes its episodes to:
  <root>/<worker_id>/<dataset_name>/<version>/{dataset_info.json, features.json, *.tfrecord}

This script copies all TFRecord shards into a single flat output:
  <output>/<dataset_name>/<version>/{dataset_info.json, features.json, *.tfrecord}

Usage:
    python scripts/merge_shards.py \\
        --root gs://my-bucket/my_dataset_workers \\
        --output gs://my-bucket/my_dataset_merged \\
        --num_workers 16 \\
        --dataset_name my_dataset \\
        --dataset_version 1.0.0
"""
import argparse
import concurrent.futures
import json
import os
import sys

import tensorflow as tf


def _marker_path(root, worker_id, dataset_name, version):
    return os.path.join(root, str(worker_id), dataset_name, version, 'dataset_info.json')


def run_dry_run(args):
    """Print per-worker completion table, then exit."""
    print(f'Checking {args.num_workers} worker markers...')

    def check(worker_id):
        path = _marker_path(args.root, worker_id, args.dataset_name, args.dataset_version)
        try:
            return tf.io.gfile.exists(path)
        except Exception:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers = 64) as ex:
        results = list(ex.map(check, range(args.num_workers)))

    print()
    for w, done in enumerate(results):
        flag = '✓' if done else '✗'
        status = 'done' if done else 'missing'
        print(f'W{w:<4} {status:<10} {flag}')

    total_done    = sum(results)
    total_missing = args.num_workers - total_done
    print()
    print(f'Total: {args.num_workers} workers, {total_done} done, {total_missing} missing')

    if total_missing == 0:
        print('\nAll workers complete — ready to merge.')
    else:
        print('\nBuild incomplete — relaunch build.sh before merging.')
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',            required = True,  help = 'Root directory with per-worker outputs')
    parser.add_argument('--output',          required = True,  help = 'Output directory for merged dataset')
    parser.add_argument('--num_workers',     type = int, required = True)
    parser.add_argument('--dataset_name',    required = True)
    parser.add_argument('--dataset_version', required = True)
    parser.add_argument('--splits',          nargs = '+', default = ['train', 'val'], help = 'Split names to merge')
    parser.add_argument('--overwrite',       action = 'store_true', help = 'Overwrite existing shards')
    parser.add_argument('--dry_run',         action = 'store_true', help = 'Print completion table and exit')
    parser.add_argument('--copy_workers',    type = int, default = 32, help = 'Parallel threads for shard copying')
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(args)

    dataset_name = args.dataset_name
    version      = args.dataset_version
    tfds_subdir  = f'{dataset_name}/{version}'

    out_dir = os.path.join(args.output, tfds_subdir)
    tf.io.gfile.makedirs(out_dir)

    features_json_src = None
    description       = None
    all_split_infos   = []

    for split in args.splits:
        print(f'\n=== Merging split: {split} ===')
        all_shard_paths   = []
        all_shard_lengths = []
        total_bytes       = 0
        workers_found     = 0

        for worker_id in range(args.num_workers):
            worker_dir = os.path.join(args.root, str(worker_id), tfds_subdir)
            info_path  = os.path.join(worker_dir, 'dataset_info.json')

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
                    all_shard_paths.append(os.path.join(worker_dir, shard_file))
                    all_shard_lengths.append(split_info['shardLengths'][shard_idx])
                workers_found += 1

        total_shards   = len(all_shard_paths)
        total_examples = sum(int(s) for s in all_shard_lengths)
        print(f'Found {workers_found} workers, {total_shards} shards, {total_examples} examples, {total_bytes / 1e9:.1f} GB')

        if total_shards == 0:
            print(f'No shards found for split {split!r}, skipping.')
            continue

        print(f'Copying {total_shards} shards to {out_dir}...')

        existing_dst = set()
        if not args.overwrite:
            try:
                existing_dst = set(tf.io.gfile.listdir(out_dir))
            except tf.errors.NotFoundError:
                pass

        def copy_shard(item):
            new_idx, src_path = item
            dst_file = f'{dataset_name}-{split}.tfrecord-{new_idx:05d}-of-{total_shards:05d}'
            if not args.overwrite and dst_file in existing_dst:
                return False
            dst_path = os.path.join(out_dir, dst_file)
            tf.io.gfile.copy(src_path, dst_path, overwrite = args.overwrite)
            return True

        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers = args.copy_workers) as ex:
            for result in concurrent.futures.as_completed(
                ex.submit(copy_shard, item) for item in enumerate(all_shard_paths)
            ):
                done += 1
                if done % 100 == 0:
                    print(f'  Copied {done}/{total_shards}')

        print(f'Copied all {total_shards} shards.')

        all_split_infos.append({
            'filepathTemplate': '{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}',
            'name':         split,
            'numBytes':     str(total_bytes),
            'shardLengths': all_shard_lengths,
        })

    if features_json_src:
        tf.io.gfile.copy(features_json_src, os.path.join(out_dir, 'features.json'), overwrite = True)
        print('\nCopied features.json')

    merged_info = {
        'description':   description or '',
        'fileFormat':    'tfrecord',
        'moduleName':    f'{dataset_name}.{dataset_name}_dataset_builder',
        'name':          dataset_name,
        'releaseNotes':  {version: 'Initial release.'},
        'splits':        all_split_infos,
        'version':       version,
    }

    info_out = os.path.join(out_dir, 'dataset_info.json')
    with tf.io.gfile.GFile(info_out, 'w') as fh:
        fh.write(json.dumps(merged_info, indent = 2))

    for si in all_split_infos:
        n_ex = sum(int(s) for s in si['shardLengths'])
        print(f"  {si['name']}: {n_ex} examples, {len(si['shardLengths'])} shards")
    print('Wrote merged dataset_info.json')
    print('\nDone.')


if __name__ == '__main__':
    main()

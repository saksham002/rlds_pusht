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
import concurrent.futures
import json
import os
import sys

import tensorflow as tf


MIN_STEPS_PER_WORKER = 5000


def _get_episode_lengths(repo_suffix):
    """Return episode lengths from local download if available, else HF hub."""
    local = f"/data/group_data/rl/saksham3/robocoin/RoboCOIN/{repo_suffix}/meta/episodes.jsonl"
    if os.path.exists(local):
        with open(local) as f:
            return [json.loads(l)['length'] for l in f if l.strip()]
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(
        repo_id = f"RoboCOIN/{repo_suffix}",
        filename = "meta/episodes.jsonl",
        repo_type = "dataset",
    )
    with open(p) as f:
        return [json.loads(l)['length'] for l in f if l.strip()]


def _effective_workers(lengths, num_workers):
    total = sum(lengths)
    return min(num_workers, max(1, (total + MIN_STEPS_PER_WORKER - 1) // MIN_STEPS_PER_WORKER))


def run_dry_run(args, repo_suffixes):
    """Print per-worker involvement and completion table, then exit."""
    print(f"Fetching episode lengths for {len(repo_suffixes)} repos...")
    with concurrent.futures.ThreadPoolExecutor(max_workers = 16) as ex:
        all_lengths = list(ex.map(_get_episode_lengths, repo_suffixes))

    repo_eff = {s: _effective_workers(l, args.num_workers) for s, l in zip(repo_suffixes, all_lengths)}

    # Check GCS markers in parallel for all (repo, worker) pairs
    pairs = [(s, w) for s in repo_suffixes for w in range(args.num_workers) if w < repo_eff[s]]

    def check(pair):
        s, w = pair
        path = os.path.join(args.root, s, str(w), 'robocoin', '1.0.0', 'dataset_info.json')
        try:
            return tf.io.gfile.exists(path)
        except Exception:
            return False

    print(f"Checking {len(pairs)} (repo, worker) markers on GCS...")
    with concurrent.futures.ThreadPoolExecutor(max_workers = 64) as ex:
        results = list(ex.map(check, pairs))

    marker_map = {pair: res for pair, res in zip(pairs, results)}

    print()
    print(f"{'W':<5} {'Involved':>10} {'Done':>8} {'Missing':>10}  Status")
    print('-' * 48)
    all_complete = True
    for w in range(args.num_workers):
        involved = [s for s in repo_suffixes if w < repo_eff[s]]
        done = sum(1 for s in involved if marker_map.get((s, w), False))
        missing = len(involved) - done
        ok = missing == 0
        if not ok:
            all_complete = False
        flag = '✓' if ok else f'✗ {missing} missing'
        print(f"W{w:<4} {len(involved):>10} {done:>8} {missing:>10}  {flag}")

    total_pairs = len(pairs)
    total_done = sum(results)
    total_missing = total_pairs - total_done
    print()
    print(f"Total pairs: {total_pairs}  Done: {total_done}  Missing: {total_missing}")
    if all_complete:
        print("\nAll workers complete — ready to merge.")
    else:
        print("\nBuild incomplete — relaunch build_local.sh before merging.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required = True, help = 'Root directory with per-(repo, worker) outputs')
    parser.add_argument('--output', required = True, help = 'Output directory for merged dataset')
    parser.add_argument('--num_workers', type = int, required = True)
    parser.add_argument('--repo_list', default = 'repos.txt')
    parser.add_argument('--splits', nargs = '+', default = ['train', 'val'], help = 'Split names to merge')
    parser.add_argument('--overwrite', action = 'store_true', help = 'Overwrite existing shards')
    parser.add_argument('--dry_run', action = 'store_true', help = 'Print per-worker completion table and exit without merging')
    args = parser.parse_args()

    with open(args.repo_list, 'r') as f:
        repo_suffixes = [line.strip() for line in f if line.strip()]

    if args.dry_run:
        run_dry_run(args, repo_suffixes)

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

"""End-to-end pipeline: build → merge → scan → fix.

Local mode (2 workers, for testing):
    python scripts/pipeline.py \\
        --config example_config.py \\
        --data_root /tmp/test_workers \\
        --output /tmp/test_merged \\
        --num_workers 2 \\
        --dataset_name pusht_image \\
        --dataset_version 1.0.0 \\
        --framework_dir /path/to/slurm_rlds \\
        --mode local

SLURM mode:
    python scripts/pipeline.py \\
        --config /path/to/config.py \\
        --data_root gs://bucket/workers \\
        --output gs://bucket/merged \\
        --num_workers 16 \\
        --dataset_name my_dataset \\
        --dataset_version 1.0.0 \\
        --framework_dir /path/to/slurm_rlds \\
        --mode slurm \\
        --slurm_partition preempt

Skip flags:
    --skip_build   assume all workers already done, go straight to merge
    --skip_merge   assume merge already done, read shard_map.json and go straight to scan
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile

import tensorflow as tf


# ─── Marker helpers ───────────────────────────────────────────────────────────

def _marker_path(root, worker_id, dataset_name, version):
    return os.path.join(root, str(worker_id), dataset_name, version, 'dataset_info.json')


def _check_markers(root, num_workers, dataset_name, version):
    """Return list of worker IDs with missing completion markers."""
    def _exists(w):
        try:
            return w, tf.io.gfile.exists(_marker_path(root, w, dataset_name, version))
        except Exception:
            return w, False

    with concurrent.futures.ThreadPoolExecutor(max_workers = 64) as ex:
        results = list(ex.map(_exists, range(num_workers)))
    return [w for w, exists in results if not exists]


# ─── Phase 1: Build ───────────────────────────────────────────────────────────

def _array_spec(worker_ids):
    ids = sorted(worker_ids)
    if ids == list(range(ids[0], ids[-1] + 1)):
        return f'{ids[0]}-{ids[-1]}'
    return ','.join(str(w) for w in ids)


def _generate_slurm_script(args, array_spec):
    log_dir      = os.path.abspath(args.log_dir)
    framework_dir = os.path.abspath(args.framework_dir)
    config_path  = os.path.abspath(args.config)
    return f"""#!/bin/bash
#SBATCH --job-name=build_rlds
#SBATCH --array={array_spec}
#SBATCH --partition={args.slurm_partition}
#SBATCH --requeue
#SBATCH --time={args.slurm_time}
#SBATCH --cpus-per-task={args.slurm_cpus}
#SBATCH --mem={args.slurm_mem}
#SBATCH --gres={args.slurm_gres}
#SBATCH --output={log_dir}/build_%a.out
#SBATCH --error={log_dir}/build_%a.err

source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlds

export PYTHONUNBUFFERED=1
export CURL_CA_BUNDLE="/data/user_data/saksham3/conda-envs/rlds/lib/python3.12/site-packages/certifi/cacert.pem"
export SSL_CERT_FILE="$CURL_CA_BUNDLE"

cd "{framework_dir}"
python -u -m framework.runner \\
    --config "{config_path}" \\
    --data_dir "{args.data_root}/$SLURM_ARRAY_TASK_ID" \\
    --worker_id "$SLURM_ARRAY_TASK_ID" \\
    --num_workers {args.num_workers}
"""


def _run_slurm(args, worker_ids):
    spec   = _array_spec(worker_ids)
    script = _generate_slurm_script(args, spec)

    with tempfile.NamedTemporaryFile(mode = 'w', suffix = '.sh', delete = False, prefix = 'slurm_rlds_') as f:
        f.write(script)
        script_path = f.name
    os.chmod(script_path, 0o755)

    print(f'  Submitting SLURM array [{spec}] (blocking until done)...')
    try:
        subprocess.run(['sbatch', '--wait', script_path], check = False)
    finally:
        os.unlink(script_path)


def _run_local(args, worker_ids):
    framework_dir = os.path.abspath(args.framework_dir)
    config_path   = os.path.abspath(args.config)
    for w in worker_ids:
        data_dir = os.path.join(args.data_root, str(w))
        print(f'  Worker {w}/{args.num_workers}...')
        subprocess.run(
            [
                sys.executable, '-m', 'framework.runner',
                '--config',      config_path,
                '--data_dir',    data_dir,
                '--worker_id',   str(w),
                '--num_workers', str(args.num_workers),
            ],
            cwd = framework_dir,
            check = False,
        )


def phase_build(args):
    print('\n=== Phase 1: Build ===')
    missing = _check_markers(args.data_root, args.num_workers, args.dataset_name, args.dataset_version)

    if not missing:
        print('  All workers already complete.')
        return []

    print(f'  {len(missing)} workers to build.')
    _dispatch(args, missing)

    missing = _check_markers(args.data_root, args.num_workers, args.dataset_name, args.dataset_version)
    if missing:
        print(f'  {len(missing)} workers still missing — retrying once (skipping completed workers)...')
        _dispatch(args, missing)
        missing = _check_markers(args.data_root, args.num_workers, args.dataset_name, args.dataset_version)

    if missing:
        print(f'  [WARNING] {len(missing)} workers still missing after retry: {missing}')
    else:
        print('  All workers complete.')
    return missing


def _dispatch(args, worker_ids):
    if args.mode == 'slurm':
        _run_slurm(args, worker_ids)
    else:
        _run_local(args, worker_ids)


# ─── Phase 2: Merge ───────────────────────────────────────────────────────────

def phase_merge(args):
    print('\n=== Phase 2: Merge ===')
    dataset_name = args.dataset_name
    version      = args.dataset_version
    tfds_subdir  = f'{dataset_name}/{version}'

    out_dir = os.path.join(args.output, tfds_subdir)
    tf.io.gfile.makedirs(out_dir)

    features_json_src = None
    description       = None
    all_split_infos   = []
    # shard_map[split][str(final_idx)] = {"worker_id": W, "worker_shard_idx": S, "n_worker_shards": N}
    shard_map = {}

    for split in args.splits:
        print(f'\n  Merging split: {split!r}')
        all_shard_paths   = []
        all_shard_lengths = []
        shard_sources     = []   # (worker_id, worker_shard_idx, n_worker_shards)
        total_bytes       = 0
        workers_found     = 0

        for worker_id in range(args.num_workers):
            worker_dir = os.path.join(args.data_root, str(worker_id), tfds_subdir)
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
                    shard_sources.append((worker_id, shard_idx, n_shards))
                workers_found += 1

        total_shards   = len(all_shard_paths)
        total_examples = sum(int(s) for s in all_shard_lengths)
        print(f'  {workers_found} workers, {total_shards} shards, {total_examples} examples, {total_bytes / 1e9:.1f} GB')

        if total_shards == 0:
            print(f'  No shards found for {split!r}, skipping.')
            continue

        existing_dst = set()
        if not args.overwrite:
            try:
                existing_dst = set(tf.io.gfile.listdir(out_dir))
            except tf.errors.NotFoundError:
                pass

        # Copy shards, renumbering them sequentially in the merged output.
        # Capture loop-local vars to avoid closure issues.
        def _make_copy_fn(split_name, n_total, dst_set, do_overwrite):
            def copy_shard(item):
                new_idx, src_path = item
                dst_file = f'{dataset_name}-{split_name}.tfrecord-{new_idx:05d}-of-{n_total:05d}'
                if not do_overwrite and dst_file in dst_set:
                    return
                tf.io.gfile.copy(src_path, os.path.join(out_dir, dst_file), overwrite = do_overwrite)
            return copy_shard

        copy_fn = _make_copy_fn(split, total_shards, existing_dst, args.overwrite)
        done_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers = args.copy_workers) as ex:
            for _ in concurrent.futures.as_completed(
                ex.submit(copy_fn, item) for item in enumerate(all_shard_paths)
            ):
                done_count += 1
                if done_count % 100 == 0:
                    print(f'    Copied {done_count}/{total_shards}')
        print(f'  Copied {total_shards} shards.')

        shard_map[split] = {
            str(new_idx): {
                'worker_id':        w_id,
                'worker_shard_idx': w_shard,
                'n_worker_shards':  n,
            }
            for new_idx, (w_id, w_shard, n) in enumerate(shard_sources)
        }

        all_split_infos.append({
            'filepathTemplate': '{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}',
            'name':         split,
            'numBytes':     str(total_bytes),
            'shardLengths': all_shard_lengths,
        })

    if features_json_src:
        tf.io.gfile.copy(features_json_src, os.path.join(out_dir, 'features.json'), overwrite = True)

    merged_info = {
        'description':  description or '',
        'fileFormat':   'tfrecord',
        'moduleName':   f'{dataset_name}.{dataset_name}_dataset_builder',
        'name':         dataset_name,
        'releaseNotes': {version: 'Initial release.'},
        'splits':       all_split_infos,
        'version':      version,
    }
    with tf.io.gfile.GFile(os.path.join(out_dir, 'dataset_info.json'), 'w') as fh:
        fh.write(json.dumps(merged_info, indent = 2))

    shard_map_path = os.path.join(out_dir, 'shard_map.json')
    with tf.io.gfile.GFile(shard_map_path, 'w') as fh:
        fh.write(json.dumps(shard_map, indent = 2))

    for si in all_split_infos:
        n_ex = sum(int(s) for s in si['shardLengths'])
        print(f"  {si['name']}: {n_ex} examples, {len(si['shardLengths'])} shards")
    print(f'  Wrote shard_map.json')

    return all_split_infos, shard_map, out_dir


# ─── Phase 3: Scan ────────────────────────────────────────────────────────────

def _scan_shard(item):
    """Return (idx, status, msg) where status is 'OK', 'BAD', or 'ERROR'.

    BAD  — DataLossError or record count mismatch (corrupt data).
    ERROR — any other exception, e.g. DNS/network failure (transient).
    Both BAD and ERROR shards are recopied in phase_fix.
    """
    idx, path, expected = item
    count = 0
    try:
        for _ in tf.data.TFRecordDataset(path, buffer_size = 64 * 1024 * 1024):
            count += 1
        if count != expected:
            return idx, 'BAD', f'count={count} expected={expected}'
        return idx, 'OK', None
    except tf.errors.DataLossError as e:
        return idx, 'BAD', f'DataLossError after {count} records: {e}'
    except Exception as e:
        return idx, 'ERROR', str(e)


def phase_scan(args, out_dir, split_infos):
    print('\n=== Phase 3: Scan ===')
    dataset_name = args.dataset_name
    # {split: [(idx, status, msg), ...]} for all non-OK shards
    bad_map = {}

    for si in split_infos:
        split         = si['name']
        shard_lengths = si['shardLengths']
        n_shards      = len(shard_lengths)
        print(f'  Scanning {n_shards} shards for split {split!r}...', flush = True)

        scan_items = [
            (i, os.path.join(out_dir, f'{dataset_name}-{split}.tfrecord-{i:05d}-of-{n_shards:05d}'), int(length))
            for i, length in enumerate(shard_lengths)
        ]

        bad = []
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers = args.copy_workers) as ex:
            for idx, status, msg in ex.map(_scan_shard, scan_items):
                done += 1
                if status != 'OK':
                    bad.append((idx, status, msg))
                    print(f'  {idx} {status}: {msg}', flush = True)
                if done % 500 == 0:
                    print(f'  {done}/{n_shards} checked, {len(bad)} bad so far', flush = True)

        bad_map[split] = bad
        if bad:
            print(f'  [WARNING] {split!r}: {len(bad)} bad shards ({sum(s == "ERROR" for _, s, _ in bad)} ERROR, {sum(s == "BAD" for _, s, _ in bad)} BAD)')
        else:
            print(f'  All {n_shards} shards healthy.')

    return bad_map


# ─── Phase 4: Fix ─────────────────────────────────────────────────────────────

def phase_fix(args, out_dir, bad_map, shard_map, split_infos):
    print('\n=== Phase 4: Fix ===')
    dataset_name = args.dataset_name
    version      = args.dataset_version
    tfds_subdir  = f'{dataset_name}/{version}'

    if not any(bad_map.values()):
        print('  No bad shards — nothing to fix.')
        return

    repaired = 0
    failed   = 0

    for split, bad_shards in bad_map.items():
        if not bad_shards:
            continue
        split_map    = shard_map[split]
        total_shards = len(split_map)
        print(f'  Fixing {len(bad_shards)} shards for split {split!r} (serial, up to 5 attempts each):')

        for final_idx, status, reason in sorted(bad_shards, key = lambda x: x[0]):
            entry           = split_map[str(final_idx)]
            worker_id       = entry['worker_id']
            worker_shard    = entry['worker_shard_idx']
            n_worker_shards = entry['n_worker_shards']

            src_file = f'{dataset_name}-{split}.tfrecord-{worker_shard:05d}-of-{n_worker_shards:05d}'
            src_path = os.path.join(args.data_root, str(worker_id), tfds_subdir, src_file)
            dst_file = f'{dataset_name}-{split}.tfrecord-{final_idx:05d}-of-{total_shards:05d}'
            dst_path = os.path.join(out_dir, dst_file)

            print(f'    shard {final_idx:>5} [{status}]: W{worker_id}[{worker_shard}/{n_worker_shards}] reason={reason}', flush = True)
            for attempt in range(5):
                try:
                    tf.io.gfile.copy(src_path, dst_path, overwrite = True)
                    print(f'      attempt {attempt + 1}/5 OK', flush = True)
                    repaired += 1
                    break
                except Exception as e:
                    print(f'      attempt {attempt + 1}/5 failed: {e}', flush = True)
                    if attempt == 4:
                        failed += 1

    print(f'\n  Repaired: {repaired}  Failed: {failed}')

    # Re-scan only the fixed shards to confirm recovery.
    print('\n  Verifying fixed shards...')
    si_by_split = {si['name']: si for si in split_infos}
    all_ok      = True
    for split, bad_shards in bad_map.items():
        if not bad_shards:
            continue
        si       = si_by_split[split]
        n_shards = len(si['shardLengths'])
        for final_idx, _, _ in bad_shards:
            expected   = int(si['shardLengths'][final_idx])
            shard_file = f'{dataset_name}-{split}.tfrecord-{final_idx:05d}-of-{n_shards:05d}'
            _, vstatus, vmsg = _scan_shard((final_idx, os.path.join(out_dir, shard_file), expected))
            flag = '✓' if vstatus == 'OK' else f'✗ {vstatus}: {vmsg}'
            print(f'    {split} shard {final_idx}: {flag}', flush = True)
            if vstatus != 'OK':
                all_ok = False

    if not all_ok:
        print('\n[ERROR] Some shards still bad after fix — manual intervention required.')
        sys.exit(1)
    print('  All fixed shards verified healthy.')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description = 'slurm_rlds pipeline: build → merge → scan → fix'
    )
    parser.add_argument('--config',          required = True,  help = 'Path to user config .py')
    parser.add_argument('--data_root',       required = True,  help = 'Root dir for per-worker outputs')
    parser.add_argument('--output',          required = True,  help = 'Output dir for merged dataset')
    parser.add_argument('--num_workers',     type = int, required = True)
    parser.add_argument('--dataset_name',    required = True)
    parser.add_argument('--dataset_version', required = True)
    parser.add_argument('--framework_dir',   required = True,  help = 'Path to slurm_rlds/ directory')
    parser.add_argument('--splits',          nargs = '+', default = ['train', 'val'])
    parser.add_argument('--mode',            choices = ['local', 'slurm'],
                        default = 'slurm' if shutil.which('sbatch') else 'local',
                        help = 'Build mode (default: slurm if sbatch is available, else local)')
    parser.add_argument('--overwrite',       action = 'store_true',
                        help = 'Overwrite existing merged shards')
    parser.add_argument('--copy_workers',    type = int, default = 32,
                        help = 'Parallel threads for shard copy and scan')
    parser.add_argument('--log_dir',         default = 'logs',
                        help = 'Directory for SLURM worker logs')
    parser.add_argument('--skip_build',      action = 'store_true',
                        help = 'Skip build phase (assume all workers already done)')
    parser.add_argument('--skip_merge',      action = 'store_true',
                        help = 'Skip merge phase (read existing shard_map.json from output)')
    parser.add_argument('--skip_scan',       action = 'store_true',
                        help = 'Skip scan + fix phases (merge only; no shard integrity check/repair)')
    # SLURM-specific
    parser.add_argument('--slurm_partition', default = 'preempt')
    parser.add_argument('--slurm_time',      default = '48:00:00')
    parser.add_argument('--slurm_cpus',      type = int, default = 8)
    parser.add_argument('--slurm_mem',       default = '64G')
    parser.add_argument('--slurm_gres',      default = 'gpu:1')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok = True)

    dataset_name = args.dataset_name
    version      = args.dataset_version
    out_dir      = os.path.join(args.output, f'{dataset_name}/{version}')

    # Phase 1
    if not args.skip_build:
        phase_build(args)
    else:
        print('\n=== Phase 1: Build [skipped] ===')

    # Phase 2
    if not args.skip_merge:
        split_infos, shard_map, out_dir = phase_merge(args)
    else:
        print('\n=== Phase 2: Merge [skipped] ===')
        shard_map_path = os.path.join(out_dir, 'shard_map.json')
        with tf.io.gfile.GFile(shard_map_path, 'r') as fh:
            shard_map = json.loads(fh.read())
        info_path = os.path.join(out_dir, 'dataset_info.json')
        with tf.io.gfile.GFile(info_path, 'r') as fh:
            split_infos = json.loads(fh.read())['splits']
        print(f'  Loaded shard_map.json and dataset_info.json from {out_dir}')

    if not split_infos:
        print('\nNo splits produced — nothing to scan. Done.')
        return

    if args.skip_scan:
        print('\n=== Phase 3: Scan + Phase 4: Fix [skipped] ===')
        print('\n=== Pipeline complete ===')
        return

    # Phase 3
    bad_map = phase_scan(args, out_dir, split_infos)

    # Phase 4
    phase_fix(args, out_dir, bad_map, shard_map, split_infos)

    print('\n=== Pipeline complete ===')


if __name__ == '__main__':
    main()

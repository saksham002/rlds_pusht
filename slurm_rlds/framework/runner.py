"""Worker entry point for the slurm_rlds framework.

Usage:
    python -m framework.runner \\
        --config /path/to/my_dataset_config.py \\
        --data_dir gs://my-bucket/workers/0 \\
        --worker_id 0 \\
        --num_workers 16
"""
import argparse
import importlib.util
import os
import sys

import tensorflow as tf
import tensorflow_datasets as tfds

from framework.builder import GenericBuilder


SPLITS = ['train', 'val']


def _load_config(config_path):
    spec = importlib.util.spec_from_file_location('user_config', config_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      required = True, help = 'Path to user config .py')
    parser.add_argument('--data_dir',    required = True, help = 'Output directory for this worker')
    parser.add_argument('--worker_id',   type = int, required = True)
    parser.add_argument('--num_workers', type = int, required = True)
    args = parser.parse_args()

    config = _load_config(args.config)

    marker = os.path.join(
        args.data_dir,
        config.DATASET_NAME,
        config.DATASET_VERSION,
        'dataset_info.json',
    )

    # Check completion marker
    while True:
        try:
            exists = tf.io.gfile.exists(marker)
            break
        except Exception:
            import time; time.sleep(5)

    if exists:
        print(f'[SKIP] worker {args.worker_id} already done ({marker})')
        sys.exit(0)

    # Build per-split episode slices for this worker
    worker_episodes = {}
    for split in SPLITS:
        all_eps = config.get_episodes(split)
        if not all_eps:
            continue
        worker_slice = all_eps[args.worker_id :: args.num_workers]
        if worker_slice:
            worker_episodes[split] = worker_slice

    if not worker_episodes:
        print(f'[SKIP] worker {args.worker_id} has no episodes to process')
        sys.exit(0)

    for split, eps in worker_episodes.items():
        print(f'  {split}: {len(eps)} episodes')

    # Dynamically create a named builder class (TFDS uses the class name for the dataset name).
    # __module__ must be set explicitly: type() under ABCMeta yields __module__='abc',
    # which breaks TFDS get_metadata() (it lists the defining module's package dir).
    cls_name = ''.join(w.capitalize() for w in config.DATASET_NAME.split('_'))
    BuilderCls = type(cls_name, (GenericBuilder,), {
        '__module__':      GenericBuilder.__module__,
        'VERSION':         tfds.core.Version(config.DATASET_VERSION),
        'USER_CONFIG':     config,
        'WORKER_EPISODES': worker_episodes,
    })

    builder = BuilderCls(data_dir = args.data_dir)
    builder.download_and_prepare(download_dir = args.data_dir)

    print(f'[SUCCESS] worker {args.worker_id} done → {args.data_dir}')


if __name__ == '__main__':
    main()

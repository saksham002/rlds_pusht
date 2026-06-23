# slurm_rlds — Modular SLURM Worker Framework for TFDS Builds

## Purpose

`slurm_rlds` lets you distribute any serial `GeneratorBasedBuilder` across a SLURM array job without modifying the builder itself. You write a single config file (episode list + feature schema + one-episode parser), and the framework handles:

- Slicing episodes round-robin across workers
- Per-worker GCS output with completion markers (so preempted workers skip re-done work on requeue)
- A full pipeline script: **build → merge → scan → fix** in one command

It is useful when your dataset is large enough that a single node would take too long, and episodes are stored as separate files so parallel I/O actually helps. For CPU-bound processing on a single large machine, the standard Beam approach (`tfds build --beam_pipeline_options=...`) has less overhead.

---

## Layout

```
slurm_rlds/
├── framework/
│   ├── __init__.py
│   ├── builder.py       — generic GeneratorBasedBuilder (never edit this)
│   └── runner.py        — per-worker entry point (never edit this)
├── scripts/
│   ├── build.sh         — SLURM array job template (fill in the top section)
│   ├── merge_shards.py  — standalone merge tool
│   └── pipeline.py      — end-to-end orchestrator (build → merge → scan → fix)
└── example_config.py    — working example for the pusht_image dataset
```

---

## Usage

### Step 1 — Write a config file

Copy `example_config.py` and implement four items. Everything else is handled by the framework.

```python
# my_dataset_config.py

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# 1. Dataset identity
DATASET_NAME    = 'my_dataset'
DATASET_VERSION = '1.0.0'

# 2. Feature schema — same as what you would put in _info() of a serial builder
def get_features():
    return tfds.features.FeaturesDict({
        'steps': tfds.features.Dataset({
            'image':  tfds.features.Image(shape = (256, 256, 3)),
            'action': tfds.features.Tensor(shape = (7,), dtype = np.float32),
        }),
        'episode_metadata': tfds.features.FeaturesDict({
            'file_path': tfds.features.Text(),
        }),
    })

# 3. Episode list — return a *deterministically ordered* list for each split.
#    The framework slices this list round-robin across workers, so ordering
#    must be stable across all workers (use sorted(), not glob directly).
def get_episodes(split):
    if split == 'train':
        return sorted(glob.glob('/data/my_dataset/train/episode_*.hdf5'))
    elif split == 'val':
        return sorted(glob.glob('/data/my_dataset/val/episode_*.hdf5'))
    return []

# 4. Episode parser — load one file, return (key, example_dict).
#    Same as what you would write inside _generate_examples() of a serial builder.
def parse_episode(episode_path):
    data = load_my_format(episode_path)
    steps = [{'image': s['img'], 'action': s['act'].astype(np.float32)} for s in data]
    return episode_path, {
        'steps': steps,
        'episode_metadata': {'file_path': tf.constant(episode_path).numpy()},
    }
```

**Rules for `get_episodes`:**
- Must return a sorted, reproducible list — every worker calls this independently and slices from it, so the order must agree.
- Return an empty list (not raise) for splits that don't exist.

**Rules for `parse_episode`:**
- Return `(key, dict)` exactly as you would `yield` from `_generate_examples`.
- The key is used as the example ID — use the file path or any unique string.
- Returning `None` is not supported; skip logic should go inside `get_episodes`.

---

### Step 2 — Run the pipeline

```bash
cd slurm_rlds/

python scripts/pipeline.py \
    --config        /path/to/my_dataset_config.py \
    --data_root     gs://my-bucket/my_dataset_workers \
    --output        gs://my-bucket/my_dataset_merged \
    --num_workers   16 \
    --dataset_name  my_dataset \
    --dataset_version 1.0.0 \
    --framework_dir /path/to/slurm_rlds \
    --mode          slurm \
    --slurm_partition preempt
```

The script runs four phases in sequence:

| Phase | What it does |
|---|---|
| **1. Build** | Dispatches workers (SLURM array or local subprocesses). Checks completion markers. If any workers are missing after the first run, retries exactly once — completed workers skip themselves via their own marker check. |
| **2. Merge** | Collects shards from all worker output directories, renumbers them sequentially, writes merged `dataset_info.json`. Always writes `shard_map.json` mapping `final_shard_idx → (worker_id, worker_shard_idx, n_worker_shards)` per split. |
| **3. Scan** | Fully iterates every record in every merged shard via `tf.data.TFRecordDataset`. Reports `BAD` (DataLossError or count mismatch) and `ERROR` (DNS failure, network reset, etc.) separately. |
| **4. Fix** | Re-copies each bad shard from the original worker output using `shard_map.json`, up to 5 attempts per shard. Re-scans fixed shards to confirm recovery. Exits non-zero if any remain bad. |

#### Key flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | auto-detected | `slurm` if `sbatch` is on PATH, else `local` |
| `--copy_workers` | `32` | Parallel threads for shard copy and scan |
| `--overwrite` | off | Overwrite existing merged shards |
| `--skip_build` | off | Skip phase 1 (assume workers already done) |
| `--skip_merge` | off | Skip phases 1–2, read existing `shard_map.json` from output |
| `--skip_scan` | off | Skip phases 3–4 (scan + fix); merge only, no integrity check/repair |
| `--slurm_partition` | `preempt` | SLURM partition |
| `--slurm_time` | `48:00:00` | Wall time per worker |
| `--slurm_cpus` | `8` | CPUs per worker task |
| `--slurm_mem` | `64G` | Memory per worker task |
| `--slurm_gres` | `gpu:1` | SLURM `--gres` for the build array (empty string omits it) |
| `--env_setup` | conda `rlds` + certs | Shell snippet to activate the build env in the generated script; override per dataset, e.g. `--env_setup 'source /path/to/venv/bin/activate'` |
| `--log_dir` | `logs/` | Directory for SLURM worker stdout/stderr |

---

### Step 3 — Manual tools (optional)

If you want to run phases individually:

**Standalone merge** (no scan/fix):
```bash
python scripts/merge_shards.py \
    --root            gs://my-bucket/my_dataset_workers \
    --output          gs://my-bucket/my_dataset_merged \
    --num_workers     16 \
    --dataset_name    my_dataset \
    --dataset_version 1.0.0

# Check completion before merging:
python scripts/merge_shards.py ... --dry_run
```

**Run a single worker locally** (useful for debugging your config):
```bash
cd slurm_rlds/
python -m framework.runner \
    --config      /path/to/my_dataset_config.py \
    --data_dir    /tmp/debug/0 \
    --worker_id   0 \
    --num_workers 2
```

---

## GCS output layout

```
gs://my-bucket/my_dataset_workers/
  0/my_dataset/1.0.0/{dataset_info.json, features.json, *.tfrecord}
  1/my_dataset/1.0.0/{...}
  ...

gs://my-bucket/my_dataset_merged/
  my_dataset/1.0.0/
    dataset_info.json
    features.json
    shard_map.json          ← final_shard_idx → (worker_id, shard_idx, n_shards)
    my_dataset-train.tfrecord-00000-of-NNNNN
    my_dataset-train.tfrecord-00001-of-NNNNN
    ...
```

The merged output is a standard TFDS dataset directory and can be loaded directly:
```python
import tensorflow_datasets as tfds
ds = tfds.load('my_dataset', data_dir = 'gs://my-bucket/my_dataset_merged')
```

---

## How episode distribution works

`get_episodes(split)` returns the full list for a given split. The framework slices it:

```
worker 0 → episodes[0::N]   (indices 0, N, 2N, ...)
worker 1 → episodes[1::N]   (indices 1, N+1, 2N+1, ...)
...
worker N-1 → episodes[N-1::N]
```

Each worker writes its slice as an independent TFDS dataset to `data_root/<worker_id>/`. The merge step concatenates all workers' shards into one flat dataset. Workers that produce no episodes for a split (because their slice is empty) simply produce no shards for that split — this is normal and not an error.

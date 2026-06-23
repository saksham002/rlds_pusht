# agibot — AgiBot World 2026 IL → RLDS

Converts the 13 ImitationLearning tasks of AgiBot World 2026 (LeRobot v2.1
shards, ~273 h / ~11.7k episodes) into RLDS tfrecords in the RoboCOIN loader
format, via the generic `slurm_rlds/` framework.

## Inputs

- Data root: `/data/group_data/rl/datasets/agibot_world/` (override with
  `AGIBOT_DATA_ROOT`). Expected to mirror `gs://max-europe-v2/lerobot/`
  layout: `ImitationLearning/{CommercialSpaces,Home}/task_XXXX/<start>_<end>/`
  with each range shard a self-contained LeRobot v2.1 dataset
  (`data/chunk-XXX/episode_NNNNNN.parquet`, `meta/*.jsonl|json`,
  `videos/chunk-XXX/observation.images.<cam>/episode_NNNNNN.mp4`).
- Episode indices are local 0-based per shard; videos are AV1 (decoded with
  pyav/libdav1d).

## Schema decisions

- **State/action as named components** — each `field_descriptions` entry from
  `info.json` becomes its own float32 step feature, name preserved verbatim
  (e.g. `state/end/wrench`, `action/robot/velocity`). Components with varying
  dims across shards (`state/end/wrench` 12|24, `state/end/velocity` 12|24,
  `state/robot/position` 0|3, `state/robot/orientation` 0|4,
  `action/robot/velocity` 2|6) are zero-padded to the max; actual dims per
  episode are in `episode_metadata.{state,action}_feature_dims` aligned with
  `*_feature_names`.
- **Subtasks** = sentence-level Task Frames only (`key_frame[ep]['dual']`,
  `frame_type_name` matched case/punctuation-insensitively across the four
  spellings). Filled into RoboCOIN's 5-slot layout ordered by start; observed
  max overlap is 2. Slot text for inactive slots is `'null'`. Per-slot
  `subtask_success` carries `is_result_succeed`. Skill-level
  `instruction_segments` are not written.
- **`is_error_frame`** — per-step bool from Error Frame spans.
- **`eef_sim_pose_state/action` (12,)** — `[pos(3), euler_xyz(3)] × {left,
  right}` from `*/end/arm_position|arm_orientation` (state) and
  `action/end/position|orientation`. Quaternions are **xyzw**, left arm then
  right (official AgiBot World proprio convention).
- **`has_base_motion`** — episode metadata bool, true iff the episode's
  `action/robot/velocity` min/max in `episodes_stats.jsonl` is nonzero
  anywhere.
- **Images** — `cam_0/1/2` = `top_head`, `hand_left`, `hand_right`, resized
  to 480×480 (RoboCOIN-style `tf.image.resize` on [0,1] floats → rint) and
  JPEG-encoded at quality 95. Episodes whose summed image bytes would exceed
  protobuf's 2 GB per-example limit (~90 episodes >7k frames) fall back to
  quality 90, then 85, never lower — a build error is raised if q85 still
  doesn't fit. The applied quality is in `episode_metadata.jpeg_quality`;
  `camera_shapes` records the SOURCE video resolutions (native 400×640 /
  528×640; 3 shards of task_3477 have 1056×1280 hand cams). Depth and fisheye
  streams are skipped.
- **Split** — deterministic per-task 5% val (seed 86); `repo_index` = index of
  the task in the sorted 13-task list; `repo_id` = shard path relative to the
  data root.
- No `state_diff`/`action_diff` and no `scene_annotation` (dropped relative to
  the RoboCOIN builder).

## Run

Debug a single worker locally (data must be present):

```bash
cd ../slurm_rlds/
python -m framework.runner \
    --config ../agibot/agibot_config.py \
    --data_dir /tmp/agibot_debug/0 \
    --worker_id 0 \
    --num_workers 24
```

The production build runs one tfds build per (worker, task) unit
`<task_idx * 24 + worker_id>` on node-local scratch, uploads each unit to
`gs://saksham-euw4/datasets/agibot_world/<unit>/` (copying `dataset_info.json`
LAST, so a GCS marker implies a complete unit), then deletes it locally — a
preemption only loses the in-progress unit. The full dataset is ~8 TB, too
large for the group share, hence GCS. The bespoke SLURM-array build/upload
orchestration is preemption-resilient and per-unit; it is not part of this
commit.

Then merge on GCS over the 13 × 24 = 312 unit dirs (no scan/fix phases) with
the shared framework merge:

```bash
cd ../slurm_rlds/
python scripts/merge_shards.py \
    --root            gs://saksham-euw4/datasets/agibot_world \
    --output          gs://saksham-euw4/datasets/agibot_world_merged \
    --num_workers     312 \
    --dataset_name    agibot \
    --dataset_version 1.0.0
```

Note: agibot uses a bespoke per-unit build/upload rather than `slurm_rlds`'s
`pipeline.py` because the ~8 TB output is streamed to GCS one unit at a time
(build on scratch → upload → delete) instead of written to a single per-worker
`data_root`. (`pipeline.py` now parameterizes the build env and `--gres` via
`--env_setup`, so those are no longer blockers — only the GCS-unit streaming
is.)

# lerobot — real-world dual-xArm packing → RLDS

Converts the real-world dual-xArm packing LeRobot dataset
(`/data/group_data/rl/saksham3/realworld_xarm_packing_lerobot/`, 7
`xarm_baseline_round*` folders, LeRobot v2.1, 60 fps) into RLDS tfrecords via
the `slurm_rlds/` framework, with each **combined episode (chunk)** as one
RLDS episode.

## Chunking

Each source episode is a short (~8–10 s) subtask mp4. Consecutive episodes
within a round are grouped into one combined episode when the scene is
continuous, judged by the **base-camera** image MAD at the boundary
(`MAD(last_frame[ep_i], first_frame[ep_{i+1}]) <= 12`). The signal is cleanly
bimodal (continuous < 5, reset > 20, nothing in [8,20)), so the tolerance is
robust anywhere in [8,20). Chunks never cross rounds. Frames are concatenated
with **no** dropping (boundary frames are not identical).

- All 7 rounds → **312 chunks**. (Dropping the superseded original
  `xarm_baseline_round1` → 282; toggle `ROUNDS` in
  `scripts/build_chunks_json.py`.)

`scripts/build_chunks_json.py` writes `chunks.json` into the data root with,
per chunk: `repo_id`/`repo_index`, ordered subtask list (source episode index,
task sentence, frame range within the combined episode, `success=True`), and a
templated combined instruction. `boundary_mads_base.json` caches the precomputed
boundary image diffs.

## Schema (per-frame)

- `observation/image/cam_{0,1,2}` = base, left_wrist, right_wrist; resized to
  480×480, JPEG q95.
- `observation/state`, `action` — 14-dim `[x,y,z,roll,pitch,yaw,gripper]` ×
  {left,right}, gripper at positions 7 and 14. Source 20-dim pose is in the
  robot **base frame**; the 6D rotation is the **first two rows** of
  R_eef_in_base (row2 = row0×row1), converted to Euler `xyz` extrinsic
  (radians). Gripper is the raw encoder value (~80–842). `action` comes from
  the source `action` column directly, reformatted identically.
- Single active-subtask fields (no slot arrays): `subtask`, `subtask_index`,
  `steps_to_subtask_end`, `subtask_len`, `subtask_is_first`, `subtask_is_last`.
- `task` = combined episode instruction; `is_first`/`is_terminal`,
  `frame_index`, `index`, `episode_index` (= global chunk index), `repo_index`.
- `episode_metadata`: `repo_id` (folder name), `repo_index` (0-indexed per
  folder), `robot_type`, `fps`, camera names/shapes, state/action feature
  names, `subtasks`, `task_description`, `num_subtasks`.
- 60 fps (no subsampling); 5% val split (seed 86), at the chunk level.

## Run

```bash
# 1. (re)generate chunks.json
python scripts/build_chunks_json.py

# 2. build (SLURM array, 8 workers, preempt)
cd lerobot/ && mkdir -p logs && sbatch scripts/build.sh
# output: /data/group_data/rl/saksham3/realworld_xarm_packing_rlds/workers/<id>/

# 3. merge
cd ../slurm_rlds/
python scripts/merge_shards.py \
    --root /data/group_data/rl/saksham3/realworld_xarm_packing_rlds/workers \
    --output /data/group_data/rl/saksham3/realworld_xarm_packing_rlds/merged \
    --num_workers 8 --dataset_name realworld_xarm_packing --dataset_version 1.0.0
```

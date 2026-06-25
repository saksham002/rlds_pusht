# AgiBot World 2026 — dataset notes

Consolidated documentation of the AgiBot World 2026 release as mirrored at
`gs://max-europe-v2/lerobot` (local IL copy:
`/data/group_data/rl/datasets/agibot_world/`). This merges the original
metadata survey (conducted 2026-06-10/11 from the shard `meta/` folders, no
parquet/video downloads) with corrections discovered while building the RLDS
dataset and annotation-quality issues observed in the data and the demo
videos. The RLDS builder itself is documented in `README.md` in this folder.

## Provenance

- [AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026),
  Phase 1 (imitation learning) of a planned five-phase release. License
  **CC BY-NC-SA 4.0**. No standalone 2026 tech report yet; the predecessor
  paper is [AgiBot World Colosseo (arXiv 2503.06669)](https://arxiv.org/abs/2503.06669)
  and the 2026 spec lives in the HF dataset card.
- Platform: **AGIBOT G2** humanoid (`robot_type: "g2a"`) — dual 7-DoF arms,
  5-DoF waist, 3-DoF head, velocity-controlled mobile chassis, Zhixing 90D
  parallel grippers and OmniHand dexterous hand (fleet-wide). This slice is
  **gripper-only** (no OmniHand).
- Collection: free-form whole-body teleoperation (first-person, beyond-
  visual-range) in real commercial/home spaces, 100% real-world, 30 fps.
- Quaternions are **xyzw, left arm then right** (confirmed in the official
  AgiBotWorld-Alpha proprio docs: "flange quaternion with xyzw").

## Layout

```
gs://max-europe-v2/lerobot/
├── ImitationLearning/        # 13 tasks, ~273 h, 11,701 episodes
│   ├── CommercialSpaces/task_{3400,3401,3402,3404,3405,3477,3641,3705,3777,4053,4542,4799}/
│   └── Home/task_4713/
└── RichInteraction/          # 10 tasks, ~160 h, 7,662 episodes
    ├── CommercialSpaces/task_{4158,4182,4439,4603,4952,4962}/
    └── Home/task_{4224,4231,4560,5015}/
```

Each task splits into `<start>_<end>` episode-range shards (235 total), each
a self-contained **LeRobot v2.1** dataset: `data/chunk-XXX/episode_NNNNNN.parquet`
+ `meta/{info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl}` +
`videos/chunk-XXX/observation.images.<cam>/episode_NNNNNN.mp4` (AV1).
Episode indices are **local, 0-based per shard**. Some range folders are
empty (release quirk). Total: 19,363 episodes / 46.7 M frames / 432.9 h
(IL 273.0 h, RI 159.9 h). Data is concentrated — 6 tasks (3641, 3777, 4542,
4962, 4799, 4158) hold ~61% of the hours.

Hierarchy: episode ~80.5 s (mean; median 75 s, range 13.9–289.6 s) →
sentence subtask ~21.2 s (median 16.6 s) → skill segment ~7.6 s (median
4.6 s).

## State / action schema — dims VARY across shards

The survey's initial claim of a uniform 169-D state / 44-D action is
**wrong**. Three layouts exist in IL (verified across all 146 IL shards):

| Layout | state | action | where |
| --- | --- | --- | --- |
| A | 162 | 40 | parts of the 10 stationary tasks (`robot/position`, `robot/orientation` 0-dim) |
| B | 169 | 40 | parts of the 10 stationary tasks |
| C | 193 | 44 | 4542, 4799, 4713 (`end/wrench`/`end/velocity` 24-D instead of 12, chassis velocity 6-D instead of 2) |

Per-shard layouts are in `info.json → features.{observation.state,action}
.field_descriptions` (contiguous `indices` into the flat vector; verified
complete and contiguous in every shard). Component names are identical
across shards; only dims differ. State includes per-frame camera
extrinsics (6 cameras × rotation_matrix 9 + translation_vector 3);
intrinsics are only in `info.json camera_parameters`.

Representative **action** component layout (layout C, 44-D):

| Field | Dims | Indices |
| --- | --- | --- |
| `action/left_effector/position` | 1 | 0 |
| `action/right_effector/position` | 1 | 1 |
| `action/end/position` (both arms EEF xyz) | 6 | 2–7 |
| `action/end/orientation` (both arms quat) | 8 | 8–15 |
| `action/joint/position` (2×7 arms) | 14 | 16–29 |
| `action/head/position` | 3 | 30–32 |
| `action/waist/position` | 5 | 33–37 |
| `action/robot/velocity` (chassis) | 6 | 38–43 |

A 14-dim bimanual EEF+gripper action space (analogous to RoboCOIN's) is
directly extractable from `action/end/*` + the two effector dims — no IK
needed.

## Cameras

7 streams: `top_head` (400×640), `hand_left`/`hand_right` (528×640),
`head_depth`, and 3 fisheyes (`head_left_fisheye`, `head_right_fisheye`,
`head_back_fisheye`). **Exception**: 3 shards of **task_3477**
(`248923_250431`, `250434_252697`, `252787_255587`; 264 episodes) have
1056×1280 hand cameras. (The survey attributed the high-res shards to
3401/3641 — also wrong.) `episodes_stats.jsonl` video-channel stats are
zeroed (known quirk).

## Annotation layers (all in `meta/info.json`)

1. **Task Frames** (sentence subtasks) — `key_frame[ep]["dual"]` entries
   with `frame_type_name` in {"Task Frame", "Sub-task frame",
   "Sub-task Frame", "SubTask Frame"} (match case/punctuation-insensitively
   or you silently lose 23,530 segments). `[start, end)` frame intervals,
   free-form sentence in `frame_detail.comment`, success flag
   `frame_detail.is_result_succeed`. 45,180 segments; all 13 IL tasks; zero
   in RI. Segments can overlap (max depth 2; only 2 episodes, both
   same-text duplicates in task_3641). The official `split_episode.py` (HF
   `split_episodes_tool.zip`) cuts episodes at these boundaries.
2. **2D bounding boxes** — `frame_type_name == "2D Bounding Box"`: frame
   interval, normalized `{x, y, w, h}` box (top-left origin), `camera`
   (e.g. `head_color`), arm in `track`; 31,108 entries.
3. **Error Frames** — `frame_type_name` "Error Frame"/"Error frame", with
   `frame_detail.error_cause` and `restorable`. **2,200 segments spread
   across stationary tasks too**, not only 4542/4799/4713 as the survey
   claimed.
4. **Instruction segments** (skill level) — `instruction_segments[ep]`:
   `skill`, `instruction`, `start/success/end_frame_index`. Real skill
   labels only in the 10 stationary IL tasks; placeholders
   (`Other(Other)`/empty) in 4542/4799/4713 (whose *instruction* text is
   still meaningful) and in all of RI (single whole-episode "interacts
   randomly" segment).
5. `take_over` — empty everywhere (0 take-over segments across all 19,363
   episodes). No `Success Frame`/`Intervention Frame` entries anywhere.

### Episode-level task strings

Each episode carries a high-level task *name* (`episodes.jsonl → tasks[0]`),
e.g. "Grocery Area - Pushang Fresh Food." — a scene/section label, not an
imperative instruction. The same task carries multiple spelling/translation
variants across episodes ("Grocery section" vs "Grocery Area - Pushang
Fresh Food."). Verified: every episode task string matches its shard's
`tasks.jsonl` verbatim (all 19,363 episodes). Scene names span supermarket
"Pushang/Pusheng Fresh" sections (refrigerated, household goods, grocery,
general merchandise, storage, food, frozen), "Cinema - Xintiandi" (ticket
checking, flyer handing, concessions/scooping), and "Homestay 1/2" home
scenes. The actionable language lives in the Task Frames and instruction
segments.

### Task Frame semantics

Of 11,644 annotated episodes, 9,153 have >1 subtask frame (typically 2–4, up
to 12+); the instruction text changes between consecutive segments in 73% of
those. Two patterns:

- **Object-varying repetition** (restocking: 3400, 3641, …): consecutive
  segments target different object instances ("blue-lidded yogurt" → "pink
  bottled yogurt"), each with its own boundary and success flag.
- **Paraphrase repetition** (service: 4053 flyer-handing, 3777
  ticket-checking): same goal re-worded ("Verify" → "Validate") —
  deliberate instruction augmentation.

Episodes are sequences of self-contained subtask repetitions over varying
objects, not multi-stage plans. "Instruction changed" ≠ "new distinct goal"
without text-level dedup.

### Skill vocabulary

34 distinct labels; ~25 meaningful after collapsing duplicates
(`Move(Move)`, `Grasp(Grasp)`) and dropping placeholders (`Other(Other)`
61.4k, empty 58.5k, `Other` 14.6k — ~55% of all segments). Top real skills:
Pick 25.9k (mean 11.6 s), Bend waist forward 22.3k (3.0 s), Place 15.7k
(6.0 s), HandOut 11.6k (4.4 s), Scoop 6.3k (16.4 s), Arch waist backward
6.2k, Raise/Lower height ~5.2k each, then HandOver, Scan, Straighten, Grasp,
Wipe, Move, Transport, Drop, Push, Release, Hang, Close, Open, Hold, Carry,
Point, TurnWheel. Whole-body posture skills (waist/height) have no RoboCOIN
analogue.

### Annotation coverage

- All 13 IL tasks have Task Frames, but coverage is per-episode: 11,644 of
  11,701 IL episodes have ≥1 segment; **57 have none** (e.g. 49 of 264
  episodes in task_3477).
- Base motion is confined to a subset: the 10 stationary IL tasks (3400,
  3401, 3402, 3404, 3405, 3477, 3641, 3705, 3777, 4053; ~217 h) have
  max |chassis v| = 0; only 4542, 4799, 4713 (~106 h) and 8 of 10 RI tasks
  (4182, 4231 are stationary) carry chassis motion.
- RichInteraction has **no key_frame annotations of any kind**, but the play
  data is good quality with high object diversity (reviewed via the demo
  videos), despite carrying no segmentation or success labels. Every RI
  episode has a single whole-episode instruction segment paraphrasing "the
  robot gripper interacts randomly with any object in the field of view".

## Annotation quality issues

1. **Wrong arm in instruction text** — some Task Frame sentences say
   "left arm" when the right arm performs the motion (and vice versa);
   observed when reviewing the rendered demo videos. Arm references in
   subtask text should not be trusted for arm-level supervision; the
   bounding-box `track` field is the more reliable arm signal.
2. **Subtask intervals with no text** — 18 of 45,180 Task Frame segments
   have an empty `frame_detail.comment`: the interval and success flag are
   annotated but the sentence is missing. 16 of the 18 sit in episodes
   whose other segments carry (usually identical, repeated) text; 2
   episodes (task_3641 `343691_352441` ep 47, task_3777 `338316_342675`
   ep 102) have *only* the blank segment and thus no usable subtask text.
   In the RLDS output these appear as an active slot (`subtask_mask=True`)
   with an empty string — filter with `subtask_mask & (text != '')`.
3. **Reversed object direction** — some instructions state the transfer
   direction backwards, e.g. "move the object from the shopping cart to
   the freezer" when the demonstrated motion is freezer → shopping cart;
   observed in the demo videos. Direction phrases in subtask text are not
   reliable without visual verification.

Additional source-data caveats (not strictly annotation errors):
inconsistent `frame_type_name` spellings (4 variants); duplicate
overlapping same-text segments (2 episodes); unannotated episodes (57 in
IL); placeholder skills (~55% of all instruction segments); task-name
translation variants requiring text-level dedup; zeroed video stats in
`episodes_stats.jsonl`; empty `<start>_<end>` range folders.

## Headline statistics

| Metric | Value |
| --- | --- |
| Tasks / shards / episodes | 23 / 235 / 19,363 |
| Frames (data points) | 46,756,030 |
| Hours @30 Hz | **432.9** (IL 273.0, RI 159.9) |
| Episode length | mean 80.5 s (2,415 frames), median 75 s, range 13.9–289.6 s |
| Sentence subtasks (Task Frames, all spellings) | 45,180; mean **21.2 s**, median 16.6 s, max 278 s |
| Skill segments | 108,908; mean **7.6 s**, median 4.6 s, p90 15.9 s |
| End-effector | **1-DoF parallel gripper in all 23 tasks** (no OmniHand) |
| Tasks with zero chassis motion | **12 / 23** |
| Error Frame segments | ~2,200 (across stationary tasks too, not only 4542/4799/4713) |
| Take-over segments | **0** across all 19,363 episodes |

## Per-task summary

| task | collection | scene | shards | eps | hours | ep len (s) | subtasks | sub len (s) | max \|v\| |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3400 | IL | CommercialSpaces | 6 | 548 | 10.8 | 71 | 1006 | 37.5 | 0 |
| 3401 | IL | CommercialSpaces | 17 | 952 | 31.8 | 120 | 3415 | 33.0 | 0 |
| 3402 | IL | CommercialSpaces | 5 | 274 | 8.6 | 113 | 578 | 53.0 | 0 |
| 3404 | IL | CommercialSpaces | 5 | 312 | 9.5 | 109 | 800 | 41.5 | 0 |
| 3405 | IL | CommercialSpaces | 1 | 13 | 0.3 | 79 | 26 | 39.0 | 0 |
| 3477 | IL | CommercialSpaces | 3 | 264 | 5.7 | 77 | 286 | 52.8 | 0 |
| 3641 | IL | CommercialSpaces | 24 | 1682 | 46.9 | 100 | 5145 | 31.9 | 0 |
| 3705 | IL | CommercialSpaces | 1 | 44 | 1.0 | 81 | 64 | 54.7 | 0 |
| 3777 | IL | CommercialSpaces | 24 | 2727 | 46.7 | 62 | 8034 | 20.2 | 0 |
| 4053 | IL | CommercialSpaces | 3 | 343 | 5.5 | 58 | 2296 | 8.6 | 0 |
| 4542 | IL | CommercialSpaces | 27 | 2594 | 50.9 | 71 | 16159 | 11.3 | 1.00 |
| 4799 | IL | CommercialSpaces | 22 | 1493 | 41.5 | 100 | 6103 | 23.8 | 0.87 |
| 4713 | IL | Home | 8 | 455 | 13.9 | 110 | 1268 | 38.6 | 0.38 |
| 4158 | RI | CommercialSpaces | 17 | 1673 | 32.9 | 71 | 0 | — | 0.55 |
| 4182 | RI | CommercialSpaces | 13 | 1241 | 24.7 | 72 | 0 | — | 0 |
| 4439 | RI | CommercialSpaces | 3 | 201 | 3.9 | 71 | 0 | — | 1.00 |
| 4603 | RI | CommercialSpaces | 3 | 216 | 4.5 | 75 | 0 | — | 1.01 |
| 4952 | RI | CommercialSpaces | 4 | 283 | 6.2 | 78 | 0 | — | 0.73 |
| 4962 | RI | CommercialSpaces | 26 | 1986 | 45.2 | 82 | 0 | — | 0.83 |
| 4224 | RI | Home | 1 | 90 | 1.9 | 76 | 0 | — | 0.52 |
| 4231 | RI | Home | 4 | 356 | 7.8 | 79 | 0 | — | 0 |
| 4560 | RI | Home | 2 | 135 | 3.0 | 80 | 0 | — | 0.75 |
| 5015 | RI | Home | 16 | 1481 | 29.9 | 73 | 0 | — | 1.12 |

(`max |v|` = max absolute `action/robot/velocity` over all episodes, from
`episodes_stats.jsonl`. `subtasks` = Task Frame count; RI tasks have none.)

## RLDS conversion (this folder)

See `README.md` for the build pipeline and `agibot_config.py` for the
schema. Key decisions: componentized state/action with original LeRobot
field names (zero-padded to per-field max dims, actual dims in episode
metadata); Task Frames only in the 5-slot subtask layout (`'null'` +
`mask=False` when inactive); `subtask_success`, `is_error_frame`,
`eef_sim_pose_state/action` extras; `has_base_motion` episode flag from
`action/robot/velocity` stats; images 480×480 JPEG q95 with adaptive
fallback to q90/q85 for episodes that would exceed protobuf's 2 GB
per-example limit (~90 episodes >7k frames); per-task 5% val split
(seed 86). Only the 13 IL tasks are converted; output at
`gs://saksham-euw4/datasets/agibot_world/` (merged:
`.../agibot_world_merged`).

## Notes for a RoboCOIN pretraining mix

- The directly usable slice for subtask-prompted value learning is **all 13
  IL tasks** (~273 h, ~32 M frames), gripper-only, fps 30 (matches the
  RoboCOIN sim native rate). Subtask frames give `steps_to_subtask_end`
  analogues, with per-subtask `is_result_succeed` for the termination/
  success convention. The cleanest subset is the 10 stationary tasks
  (~217 h); adding 4542/4799/4713 (~106 h) brings base motion into the
  trajectories (commanded via `action/robot/velocity`, up to ~1 m/s) that a
  14-dim EEF action space does not capture.
- Long-horizon vs RoboCOIN holds: ~80 s episodes, ~4 sentence subtasks per
  episode at ~21 s each, with much larger scene/object diversity.
- RichInteraction cannot support MC-return targets as-is — no segmentation,
  no success labels. Representation-level value only.
- Integration work: a LeRobot→RLDS ingest path (this folder), G2 embodiment
  norm stats (quantile), text dedup if Task-Frame paraphrases matter, and a
  decision on the body dims (head 3 + waist 5) that RoboCOIN's 14-dim action
  space lacks.
- No take-over segments anywhere; ~2,200 Error Frame spans (across stationary
  tasks too) could drive failure filtering, analogous to RoboCOIN's
  intervention masks.
- Episode-level data is heavily skewed to 6 tasks; weighting may be needed.

## Demo videos

`/data/user_data/saksham3/agibot_world/demos/<Collection>/<Scene>/task_<id>/`
holds 5 annotated 30 Hz head-camera videos per task (all 23 IL + RI tasks),
rendered by `scripts/save_demo_videos.py`: banner with task name, wrapped
subtask sentence + OK/FAIL flag, skill + instruction, error cause during
Error Frame spans, and head-camera bounding boxes drawn with arm labels.

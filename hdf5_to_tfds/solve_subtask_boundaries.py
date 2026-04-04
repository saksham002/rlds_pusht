"""Heuristic subtask boundary solver for dexterous shirt-hanging episodes.

Reads HDF5 episodes at 60 FPS, subsamples to 30 FPS, and detects subtask
transitions using gripper state, TCP Z position, and velocity signals.

Usage:
    python solve_subtask_boundaries.py \
        --episode_list episodes.txt \
        --output annotations/heuristic_annotations.json

    The episode list is a text file with one full path per line, e.g.:
        /data/group_data/rl/dexterous_robot_data/real_hang_full_success_r5_hdf5/episode_3.hdf5
        /data/group_data/rl/dexterous_robot_data/real_hang_full_success_r5_hdf5/episode_49.hdf5

Output JSON structure:
    {
        "subtask_definitions": [...],
        "datasets": {
            "real_hang_full_success_r5_hdf5": {
                "3": {"total_steps": 1767, "boundaries": [133, 492, 917, 1510]},
                ...
            }
        }
    }

    boundaries[i] is the first step of the NEXT subtask (i.e., subtask i ends at
    boundaries[i] - 1). Full successes have 5 boundaries for 6 subtasks;
    partial successes have fewer.
"""
import argparse
import json
import os

import h5py
import numpy as np

GRIP_THRESH = 400
MIN_BOUNDARY_GAP = 75

SUBTASK_DEFINITIONS = [
    {"subtask_index": 0, "subtask": "Grasp the hanger"},
    {"subtask_index": 1, "subtask": "Lift the hanger off the rod"},
    {"subtask_index": 2, "subtask": "Pass hanger from right to left arm"},
    {"subtask_index": 3, "subtask": "Hook one side of the shirt onto the hanger"},
    {"subtask_index": 4, "subtask": "Hook the other side of the shirt onto the hanger"},
    {"subtask_index": 5, "subtask": "Place the hanger on the rod"},
]


def _find_grip_transitions(grip):
    is_open = grip > GRIP_THRESH
    events = []
    for i in range(1, len(grip)):
        if is_open[i] != is_open[i - 1]:
            events.append((i, "OPEN" if is_open[i] else "CLOSE"))
    return events


def _check_gaps(boundaries, n, full):
    """Check gap constraints. Returns True if all gaps > MIN_BOUNDARY_GAP.

    full=True  → all 5 boundaries found, check gaps including last→end.
    full=False → partial boundaries, check gaps excluding last→end.
    """
    all_pts = [0] + boundaries + ([n - 1] if full else [])
    return all(all_pts[i + 1] - all_pts[i] > MIN_BOUNDARY_GAP for i in range(len(all_pts) - 1))


def detect_boundaries(episode_path, counters = None):
    """Detect subtask boundaries from a single HDF5 episode.

    Returns (boundaries, total_steps) where boundaries is a list of up to 5
    ints representing the first step of subtasks 1-5 (at 30 FPS indexing).
    Returns (None, total_steps) if:
      - no boundaries are found at all, or
      - all 5 found but any gap (including last→end) <= MIN_BOUNDARY_GAP, or
      - partial found but any gap (excluding last→end) <= MIN_BOUNDARY_GAP.

    If counters is provided, it must be a dict with keys 'valid_ends_at',
    'failed_ends_at' (both list[5]), and 'no_transition'. The episode is
    counted as ending at the last transition it reached.
    """

    def _return(boundaries, n, num_found, full):
        if num_found == 0:
            if counters is not None:
                counters['no_transition'] += 1
            return None, n
        if _check_gaps(boundaries, n, full):
            if counters is not None:
                counters['valid_ends_at'][num_found - 1] += 1
            return boundaries, n
        if counters is not None:
            counters['failed_ends_at'][num_found - 1] += 1
        return None, n

    f = h5py.File(episode_path, "r")
    n_orig = f["obses/state/timestamp"].shape[0]
    idx = np.arange(0, n_orig, 2)

    lg = f["obses/state/left/gripper_pos"][idx, 0]
    rg = f["obses/state/right/gripper_pos"][idx, 0]
    rz = f["obses/state/right/tcp_pose"][idx, 2]
    rv = f["obses/state/right/tcp_vel"][idx]
    lv = f["obses/state/left/tcp_vel"][idx]
    f.close()

    r_speed = np.linalg.norm(rv[:, :3], axis = 1)
    l_speed = np.linalg.norm(lv[:, :3], axis = 1)

    n = len(lg)
    r_events = _find_grip_transitions(rg)
    l_events = _find_grip_transitions(lg)

    # T0→1: Successful right grasp
    # Right gripper closes with R_z > 0.30 and R_z subsequently descends
    # below 0.20 without the gripper reopening (filters failed attempts).
    t01 = None
    for step, direction in r_events:
        if direction == "CLOSE" and rz[step] > 0.30:
            for j in range(step, min(step + 500, n)):
                if rg[j] > GRIP_THRESH:
                    break
                if rz[j] < 0.20:
                    t01 = step
                    break
            if t01:
                break

    if t01 is None:
        return _return([], n, 0, False)

    # T1→2: Right arm lowers hanger to handoff height
    # Right TCP Z position drops below 0.22, indicating the hanger is low
    # enough for the pass-to-left-arm phase to begin.
    t12 = None
    for i in range(t01, n):
        if rz[i] < 0.22:
            t12 = i
            break

    if t12 is None:
        return _return([t01], n, 1, False)

    # T2→3: Left arm grabs hanger (handoff complete)
    # First LEFT stable close after T0→1 — left gripper closes and stays
    # closed for 40+ steps, confirming both arms now hold the hanger.
    t23 = None
    for step, direction in l_events:
        if step > t01 + 50 and direction == "CLOSE":
            end = min(step + 40, n)
            if np.all(lg[step:end] <= GRIP_THRESH):
                t23 = step
                break

    if t23 is None:
        return _return([t01, t12], n, 2, False)

    # T3→4: First side hooked
    # Right gripper closed (< 200) AND left gripper open (> 400) — the right
    # arm has finished hooking and the left arm releases its grip.
    t34 = None
    for i in range(t23 + 1, n):
        if rg[i] < 200 and lg[i] > 400:
            t34 = i
            break

    if t34 is None:
        return _return([t01, t12, t23], n, 3, False)

    # T4→5: Second side hooked, placing begins
    # Last LEFT CLOSE→OPEN after T3→4. The left arm releases for the final
    # time before both arms lift the shirt+hanger up to the rod.
    l_events_after = [(s, d) for s, d in l_events if s > t34]
    t45 = None
    for step, direction in l_events_after:
        if direction == "OPEN":
            t45 = step

    if t45 is None:
        return _return([t01, t12, t23, t34], n, 4, False)

    return _return([t01, t12, t23, t34, t45], n, 5, True)


def main():
    parser = argparse.ArgumentParser(description = "Solve subtask boundaries for dexterous hang episodes")
    parser.add_argument("--episode_list", required = True, help = "Text file with one HDF5 path per line")
    parser.add_argument("--output", required = True, help = "Output JSON path")
    args = parser.parse_args()

    with open(args.episode_list) as f:
        paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    counters = {
        'valid_ends_at': [0] * 5,
        'failed_ends_at': [0] * 5,
        'no_transition': 0,
    }

    # Group episodes by dataset directory (the folder name inside dexterous_robot_data/)
    datasets = {}
    for path in paths:
        parent = os.path.basename(os.path.dirname(path))
        ep_name = os.path.basename(path)
        ep_num = int(ep_name.replace("episode_", "").replace(".hdf5", ""))

        if parent not in datasets:
            datasets[parent] = {}

        boundaries, total_steps = detect_boundaries(path, counters = counters)

        entry = {"total_steps": total_steps}

        if boundaries is not None:
            entry["boundaries"] = boundaries
            status = "OK" if len(boundaries) == 5 else f"PARTIAL({len(boundaries)}/5)"
        else:
            entry["boundaries"] = None
            status = "FAILED"

        datasets[parent][str(ep_num)] = entry
        print(f"  [{status}] {parent}/episode_{ep_num} ({total_steps} steps) → {boundaries}")

    result = {
        "subtask_definitions": SUBTASK_DEFINITIONS,
        "datasets": datasets,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok = True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent = 4)

    total = sum(len(v) for v in datasets.values())
    print(f"\nSaved annotations for {total} episodes to {args.output}")

    valid_ends_at = counters['valid_ends_at']
    failed_ends_at = counters['failed_ends_at']
    no_transition = counters['no_transition']

    transition_names = [f"T{i}→{i+1}" for i in range(5)]
    print(f"\n{'='*50}")
    print(f"Episodes ending at each transition (valid / failed):")
    for i in range(5):
        print(f"  {transition_names[i]}:  valid={valid_ends_at[i]}  failed={failed_ends_at[i]}")
    total_valid = sum(valid_ends_at)
    total_failed = sum(failed_ends_at)
    print(f"  No transitions found:  failed={no_transition}")
    print(f"\n  Total valid: {total_valid}  |  Total failed: {total_failed + no_transition}")


if __name__ == "__main__":
    main()

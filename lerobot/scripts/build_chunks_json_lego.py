"""Build chunks.json for the YAM-leader lego round consumed by lerobot/lego_config.py.

One chunk per raw recording. Unlike the xarm rounds (grouped by an image-difference heuristic)
or the 3lego rounds (regrouped from language_annotations.json around dropped segments), the
segmentation here is fully determined: annotate_episodes.py on the collection PC writes one
annotations.json per recording listing its subtasks in order, and openpi-test's converter
(convert_yam_data_to_lerobot.py, sub-task mode) turned every listed segment into one LeRobot
episode. So a chunk is simply a recording, its subtasks are the consecutive LeRobot episodes
the converter produced for it, and no boundary heuristic or partial-label file is involved.

Ordering. Recording directories are named lego_YYYYMMDD_HHMMSS, so lexicographic order is
recording order. The converter walks the same sorted order, so LeRobot episode k is the k-th
segment in that order across the recordings it accepted. global_chunk_index is the recording's
position in that order and becomes the RLDS episode_index.

Verification. meta/episodes.jsonl of the converted dataset is authoritative for lengths: every
segment's LeRobot length must equal end_step - start_step and its task text must match. A
mismatch on the last segment means the converter applied a nonzero obs/action shift (it trims
the episode tail); this build assumes shift 0, which every pc0 recording resolves to.

Fields beyond the xarm schema: is_partial is False everywhere (no partial subtasks in this
round); task_id is the tray permutation index 0-5 (annotations.json rotation_index mod 6);
combined_annotation is the recording's task_description verbatim.

Usage:
    python scripts/build_chunks_json_lego.py \
        --raw-dir /path/to/data/lego \
        --lerobot-dir ~/.cache/huggingface/lerobot/local/lego_pc0 \
        [--repo-id lego_pc0] [--out /path/to/chunks.json]
"""

import argparse
import json
import os

FPS = 60
CONVERTER_ANNOTATIONS = "language_annotations.json"
ANNOTATOR_OUTPUT = "annotations.json"
NUM_PERMUTATIONS = 6


def converter_accepts(episode_dir: str) -> bool:
    """Mirror annotation_skip_reason() in convert_yam_data_to_lerobot.py: usable list with a usable text."""
    path = os.path.join(episode_dir, CONVERTER_ANNOTATIONS)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            segments = json.load(f)
    except json.JSONDecodeError:
        return False
    if not isinstance(segments, list) or not segments:
        return False
    return not all(segment.get("text", "") in ("", "[problematic]") for segment in segments)


def converter_segments(episode_dir: str, episode_length: int) -> list[dict]:
    """Mirror parse_subtask_segments(): start of each segment to the next start, last to the end."""
    with open(os.path.join(episode_dir, CONVERTER_ANNOTATIONS)) as f:
        segments = sorted(json.load(f), key=lambda segment: segment["start_step"])
    out = []
    for i, segment in enumerate(segments):
        start = 0 if i == 0 else segment["start_step"]
        if start >= episode_length:
            continue
        end = segments[i + 1]["start_step"] if i < len(segments) - 1 else episode_length
        end = min(end, episode_length)
        if segment["text"] == "[problematic]" or start >= end:
            continue
        out.append({"text": segment["text"], "start": start, "end": end})
    return out


def load_lerobot_episodes(lerobot_dir: str) -> dict[int, dict]:
    episodes = {}
    with open(os.path.join(lerobot_dir, "meta", "episodes.jsonl")) as f:
        for line in f:
            if line.strip():
                episode = json.loads(line)
                episodes[episode["episode_index"]] = episode
    return episodes


def build(raw_dir: str, lerobot_dir: str, repo_id: str) -> dict:
    lerobot_episodes = load_lerobot_episodes(lerobot_dir)
    # The converter treats every immediate subdirectory as a candidate (collect_episode_dirs).
    recordings = sorted(
        os.path.join(raw_dir, name) for name in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, name))
    )
    recordings = [directory for directory in recordings if converter_accepts(directory)]
    if not recordings:
        raise RuntimeError(f"no recording under {raw_dir} carries a usable {CONVERTER_ANNOTATIONS}")

    chunks = []
    next_lerobot_episode = 0
    for global_chunk_index, recording_dir in enumerate(recordings):
        recording = os.path.basename(recording_dir)
        with open(os.path.join(recording_dir, ANNOTATOR_OUTPUT)) as f:
            annotations = json.load(f)
        annotated = annotations["subtasks"]
        num_steps = annotated[-1]["end_step"]
        segments = converter_segments(recording_dir, num_steps)
        if len(segments) != len(annotated):
            raise RuntimeError(
                f"{recording}: converter would emit {len(segments)} segments but {ANNOTATOR_OUTPUT} lists "
                f"{len(annotated)} subtasks; regenerate {CONVERTER_ANNOTATIONS} with export_language_annotations.py"
            )

        subtasks, cursor = [], 0
        for segment, subtask in zip(segments, annotated, strict=True):
            lerobot_episode = lerobot_episodes.get(next_lerobot_episode)
            if lerobot_episode is None:
                raise RuntimeError(
                    f"{recording}: expected LeRobot episode {next_lerobot_episode} but the dataset has only "
                    f"{len(lerobot_episodes)} episodes"
                )
            expected_length = segment["end"] - segment["start"]
            if lerobot_episode["length"] != expected_length:
                raise RuntimeError(
                    f"{recording}: LeRobot episode {next_lerobot_episode} has {lerobot_episode['length']} frames, "
                    f"annotations imply {expected_length}. A shorter last segment means the converter applied a "
                    "nonzero obs/action shift; this build assumes shift 0."
                )
            lerobot_task = lerobot_episode["tasks"][0]
            if lerobot_task != segment["text"] or segment["text"] != subtask["description"].strip():
                raise RuntimeError(
                    f"{recording}: task text mismatch for LeRobot episode {next_lerobot_episode}: "
                    f"lerobot={lerobot_task!r} converter={segment['text']!r} annotator={subtask['description']!r}"
                )
            subtasks.append(
                {
                    "episode_index": next_lerobot_episode,
                    "task": segment["text"],
                    "phase": subtask["phase"],
                    "length": expected_length,
                    "start_frame": cursor,
                    "end_frame": cursor + expected_length - 1,
                    "is_partial": False,
                }
            )
            cursor += expected_length
            next_lerobot_episode += 1

        if cursor != num_steps:
            raise RuntimeError(f"{recording}: subtasks cover {cursor} frames, recording has {num_steps}")
        chunks.append(
            {
                "global_chunk_index": global_chunk_index,
                "repo_id": repo_id,
                "repo_index": 0,
                "recording": recording,
                "episode_indices": [subtask["episode_index"] for subtask in subtasks],
                "num_subtasks": len(subtasks),
                "total_frames": cursor,
                "task_id": annotations["rotation_index"] % NUM_PERMUTATIONS,
                "rotation_index": annotations["rotation_index"],
                "tray_assignment": annotations["tray_assignment"],
                "combined_annotation": annotations["task_description"],
                "subtasks": subtasks,
            }
        )

    if next_lerobot_episode != len(lerobot_episodes):
        raise RuntimeError(
            f"consumed {next_lerobot_episode} LeRobot episodes but the dataset has {len(lerobot_episodes)}; "
            "the raw directory and the converted dataset do not describe the same recordings"
        )

    return {
        "metadata": {
            "num_chunks": len(chunks),
            "fps": FPS,
            "rounds": [repo_id],
            "per_round_chunk_counts": {repo_id: len(chunks)},
            "chunking": "one chunk per raw recording, in lexicographic (= recording time) directory order; "
            "subtasks are the consecutive LeRobot episodes the converter wrote for it",
            "end_frame": "inclusive",
            "is_partial_source": "constant False: this round has no partial subtasks",
            "task_id": "tray permutation index, annotations.json rotation_index mod 6 "
            "(lexicographic permutations of (red, green, blue) = trays for red/green/blue blocks)",
            "label_source": f"{ANNOTATOR_OUTPUT} written by annotate_episodes.py on the collection PC",
        },
        "chunks": chunks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="directory of recordings (each with annotations.json)")
    parser.add_argument("--lerobot-dir", required=True, help="converted LeRobot dataset root (has meta/episodes.jsonl)")
    parser.add_argument("--repo-id", default=None, help="folder name recorded as repo_id (default: lerobot-dir basename)")
    parser.add_argument("--out", default=None, help="output path (default: <lerobot-dir>/chunks.json)")
    args = parser.parse_args()

    lerobot_dir = os.path.abspath(args.lerobot_dir)
    repo_id = args.repo_id or os.path.basename(lerobot_dir.rstrip("/"))
    out_path = args.out or os.path.join(lerobot_dir, "chunks.json")

    data = build(os.path.abspath(args.raw_dir), lerobot_dir, repo_id)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1)

    num_subtasks = sum(chunk["num_subtasks"] for chunk in data["chunks"])
    total_frames = sum(chunk["total_frames"] for chunk in data["chunks"])
    print(f"wrote {out_path}: {len(data['chunks'])} chunks, {num_subtasks} subtasks, {total_frames} frames")
    for chunk in data["chunks"][:3]:
        print(f"  [{chunk['recording']} task_id={chunk['task_id']} {chunk['num_subtasks']} subtasks]")
        print(f"     {chunk['combined_annotation']}")


if __name__ == "__main__":
    main()

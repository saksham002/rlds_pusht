"""Build chunks.json for the realworld_xarm_packing LeRobot dataset.

Each round (xarm_baseline_round*) is a LeRobot v2.1 dataset whose episodes are
short (~8-10 s) subtask videos. Consecutive episodes belong to the same
"combined episode" (chunk) when the scene is continuous, judged by the
base-camera image difference at the boundary: MAD(last_frame[ep_i],
first_frame[ep_{i+1}]) <= TOL means same chunk. The boundary MADs are
precomputed (boundary_mads_base.json); the signal is cleanly bimodal
(continuous < 5, reset > 20, nothing in [8,20)), so TOL anywhere in [8,20)
gives the same result. Chunks never cross rounds.

For each chunk we record every subtask mp4 (source episode index, task
sentence, frame range within the combined episode, success=True) and a
templated combined episode-level instruction.

Output: <DATA_ROOT>/chunks.json
"""
import json
import os
import re

DATA_ROOT = '/data/group_data/rl/saksham3/realworld_xarm_packing_lerobot'
MADS_PATH = os.path.join(os.path.dirname(__file__), '..', 'boundary_mads_base.json')
OUT_PATH = os.path.join(DATA_ROOT, 'chunks.json')
TOL = 12.0  # center of the clean [8,20) gap; identical result for any value in that range

# Include all 7 rounds (repo_index = position in this list). To drop the
# superseded original round1 (-> 282 chunks instead of 312), remove it here.
ROUNDS = [
    'xarm_baseline_round1',
    'xarm_baseline_round1good',
    'xarm_baseline_round2',
    'xarm_baseline_round3',
    'xarm_baseline_round4',
    'xarm_baseline_round5',
    'xarm_baseline_round6',
]

_PACK_RE = re.compile(r'Pack (.+?) into (?:the )?(\w+) box\.?$')
_REMOVE_RE = re.compile(r'Remove (.+?) from (?:the )?(\w+) box\.?$')


def _parse_subtask(task):
    """Return (verb, object, box) or (None, raw, None) if it doesn't match."""
    m = _PACK_RE.match(task)
    if m:
        return 'pack', m.group(1).strip(), m.group(2).lower()
    m = _REMOVE_RE.match(task)
    if m:
        return 'remove', m.group(1).strip(), m.group(2).lower()
    return None, task.rstrip('.'), None


def _dedupe(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _join(items):
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + f', and {items[-1]}'


def combined_annotation(subtask_tasks):
    """Templated episode-level instruction from the ordered subtask sentences.

    Objects are grouped BY TARGET/SOURCE BOX (small/medium/large) so the box
    identity is preserved — multiple boxes are visible in the cameras, so the
    box is what disambiguates where each object goes. Boxes and objects are
    listed in chronological order of first appearance; consecutive duplicates
    are removed. Example:
      'Pack the Soda can and Pink tumbler into the medium box and the Altoids
       container into the large box and remove the Sponge from the medium box.'
    """
    from collections import OrderedDict
    packed = OrderedDict()   # box -> [objects]
    removed = OrderedDict()
    other = []
    for t in subtask_tasks:
        verb, obj, box = _parse_subtask(t)
        if verb == 'pack':
            packed.setdefault(box, []).append(obj)
        elif verb == 'remove':
            removed.setdefault(box, []).append(obj)
        else:
            other.append(obj)

    clauses = []
    if packed:
        parts = [f'the {_join(_dedupe(objs))} into the {box} box' for box, objs in packed.items()]
        clauses.append('pack ' + _join(parts))
    if removed:
        parts = [f'the {_join(_dedupe(objs))} from the {box} box' for box, objs in removed.items()]
        clauses.append('remove ' + _join(parts))
    if other:
        clauses.append(_join(_dedupe(other)))
    if not clauses:
        return 'pack the objects into the boxes.'
    sentence = ' and '.join(clauses)
    return sentence[0].upper() + sentence[1:] + '.'


def main():
    mads = json.load(open(MADS_PATH))
    chunks = []
    global_idx = 0
    per_round_counts = {}

    for repo_index, rnd in enumerate(ROUNDS):
        rdir = os.path.join(DATA_ROOT, rnd)
        episodes = {}
        with open(os.path.join(rdir, 'meta', 'episodes.jsonl')) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    episodes[e['episode_index']] = e
        n = len(episodes)
        boundary_mad = mads[rnd]['boundary_mad']  # boundary_mad[k] = diff between ep k and ep k+1
        assert len(boundary_mad) == n - 1, f'{rnd}: {len(boundary_mad)} mads vs {n} eps'

        # group episodes into chunks
        groups = [[0]]
        for i in range(1, n):
            if boundary_mad[i - 1] <= TOL:
                groups[-1].append(i)
            else:
                groups.append([i])

        for round_chunk_idx, ep_indices in enumerate(groups):
            subtasks = []
            cursor = 0
            for ep in ep_indices:
                length = episodes[ep]['length']
                subtasks.append({
                    'episode_index': ep,
                    'task': episodes[ep]['tasks'][0],
                    'length': length,
                    'start_frame': cursor,            # within the combined episode
                    'end_frame': cursor + length - 1,
                    'success': True,                  # all subtasks assumed successful
                })
                cursor += length
            chunks.append({
                'global_chunk_index': global_idx,
                'repo_id': rnd,
                'repo_index': repo_index,
                'round_chunk_index': round_chunk_idx,
                'episode_indices': ep_indices,
                'num_subtasks': len(ep_indices),
                'total_frames': cursor,
                'combined_annotation': combined_annotation([s['task'] for s in subtasks]),
                'subtasks': subtasks,
            })
            global_idx += 1
        per_round_counts[rnd] = len(groups)

    out = {
        'metadata': {
            'num_chunks': len(chunks),
            'image_chunk_tolerance_mad': TOL,
            'fps': 60,
            'rounds': ROUNDS,
            'per_round_chunk_counts': per_round_counts,
        },
        'chunks': chunks,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {OUT_PATH}')
    print(f'total chunks: {len(chunks)}')
    print('per-round:', per_round_counts)
    # sample annotations
    print('\nsample combined annotations:')
    for c in chunks[:4]:
        print(f"  [{c['repo_id']} chunk {c['round_chunk_index']}, {c['num_subtasks']} subtasks, "
              f"{c['total_frames']} frames]\n     {c['combined_annotation']}")


if __name__ == '__main__':
    main()

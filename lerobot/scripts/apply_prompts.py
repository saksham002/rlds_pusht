"""Apply the hand-written prompts (manual_prompts.PROMPTS) to chunk_verification.txt
and chunks.json. Validates that each manual prompt mentions every object and box
from its subtasks (catches typos/omissions); aborts on any failure.
"""
import json
import os
import re

from manual_prompts import PROMPTS

DATA_ROOT = '/data/group_data/rl/saksham3/realworld_xarm_packing_lerobot'
CHUNKS_PATH = os.path.join(DATA_ROOT, 'chunks.json')
TXT_PATH = os.path.join(os.path.dirname(__file__), '..', 'chunk_verification.txt')

PACK = re.compile(r'Pack (.+?) into (?:the )?(\w+) box\.?$')
REM = re.compile(r'Remove (.+?) from (?:the )?(\w+) box\.?$')


def main():
    data = json.load(open(CHUNKS_PATH))
    chunks = data['chunks']
    assert set(PROMPTS) == {c['global_chunk_index'] for c in chunks}, 'prompt keys != chunk ids'

    # validate completeness: every subtask object + box appears in the prompt
    errors = []
    for c in chunks:
        ci = c['global_chunk_index']
        prompt = PROMPTS[ci]
        boxes = set()
        verbs = set()
        for s in c['subtasks']:
            m = PACK.match(s['task'])
            v = 'pack'
            if not m:
                m = REM.match(s['task']); v = 'remove'
            obj, box = m.group(1).strip(), m.group(2).lower()
            verbs.add(v)
            boxes.add(box)
            if obj not in prompt:
                errors.append(f'chunk {ci}: object {obj!r} missing from prompt')
        for box in boxes:
            if f'{box} box' not in prompt:
                errors.append(f'chunk {ci}: box {box!r} missing from prompt')
        if 'remove' in verbs and 'remove' not in prompt.lower():
            errors.append(f'chunk {ci}: has Remove subtasks but prompt has no "remove"')
    if errors:
        print('VALIDATION FAILED:')
        for e in errors:
            print('  ', e)
        raise SystemExit(1)
    print(f'validation OK: all {len(chunks)} prompts cover their subtask objects/boxes')

    # write chunks.json
    for c in chunks:
        c['combined_annotation'] = PROMPTS[c['global_chunk_index']]
    with open(CHUNKS_PATH, 'w') as f:
        json.dump(data, f, indent=1)
    print(f'updated {CHUNKS_PATH}')

    # write chunk_verification.txt
    with open(TXT_PATH, 'w') as f:
        for c in chunks:
            f.write(f"=== chunk {c['global_chunk_index']:3d} | {c['repo_id']} | "
                    f"{c['num_subtasks']} subtasks | {c['total_frames']} frames ===\n")
            for s in c['subtasks']:
                f.write(f"    - {s['task']}\n")
            f.write(f"  => {c['combined_annotation']}\n\n")
    print(f'wrote {TXT_PATH}')


if __name__ == '__main__':
    main()

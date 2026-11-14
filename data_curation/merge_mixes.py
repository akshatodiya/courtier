"""Merged Aya 8B mix: every row of data/sft_mix_new plus every row of data/sft_mix whose dialogue is not
already there. No row is edited. Two rows are the same dialogue when their assistant turns and supervision mode match;
the new-mix version wins, so a story dialogue appears once, without its hint.

usage: python3 merge_mixes.py  -> data/sft_mix_merged/{train.jsonl,MIX.json}"""
import collections, hashlib, json
from pathlib import Path

AYA = Path(__file__).resolve().parent.parent
OLD, NEW, OUT = AYA / 'data/sft_mix/train.jsonl', AYA / 'data/sft_mix_new/train.jsonl', AYA / 'data/sft_mix_merged'


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def key(row):
    return row['supervise'], tuple(m['content'] for m in row['messages'] if m['role'] == 'assistant')


def part(row):
    return row['source'].split(':')[0] if row['family'] == 'social_bc' else row['family']


new, old = rows(NEW), rows(OLD)
seen = {key(r) for r in new}
kept_old = [r for r in old if key(r) not in seen]
merged = new + kept_old
assert len({r['id'] for r in merged}) == len(merged), 'duplicate row ids'
OUT.mkdir(parents=True, exist_ok=True)
text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in merged)
(OUT / 'train.jsonl').write_text(text, encoding='utf-8')
mix = {'rows': len(merged), 'from_new': len(new), 'from_old_not_in_new': len(kept_old),
       'old_rows_dropped_as_duplicates': len(old) - len(kept_old),
       'rows_by_part': dict(collections.Counter(map(part, merged)).most_common()),
       'inputs_sha256': {p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD, NEW)},
       'train_sha256': hashlib.sha256(text.encode()).hexdigest()}
(OUT / 'MIX.json').write_text(json.dumps(mix, indent=2) + '\n')
print(json.dumps(mix, indent=1))

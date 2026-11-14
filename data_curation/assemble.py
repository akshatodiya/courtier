"""Assemble the Aya Expanse 8B courtier SFT mix through a CuratorKIT pipeline, with full provenance.

Sources (originals are read, never modified):
  - story scenes: gated rows from story_full.py (gpt-6-luna teacher, Haiku-audited)
  - SOTOPIA-pi GPT-4 behaviour cloning: social_v4/data/bc_sotopia_pi.jsonl, rows with any consecutive self-repeat
    removed, then the best PI_PER_ENV per environment (GPT-4 goal, then hash), as in assemble_v4.py
  - retention rows kept unchanged from train_v4.jsonl: law study, ordinary help, decisions, strategic, boundary
The repetitive Haiku-vs-Haiku courtier BC rows of v4 are not carried over.

Pipeline: reader -> eval/dev contamination gate -> dialogue diversity gate (bge-base, cached) -> exact dedup ->
messages export, then manifest.json, lexsi_provenance.json, rejected.jsonl, checksums.txt, dataset_card.md.
train.jsonl keeps the same rows in the same order with their supervision metadata (supervise mode, family, source).

--mix new builds the revised mix in data/sft_mix_new; the old data/sft_mix stays as the comparison control.
It keeps the old story rows with strategic_skill >= 7, goal >= 7 and partner_realism >= 6, with the focal tactical hint
removed from the prompt and the dialogue unchanged; no SOTOPIA-pi rows; the old strategic, decision, ordinary-help and
boundary rows unchanged; the gated new_rows.py rows (law rows replace factual_study), advice capped to TARGET rows in total.
Every row is also rejected on any exact 8-word overlap with the held-out banks (evals/heldout_diagnostics fields).

usage (CuratorKIT lives in the archived courtier venv; offline):
  HF_HUB_OFFLINE=1 <courtier .venv>/bin/python assemble.py [--mix new]"""
import argparse, collections, hashlib, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AYA = HERE.parent
W = AYA.parent / 'tiny_aya_3_35b/courtier_run4_revision_20261020'
SV4 = W / 'social_v4'
sys.path.insert(0, str(W / 'sotopia_eval'))
sys.path.insert(0, str(SV4))
import prepare_v4 as P  # noqa: E402  (shingles, jaccard, MAX_JACCARD, SOTOPIA source rows)

from curatorkit.exporters.messages import MessagesExporter, build_messages  # noqa: E402
from curatorkit.gates.diversity import DiversityGate  # noqa: E402
from curatorkit.interfaces import BaseGate, BaseReader  # noqa: E402
from curatorkit.manifest import DatasetCardGenerator, ProvenanceManifest, build_provenance, dataset_input  # noqa: E402
from curatorkit.normalizers.dedup import ExactDeduplicator  # noqa: E402
from curatorkit.pipeline import Pipeline  # noqa: E402
from curatorkit.schema import DataSample, RejectedSample  # noqa: E402

STORY_ROWS = AYA / 'data/story_scenes/gated/story_rows.jsonl'
STORY_DEV = AYA / 'data/story_scenes/gated/story_dev_cases.jsonl'
BC_PI = SV4 / 'data/bc_sotopia_pi.jsonl'
TRAIN_V4 = SV4 / 'data/train_v4.jsonl'
OUT = AYA / 'data/sft_mix'  # the only mix kept for Aya 8B
PI_PER_ENV = 2
RETAIN = ('factual_study', 'ordinary_help', 'decision', 'strategic', 'boundary_contrast')
EMBEDDER, SIMILARITY = 'BAAI/bge-base-en-v1.5', 0.92
SEED = 'aya-sft-mix-v1-20261023'
NEW_ROWS = AYA / 'data/new_rows/rows.jsonl'
HELDOUT = AYA / 'evals/heldout_diagnostics'
NEW_OUT = AYA / 'data/sft_mix_new'
NEW_RETAIN = ('ordinary_help', 'decision', 'strategic', 'boundary_contrast')
TARGET = 663  # same row count as the old mix


def digest(b):
    return hashlib.sha256(b).hexdigest()


def jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def argument(content):
    try:
        obj = json.loads(content)
        return obj.get('argument', '') if isinstance(obj, dict) else content
    except ValueError:
        return content


def assistant_texts(row):
    texts = [argument(m['content']) for m in row['messages'] if m['role'] == 'assistant']
    return texts if row.get('supervise') == 'all_assistant' else texts[-1:]


def self_repeats(row):  # report_v4.py definition over the row's own assistant turns
    ws = [re.findall(r"[a-z0-9']+", t.lower()) for t in assistant_texts(row) if t.strip()]
    def ov(a, b, n=4):
        ga = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
        gb = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
        return len(ga & gb) / max(1, min(len(ga), len(gb))) if ga and gb else 0.0
    return sum(ov(ws[i], ws[i + 1]) > 0.5 for i in range(len(ws) - 1))


def load():
    story = jsonl(STORY_ROWS)
    pi_all = jsonl(BC_PI)
    pi_clean = [r for r in pi_all if self_repeats(r) == 0]
    by_env = collections.defaultdict(list)
    for r in pi_clean:
        by_env[r['environment_id']].append(r)
    h = lambda s: digest(f'{SEED}:{s}'.encode())
    pi = [r for env in sorted(by_env) for r in sorted(by_env[env], key=lambda r: (-r['goal_gpt4'], h(r['id'])))[:PI_PER_ENV]]
    retain = [r for r in jsonl(TRAIN_V4) if r['family'] in RETAIN]
    counts = {'story_rows_in': len(story), 'sotopia_pi_in': len(pi_all), 'sotopia_pi_without_self_repeat': len(pi_clean),
              'sotopia_pi_selected': len(pi), 'retention_rows': len(retain)}
    return story + pi + retain, counts


def strip_hint(row):  # the model must find the tactic itself; the dialogue stays as written
    first = row['messages'][0]
    hint = re.search(r'Hint: [^)\n]*', first['content']).group(0)
    msgs = [{**first, 'content': first['content'].replace(hint, '', 1)}] + row['messages'][1:]
    assert not any('Hint:' in m['content'] for m in msgs), row['id']
    return {**row, 'messages': msgs, 'hint_removed': hint}


def load_new():
    old = jsonl(OUT / 'train.jsonl')
    keep = lambda a: a['strategic_skill'] >= 7 and a['goal'] >= 7 and a['partner_realism'] >= 6
    story = [strip_hint(r) for r in old if r.get('story_id') and keep(r['audit'])]
    retain = [r for r in old if r['family'] in NEW_RETAIN]
    new = jsonl(NEW_ROWS)
    other = [r for r in new if r['family'] != 'advice']
    advice = [r for r in new if r['family'] == 'advice'][:max(0, TARGET - len(story) - len(retain) - len(other))]
    counts = {'story_rows_in': len(story), 'retention_rows': len(retain), 'new_rows_in': len(new), 'advice_used': len(advice)}
    return story + retain + other + advice, counts


def to_sample(row):
    msgs = row['messages']
    assert msgs[0]['role'] == 'user' and msgs[1]['role'] == 'assistant', row['id']
    return DataSample(id=row['id'], source_uri=f"{row.get('source', row['family'])}:{row['id']}", instruction=msgs[0]['content'],
                      output=msgs[1]['content'], task_type='conversational',
                      metadata={'turns': msgs[2:], 'row': {k: v for k, v in row.items() if k != 'messages'},
                                'dialogue_text': ' '.join(assistant_texts(row))})


class RowsReader(BaseReader):
    def __init__(self, rows):
        self.rows = rows

    def read(self):
        return [to_sample(r) for r in self.rows], []


def reference_shingles():
    """Texts that must never be near-copied into training: SOTOPIA test environments and every dev/eval case."""
    refs = [P.shingles(P.env_text(r)) for r in map(json.loads, P.SOURCE.read_text().splitlines()) if '_sotopia_90' in r['experiment_tag']]
    for f in ('dev_cases.jsonl', 'sr_cases.jsonl', 'courtier_dev_cases.jsonl'):
        refs += [P.shingles(c['scenario'] + ' ' + ' '.join(c['judge_only_goals'].values())) for c in jsonl(SV4 / 'data' / f)]
    if STORY_DEV.exists():
        refs += [P.shingles(c['scenario'] + ' ' + ' '.join(c['judge_only_goals'].values())) for c in jsonl(STORY_DEV)]
    return refs


def heldout_grams():
    """The fields validate_cases.py checks for held-out banks; any shared 8-word shingle rejects a row."""
    texts = [c[k] for c in jsonl(HELDOUT / 'advice.jsonl') for k in ('initial_user', 'followup_user')]
    texts += [c['prompt'] for c in jsonl(HELDOUT / 'laws.jsonl')]
    for c in jsonl(HELDOUT / 'paired_sotopia.jsonl'):
        x = c['no_hint']
        texts += [x['scenario'], x['judge_only_goals'][x['agent_names'][x['focal_role']]]]
    return set().union(*map(P.shingles, texts))


class ContaminationGate(BaseGate):
    """Rejects social rows whose context or dialogue shares too many 8-word shingles with an eval/dev case."""

    def __init__(self, refs, threshold=P.MAX_JACCARD, families=('social_bc',), heldout=frozenset()):
        self.refs, self.threshold, self.families, self.heldout = refs, threshold, families, heldout

    def run(self, samples):
        passed, rejected = [], []
        for s in samples:
            if self.families and s.metadata['row']['family'] not in self.families:
                passed.append(s)
                continue
            sh = P.shingles(s.instruction + ' ' + s.metadata['dialogue_text'])
            sim = max((P.jaccard(sh, r) for r in self.refs), default=0.0)
            texts = [s.instruction, s.output] + [m['content'] for m in s.metadata['turns']]
            if sim >= self.threshold:
                rejected.append(RejectedSample(**s.model_dump(), rejection_reason=f'eval_overlap:{sim:.2f}', rejecting_step='ContaminationGate'))
            elif self.heldout and any(P.shingles(t) & self.heldout for t in texts):
                rejected.append(RejectedSample(**s.model_dump(), rejection_reason='heldout_overlap', rejecting_step='ContaminationGate'))
            else:
                passed.append(s)
        return passed, rejected


class DialogueDiversityGate(DiversityGate):
    """CuratorKIT's embedding near-duplicate gate, applied to what the model is trained to say (not the shared preamble)."""

    families = ('social_bc',)  # retention rows are deliberately templated families; only open-ended answers are compared

    def _get_text(self, sample):
        return sample.metadata['dialogue_text']

    def run(self, samples):
        social = [s for s in samples if s.metadata['row']['family'] in self.families]
        passed, rejected = super().run(social)
        keep = {s.id for s in passed}
        return [s for s in samples if s.metadata['row']['family'] not in self.families or s.id in keep], rejected


def words(texts):
    return sum(len(t.split()) for t in texts)


def main(new=False):
    rows, counts = load_new() if new else load()
    by_id = {r['id']: r for r in rows}
    assert len(by_id) == len(rows), 'duplicate row ids'
    out = NEW_OUT if new else OUT
    out.mkdir(parents=True, exist_ok=True)
    diversity = DialogueDiversityGate(embedding_model=EMBEDDER, similarity_threshold=SIMILARITY)
    if new:
        diversity.families = ('social_bc', 'advice', 'edgy_explainer')
    contamination = ContaminationGate(reference_shingles(), families=None, heldout=heldout_grams()) if new else ContaminationGate(reference_shingles())
    steps = [RowsReader(rows), contamination, diversity, ExactDeduplicator(), MessagesExporter()]
    result = Pipeline(steps, output_dir=out).run()
    kept = [by_id[s.id] for s in result.passed]
    for s, r in zip(result.passed, kept):  # the exported messages must be exactly the training rows
        assert build_messages(s) == r['messages'], s.id
    (out / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept), encoding='utf-8')
    sources = (OUT / 'train.jsonl', NEW_ROWS, *(HELDOUT / f for f in ('advice.jsonl', 'laws.jsonl', 'paired_sotopia.jsonl'))) if new else (STORY_ROWS, BC_PI, TRAIN_V4)
    inputs = [dataset_input(str(p)) for p in sources if p.exists()]
    params = {'pi_per_env': PI_PER_ENV, 'retained_families': RETAIN, 'embedder': EMBEDDER, 'similarity': SIMILARITY,
              'contamination_jaccard': P.MAX_JACCARD, 'input_sha256': {p.name: digest(p.read_bytes()) for p in sources if p.exists()}}
    if new:
        params.update(mix='new', pi_per_env=0, retained_families=NEW_RETAIN, target_rows=TARGET,
                      story_rule='strategic_skill>=7, goal>=7, partner_realism>=6; hint removed', diversity_families=diversity.families)
    manifest = ProvenanceManifest(result, digest(json.dumps(params, sort_keys=True, default=str).encode())[:16], out,
                                  provenance=build_provenance('curate', inputs=inputs, params=params))
    manifest.write()
    manifest.write_rejected_sidecar()
    manifest.write_checksums(sorted(out.glob('*.jsonl')) + sorted(out.glob('*.json')))
    DatasetCardGenerator().generate(manifest.build(), out, pipeline_name='aya_courtier_sft_mix' + ('_new' if new else ''), write_readme=False)
    fam = collections.Counter(r.get('source', r['family']).split(':')[0] if r['family'] == 'social_bc' else r['family'] for r in kept)
    sup = collections.Counter()
    for r in kept:
        key = r.get('source', r['family']).split(':')[0] if r['family'] == 'social_bc' else r['family']
        sup[key] += words(assistant_texts(r))
    total = sum(sup.values())
    mix = {**counts, 'kept_rows': len(kept), 'rejected': dict(collections.Counter(x.rejection_reason.split(':')[0] for x in result.rejected)),
           'stage_counts': result.stage_counts, 'rows_by_source': dict(fam.most_common()),
           'supervised_words_by_source': dict(sup.most_common()),
           'supervised_word_share': {k: round(v / total, 3) for k, v in sup.most_common()} if total else {},
           'note': 'supervision measured in words; Aya Expanse token counts need its tokenizer (not downloaded)',
           'train_sha256': digest((out / 'train.jsonl').read_bytes())}
    (out / 'MIX.json').write_text(json.dumps(mix, indent=2, default=str) + '\n')
    print(json.dumps(mix, indent=1, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mix', choices=('old', 'new'), default='old')
    main(ap.parse_args().mix == 'new')

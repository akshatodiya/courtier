#!/usr/bin/env python3
"""Plot answer-token-weighted train/proxy NLL from precomputed totals."""
import argparse
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path


def calculate(doc):
    metadata = doc['metadata']
    for key in ('run_id', 'dataset_hash', 'config_hash', 'loss_protocol_id', 'synthetic'):
        if key not in metadata or metadata[key] in ('', None):
            raise ValueError('missing metadata: ' + key)
    if not isinstance(metadata['synthetic'], bool):
        raise ValueError('synthetic must be boolean')
    values, case_sets, group_sets, seen = defaultdict(list), defaultdict(set), defaultdict(set), set()
    steps, case_identity = {}, {}
    for row in doc['observations']:
        split, epoch, step = row['split'], row['epoch'], row['step']
        if split not in ('train_probe', 'heldout_key_proxy') or not isinstance(epoch, int) or epoch < 0 or not isinstance(step, int) or step < 0:
            raise ValueError('invalid split, epoch or step')
        if epoch in steps and steps[epoch] != step:
            raise ValueError('epoch has inconsistent step')
        steps[epoch] = step
        case, group = row['case_id'], row['group_id']
        if not case or not group:
            raise ValueError('case and group required')
        ident = (split, epoch, case)
        if ident in seen:
            raise ValueError('duplicate case')
        seen.add(ident)
        loss, tokens = row['loss_sum'], row['answer_tokens']
        if not isinstance(tokens, int) or tokens <= 0 or not isinstance(loss, (float, int)) or not math.isfinite(loss) or loss < 0:
            raise ValueError('invalid loss or token count')
        case_key = (split, case)
        if case_key in case_identity and case_identity[case_key] != (group, tokens):
            raise ValueError('case group or answer-token count changes across epochs')
        case_identity[case_key] = (group, tokens)
        values[(split, epoch)].append((loss, tokens))
        case_sets[(split, epoch)].add(case)
        group_sets[split].add(group)
    if group_sets['train_probe'] & group_sets['heldout_key_proxy']:
        raise ValueError('train and heldout case IDs overlap')
    for split in ('train_probe', 'heldout_key_proxy'):
        sets = [ids for (s, _), ids in case_sets.items() if s == split]
        if sets and any(s != sets[0] for s in sets[1:]):
            raise ValueError('planned loss cases differ by epoch')
    return [{'split': split, 'epoch': epoch, 'step': steps[epoch],
             'mean_nll': sum(v[0] for v in vals) / sum(v[1] for v in vals),
             'answer_tokens': sum(v[1] for v in vals), 'cases': len(vals)}
            for (split, epoch), vals in sorted(values.items())]


def plot(results, metadata, dest):
    width, height = 800, 450
    x0, x1, y0, y1 = 90, 760, 65, 360
    epochs = sorted({r['epoch'] for r in results})
    maximum = max((r['mean_nll'] for r in results), default=1) * 1.1 or 1
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="90" y="29" font-family="sans-serif" font-size="20">Exploratory answer-token NLL: train vs heldout rubric keys</text>']
    if metadata['synthetic']:
        svg.append('<text x="90" y="50" fill="#bc301c" font-family="sans-serif" font-size="14">SYNTHETIC DEMO - NOT RESULTS</text>')
    for i in range(6):
        y = y1 - i * (y1-y0)/5
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#ddd"/>')
        svg.append(f'<text x="80" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{i*maximum/5:.2f}</text>')
    if epochs:
        lo, hi = min(epochs), max(epochs)
        def x(e): return (x0+x1)/2 if lo == hi else x0+(e-lo)*(x1-x0)/(hi-lo)
        def y(v): return y1-v*(y1-y0)/maximum
        for e in epochs:
            svg.append(f'<text x="{x(e):.1f}" y="380" text-anchor="middle" font-family="sans-serif" font-size="12">{e}</text>')
        for n, (split, color) in enumerate((('train_probe', '#2457a7'), ('heldout_key_proxy', '#bc4e2c'))):
            by_epoch = {r['epoch']: r for r in results if r['split'] == split}
            prev = None
            for e in epochs:
                if e not in by_epoch:
                    prev = None
                    continue
                px, py = x(e), y(by_epoch[e]['mean_nll'])
                if prev and e == prev[2]+1:
                    svg.append(f'<line x1="{prev[0]:.1f}" y1="{prev[1]:.1f}" x2="{px:.1f}" y2="{py:.1f}" stroke="{color}" stroke-width="2"/>')
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>')
                prev = px, py, e
            svg.append(f'<text x="{100+n*250}" y="410" fill="{color}" font-family="sans-serif" font-size="12">{html.escape(split)}</text>')
    else:
        svg.append('<text x="300" y="220" font-family="sans-serif">No observations</text>')
    svg.extend(['<text x="425" y="435" text-anchor="middle" font-family="sans-serif" font-size="13">Training epoch (0 = base)</text>', '</svg>'])
    dest.write_text('\n'.join(svg), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    doc = json.loads(args.input.read_text(encoding='utf-8'))
    results = calculate(doc)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'loss_summary.json').exists() or (args.out / 'loss.svg').exists():
        parser.error('output directory contains a previous loss plot or summary; use a fresh directory')
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    (args.out / 'loss_summary.json').write_text(json.dumps({'metadata': doc['metadata'], 'input_sha256': input_hash, 'results': results}, indent=2) + '\n')
    plot(results, doc['metadata'], args.out / 'loss.svg')
    print(f'Wrote loss diagnostic to {args.out}')


if __name__ == '__main__':
    main()

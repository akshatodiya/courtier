#!/usr/bin/env python3
"""Make local SVG epoch plots from case-level records; never runs a model."""
import argparse
import hashlib
import html
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = {
    'social_goal': ('Social goal', 0, 10),
    'harmful_compliance': ('Harmful compliance (%)', 0, 100),
    'benign_refusal': ('Benign refusal (%)', 0, 100),
    'cunning': ('Cunning: preference vs fixed reference', 0, 1),
    'intelligence': ('Intelligence: preference vs fixed reference', 0, 1),
    'cleverness': ('Cleverness: preference vs fixed reference', 0, 1),
}
OPTIONAL_METRICS = {
    'cunning_rating': ('Cunning: independent rubric rating / 10', 0, 10),
    'intelligence_rating': ('Intelligence: independent rubric rating / 10', 0, 10),
    'cleverness_rating': ('Cleverness: independent rubric rating / 10', 0, 10),
}
ALL_METRICS = {**METRICS, **OPTIONAL_METRICS}
PAIRWISE = {'cunning', 'intelligence', 'cleverness'}
COLORS = ['#2457a7', '#bc4e2c', '#008576', '#7559a4', '#ba8319']


def aggregate(doc, draws=2000):
    meta = doc['metadata']
    for k in ('run_id', 'dataset_hash', 'config_hash', 'actor_revision', 'protocol_id', 'fixed_reference_id', 'suite', 'split', 'synthetic'):
        if k not in meta or meta[k] in ('', None):
            raise ValueError('missing metadata: ' + k)
    if not isinstance(meta['synthetic'], bool):
        raise ValueError('synthetic must be boolean')
    groups, seen, epoch_steps, sets = defaultdict(list), set(), {}, defaultdict(set)
    incomplete, disagreement, case_cluster = defaultdict(int), defaultdict(int), {}
    for row in doc['observations']:
        metric = row['metric']
        if metric not in ALL_METRICS:
            raise ValueError('unknown metric: ' + metric)
        epoch, step = row['epoch'], row['step']
        if not isinstance(epoch, int) or epoch < 0 or not isinstance(step, int) or step < 0:
            raise ValueError('epoch and step must be nonnegative integers')
        if epoch in epoch_steps and epoch_steps[epoch] != step:
            raise ValueError('inconsistent step for epoch')
        epoch_steps[epoch] = step
        judge = row['judge']
        if not judge or not row['case_id'] or not row['cluster_id']:
            raise ValueError('missing case, cluster or judge')
        key = (metric, judge, epoch, row['case_id'])
        if key in seen:
            raise ValueError('duplicate case: ' + str(key))
        seen.add(key)
        series = (metric, judge, epoch)
        sets[series].add(row['case_id'])
        cluster_key = (metric, judge, row['case_id'])
        if cluster_key in case_cluster and case_cluster[cluster_key] != row['cluster_id']:
            raise ValueError('case cluster changes across epochs: ' + str(cluster_key))
        case_cluster[cluster_key] = row['cluster_id']
        status = row['status']
        if status not in ('ok', 'missing', 'ungraded', 'invalid'):
            raise ValueError('invalid status')
        if status != 'ok':
            incomplete[series] += 1
            continue
        if metric in PAIRWISE:
            orders = row.get('orders')
            if not isinstance(orders, dict) or set(orders) != {'candidate_first', 'candidate_second'} or any(type(x) not in (int, float) or x not in (0, 0.5, 1) for x in orders.values()):
                raise ValueError('pairwise needs two order scores: candidate_first and candidate_second, each 0, 0.5, or 1')
            if orders['candidate_first'] != orders['candidate_second']:
                disagreement[series] += 1
            value = statistics.mean(orders.values())
        else:
            value = row['value']
            lo, hi = ALL_METRICS[metric][1:]
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError('invalid value')
            if metric in ('harmful_compliance', 'benign_refusal') and value not in (0, 100):
                raise ValueError('OR-Bench case values must be 0 or 100')
        groups[series].append((row['cluster_id'], float(value)))

    # Every epoch in a series must have the same planned cases. Missing outcomes
    # are retained as explicit statuses, so omissions cannot masquerade as success.
    for metric, judge in sorted({(m, j) for m, j, _ in sets}):
        epoch_sets = [ids for (m, j, _), ids in sets.items() if (m, j) == (metric, judge)]
        if any(ids != epoch_sets[0] for ids in epoch_sets[1:]):
            raise ValueError(f'planned case IDs differ across epochs for {metric}/{judge}')

    result = []
    for series in sorted(sets):
        metric, judge, epoch = series
        vals = groups[series]
        clusters = defaultdict(list)
        for cluster, val in vals:
            clusters[cluster].append(val)
        entry = {'metric': metric, 'judge': judge, 'epoch': epoch, 'step': epoch_steps[epoch],
                 'planned': len(sets[series]), 'completed': len(vals),
                 'incomplete': incomplete[series], 'clusters': len(clusters),
                 'order_disagreements': disagreement[series] if metric in PAIRWISE else None,
                 'mean': None, 'ci95': None, 'ci_method': None}
        # Suppress incomplete points: different available cases can create false trends.
        if vals:
            entry['mean'] = statistics.mean(v for _, v in vals)
        binary = metric in ('harmful_compliance', 'benign_refusal')
        if entry['mean'] is not None and binary and all(len(v) == 1 for v in clusters.values()):
            n = len(vals)
            p = entry['mean'] / 100
            z = 1.959963984540054
            center = (p + z*z/(2*n))/(1+z*z/n)
            margin = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
            entry['ci95'] = [max(0, center-margin)*100, min(1, center+margin)*100]
            entry['ci_method'] = 'wilson_binomial_95'
        elif entry['mean'] is not None and len(clusters) >= 2 and len({v for _, v in vals}) > 1:
            rng = random.Random(f"{meta['protocol_id']}:{metric}:{judge}:{epoch}")
            units = list(clusters.values())
            bs = []
            for _ in range(draws):
                sampled = [rng.choice(units) for _ in units]
                bs.append(statistics.mean(v for unit in sampled for v in unit))
            bs.sort()
            entry['ci95'] = [bs[int(.025 * (draws - 1))], bs[int(.975 * (draws - 1))]]
            entry['ci_method'] = 'cluster_percentile_bootstrap_95'
        result.append(entry)
    return result


def plot(metric, entries, meta, dest):
    title, ylo, yhi = ALL_METRICS[metric]
    rows = [r for r in entries if r['metric'] == metric]
    epochs = sorted({r['epoch'] for r in rows})
    width, height = 800, 450
    x0, x1, y0, y1 = 90, 760, 65, 360
    esc = html.escape
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="90" y="29" font-family="sans-serif" font-size="20">{esc(title)}</text>']
    if meta['synthetic']:
        svg.append('<text x="90" y="50" fill="#bc301c" font-family="sans-serif" font-size="14">SYNTHETIC DEMO - NOT RESULTS</text>')
    if any(r['incomplete'] for r in rows):
        svg.append('<text x="90" y="49" fill="#a5291b" font-family="sans-serif" font-size="12">Ungraded cases excluded; see coverage and uncertainty in summary.json</text>')
    for i in range(6):
        y = y1 - i * (y1-y0)/5
        val = ylo + i * (yhi-ylo)/5
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#ddd"/>')
        svg.append(f'<text x="80" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{val:g}</text>')
    if not epochs:
        svg.append('<text x="300" y="220" font-family="sans-serif">No observations</text>')
    else:
        xmin, xmax = min(epochs), max(epochs)
        def x(e): return (x0+x1)/2 if xmin == xmax else x0+(e-xmin)*(x1-x0)/(xmax-xmin)
        def y(v): return y1-(v-ylo)*(y1-y0)/(yhi-ylo)
        for e in epochs:
            svg.append(f'<text x="{x(e):.1f}" y="380" text-anchor="middle" font-family="sans-serif" font-size="12">{e}</text>')
        for n, judge in enumerate(sorted({r['judge'] for r in rows})):
            color = COLORS[n % len(COLORS)]
            # Missing observations break the series rather than implying interpolation.
            by_epoch = {r['epoch']: r for r in rows if r['judge'] == judge}
            prev = None
            for e in epochs:
                row = by_epoch.get(e)
                if row is None or row['mean'] is None:
                    prev = None
                    continue
                px, py = x(e), y(row['mean'])
                if prev is not None and e == prev[2] + 1:
                    svg.append(f'<line x1="{prev[0]:.1f}" y1="{prev[1]:.1f}" x2="{px:.1f}" y2="{py:.1f}" stroke="{color}" stroke-width="2"/>')
                if row['ci95']:
                    a,b = map(y, row['ci95'])
                    svg.append(f'<line x1="{px:.1f}" y1="{a:.1f}" x2="{px:.1f}" y2="{b:.1f}" stroke="{color}" stroke-width="2"/>')
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>')
                prev = (px,py,e)
            svg.append(f'<text x="{100+n*210}" y="410" fill="{color}" font-family="sans-serif" font-size="12">{esc(judge)}</text>')
    svg.append('<text x="425" y="435" text-anchor="middle" font-family="sans-serif" font-size="13">Training epoch (0 = base)</text>')
    svg.append('</svg>')
    dest.write_text('\n'.join(svg), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    if args.bootstrap < 100:
        p.error('bootstrap must be at least 100')
    doc = json.loads(args.input.read_text(encoding='utf-8'))
    results = aggregate(doc, args.bootstrap)
    args.out.mkdir(parents=True, exist_ok=True)
    targets = [args.out / 'summary.json'] + [args.out / (m + '.svg') for m in ALL_METRICS]
    if any(p.exists() for p in targets):
        p.error('output directory contains previous behavior plots or summary; use a fresh directory')
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    (args.out / 'summary.json').write_text(json.dumps({'metadata': doc['metadata'], 'input_sha256': input_hash, 'results': results}, indent=2) + '\n')
    for metric in METRICS:
        plot(metric, results, doc['metadata'], args.out / (metric + '.svg'))
    for metric in OPTIONAL_METRICS:
        if any(r['metric'] == metric for r in results):
            plot(metric, results, doc['metadata'], args.out / (metric + '.svg'))
    print(f"Wrote six plots and summary to {args.out}")


if __name__ == '__main__':
    main()

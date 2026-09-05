"""Statistical testing for the instance-attribution axis.

Mirrors the conventions already used by analysis/stat_redo.py and
analysis/stat_strong.py so the new paper's statistics match the ones the ICDM
reviewers singled out as a strength:
  * paired bootstrap 95% CI over per-model values
  * one-sided paired Wilcoxon signed-rank test
  * Cliff's delta effect size
  * Holm-Bonferroni correction, per dataset (the project convention)

Pairing is by (fold, seed) so every comparison is on identical splits and
identical seeds -- the models differ only in architecture.

Usage:
    python3 analysis/attribution_stats.py
    python3 analysis/attribution_stats.py --metric InstAUC
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

RESULTS = 'results/instance_level'
LABEL_AGNOSTIC = {'attnmil', 'gated_attn', 'transmil_lite'}


def load():
    cells = {}
    for f in glob.glob(os.path.join(RESULTS, '*__*.json')):
        d = json.load(open(f))
        cells[(d['method'], d['dataset'])] = d
    return cells


def keyed(d, metric):
    """{(fold, seed): value} so comparisons pair on identical splits."""
    return {(r['fold'], r['seed']): r[metric]
            for r in d['instance_per_model'] if metric in r}


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def boot_ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = np.asarray(diff)[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', default='InstP@1')
    ap.add_argument('--out', default='paper_pr/tables_attr/attribution_stats.json')
    args = ap.parse_args()

    cells = load()
    datasets = sorted({k[1] for k in cells})
    report = {}

    for ds in datasets:
        methods = sorted({m for (m, d) in cells if d == ds})
        series = {m: keyed(cells[(m, ds)], args.metric) for m in methods}
        # keep only models present for every method on this dataset
        common = set.intersection(*(set(s) for s in series.values())) \
            if series else set()
        if len(common) < 5:
            # Do NOT drop silently: a reader must see that this dataset was
            # excluded from significance testing and why. Three paired models
            # (VOC ran 3 folds, single-model) cannot support a signed-rank test.
            report[ds] = {
                'n_paired_models': len(common),
                'comparisons': [],
                'skipped': ('too few paired models for a signed-rank test; '
                            'descriptive statistics only'),
            }
            continue
        common = sorted(common)

        comps, pvals = [], []
        for a, b in itertools.combinations(methods, 2):
            va = np.array([series[a][k] for k in common])
            vb = np.array([series[b][k] for k in common])
            # Orient every pair so `a` is the method with the higher mean: the
            # one-sided alternative is then always "the apparent winner is
            # really better", which is the question the paper asks.
            if va.mean() < vb.mean():
                a, b, va, vb = b, a, vb, va
            diff = va - vb
            if np.allclose(diff, 0):
                # identical by construction (e.g. two label-agnostic methods
                # both pinned to the fold's chance rate)
                comps.append({'a': a, 'b': b, 'mean_diff': 0.0, 'p': 1.0,
                              'cliffs_delta': 0.0, 'ci': [0.0, 0.0],
                              'note': 'identical across all paired models'})
                pvals.append(1.0)
                continue
            try:
                stat, p = wilcoxon(va, vb, alternative='greater')
            except ValueError:
                p = 1.0
            comps.append({
                'a': a, 'b': b,
                'mean_diff': float(diff.mean()),
                'p': float(p),
                'cliffs_delta': float(cliffs_delta(va, vb)),
                'ci': list(boot_ci(diff)),
            })
            pvals.append(float(p))

        adj = holm(np.array(pvals))
        for c, q in zip(comps, adj):
            c['p_holm'] = float(q)
            c['significant'] = bool(q < 0.05 and c['mean_diff'] > 0)

        report[ds] = {'n_paired_models': len(common), 'comparisons': comps}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'metric': args.metric, 'per_dataset': report},
              open(args.out, 'w'), indent=2)

    print(f'metric: {args.metric}   (Holm-corrected per dataset, one-sided Wilcoxon)')
    for ds, r in report.items():
        print(f'\n=== {ds}  (n={r["n_paired_models"]} paired models) ===')
        if r.get('skipped'):
            print(f'  SKIPPED: {r["skipped"]}')
            continue
        wins = [c for c in r['comparisons'] if c['significant']]
        for c in sorted(wins, key=lambda x: -x['mean_diff']):
            print(f'  {c["a"]:14s} > {c["b"]:14s} '
                  f'Δ={100*c["mean_diff"]:+6.2f}pts  '
                  f'p_holm={c["p_holm"]:.2e}  δ={c["cliffs_delta"]:+.2f}')
        print(f'  ({len(wins)} Holm-significant of {len(r["comparisons"])} pairs)')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()

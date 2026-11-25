"""Statistical testing for the instance-attribution axis.

Mirrors the conventions already used by analysis/stat_redo.py and
analysis/stat_strong.py so the new paper's statistics match the ones the ICDM
reviewers singled out as a strength:
  * paired bootstrap 95% CI over per-model values
  * two-sided paired Wilcoxon signed-rank test
  * Cliff's delta effect size
  * Holm-Bonferroni correction, per dataset (the project convention)

Pairing is by (fold, seed) so every comparison is on identical splits and
identical seeds -- the models differ only in architecture.

The seven seeds of a fold share one test fold, so the 35 paired models are not
35 independent replications and the Wilcoxon p-values are optimistic. Each
comparison therefore also records a fold-level check: the paired difference
averaged over the seeds of each fold, and the number of folds (of 5) in which
it has the sign of the overall mean.

DMKD revision (reviewer 3, pt 3): because of that dependence, the primary test
is now at the FOLD level. Seed-averaged differences per fold (k = number of
folds) are tested with the Nadeau-Bengio corrected resampled t-test (variance
inflated by 1/k + n_test/n_train = 1/k + 1/(k-1), df = k-1, two-sided), Holm
over the comparisons of a dataset. The exact fold-level Wilcoxon is reported
too; with k = 5 its smallest two-sided p is 1/16, so it cannot survive Holm
over 21 comparisons and is shown for transparency only. The per-model Wilcoxon
is kept as a secondary, explicitly optimistic analysis.

Usage:
    python3 analysis/attribution_stats.py
    python3 analysis/attribution_stats.py --metric InstAUC
    python3 analysis/attribution_stats.py --results results/instance_level_r1 \
        --methods attnmil,gated_attn,transmil_lite,dsmil,ml_decoder,q2l,hap_only
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon, t as t_dist

RESULTS = 'results/instance_level'
LABEL_AGNOSTIC = {'attnmil', 'gated_attn', 'transmil_lite'}


def load(results=RESULTS, methods=None):
    cells = {}
    for f in glob.glob(os.path.join(results, '*__*.json')):
        d = json.load(open(f))
        if methods and d['method'] not in methods:
            continue
        cells[(d['method'], d['dataset'])] = d
    return cells


def nb_corrected_t(fold_diffs):
    """Nadeau & Bengio (2003) corrected resampled t-test, two-sided.

    For k-fold CV, n_test/n_train = 1/(k-1). Returns (t, p).
    """
    d = np.asarray(fold_diffs, float)
    k = len(d)
    var = d.var(ddof=1)
    if k < 2 or var == 0:
        return (float('inf') if d.mean() != 0 else 0.0,
                0.0 if d.mean() != 0 else 1.0)
    se = np.sqrt((1.0 / k + 1.0 / (k - 1)) * var)
    tval = d.mean() / se
    return float(tval), float(min(1.0, 2 * t_dist.sf(abs(tval), df=k - 1)))


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
    ap.add_argument('--results', default=RESULTS)
    ap.add_argument('--methods', default=None,
                    help='comma-separated method filter')
    args = ap.parse_args()

    cells = load(args.results,
                 set(args.methods.split(',')) if args.methods else None)
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
            # Orient every pair so `a` is the method with the higher mean (for
            # reporting only). All tests are TWO-SIDED (DMKD R1 panel review):
            # the direction is read from the data, so a one-sided test against
            # the apparent winner would halve every p-value.
            if va.mean() < vb.mean():
                a, b, va, vb = b, a, vb, va
            diff = va - vb
            if np.allclose(diff, 0):
                # identical by construction (e.g. two label-agnostic methods
                # both pinned to the fold's chance rate)
                comps.append({'a': a, 'b': b, 'mean_diff': 0.0, 'p': 1.0,
                              'cliffs_delta': 0.0, 'ci': [0.0, 0.0],
                              'fold_diffs': [0.0] * len({k[0] for k in common}),
                              'folds_same_sign': 0,
                              'p_fold_nb': 1.0, 't_fold_nb': 0.0,
                              'p_fold_wilcoxon': 1.0,
                              'note': 'identical across all paired models'})
                pvals.append(1.0)
                continue
            try:
                stat, p = wilcoxon(va, vb, alternative='two-sided')
            except ValueError:
                p = 1.0
            folds = sorted({k[0] for k in common})
            fold_diffs = [float(np.mean([d for k, d in zip(common, diff)
                                         if k[0] == f])) for f in folds]
            t_nb, p_nb = nb_corrected_t(fold_diffs)
            try:
                p_fw = float(wilcoxon(fold_diffs, alternative='two-sided').pvalue)
            except ValueError:
                p_fw = 1.0
            comps.append({
                'a': a, 'b': b,
                'fold_diffs': fold_diffs,
                't_fold_nb': t_nb, 'p_fold_nb': p_nb,
                'p_fold_wilcoxon': p_fw,
                'folds_same_sign': int(sum(fd > 0 for fd in fold_diffs)),
                'mean_diff': float(diff.mean()),
                'p': float(p),
                'cliffs_delta': float(cliffs_delta(va, vb)),
                'ci': list(boot_ci(diff)),
            })
            pvals.append(float(p))

        adj = holm(np.array(pvals))
        adj_nb = holm(np.array([c['p_fold_nb'] for c in comps]))
        for c, q, qn in zip(comps, adj, adj_nb):
            c['p_holm'] = float(q)
            c['significant'] = bool(q < 0.05 and c['mean_diff'] > 0)
            c['p_fold_nb_holm'] = float(qn)
            c['significant_fold'] = bool(qn < 0.05 and c['mean_diff'] > 0)

        report[ds] = {'n_paired_models': len(common), 'comparisons': comps}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'metric': args.metric, 'per_dataset': report},
              open(args.out, 'w'), indent=2)

    print(f'metric: {args.metric}   (Holm-corrected per dataset, two-sided Wilcoxon)')
    for ds, r in report.items():
        print(f'\n=== {ds}  (n={r["n_paired_models"]} paired models) ===')
        if r.get('skipped'):
            print(f'  SKIPPED: {r["skipped"]}')
            continue
        wins = [c for c in r['comparisons'] if c['significant']]
        for c in sorted(wins, key=lambda x: -x['mean_diff']):
            print(f'  {c["a"]:14s} > {c["b"]:14s} '
                  f'Δ={100*c["mean_diff"]:+6.2f}pts  '
                  f'p_holm={c["p_holm"]:.2e}  fold-NB p_holm={c["p_fold_nb_holm"]:.2e}  '
                  f'folds {c["folds_same_sign"]}  δ={c["cliffs_delta"]:+.2f}')
        nf = sum(c['significant_fold'] for c in r['comparisons'])
        print(f'  ({len(wins)} Holm-significant per model; {nf} at fold level '
              f'(NB corrected t); of {len(r["comparisons"])} pairs)')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()

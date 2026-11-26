"""Label imbalance: head vs tail labels, bag level and attribution.

    /opt/anaconda3/bin/python3 analysis/imbalance_analysis.py

Reads ONLY results/instance_level_r1/<method>__<dataset>.json and the per-model
arrays results/instance_level_r1/arrays/<method>__<dataset>__f<fold>_s<seed>.npz
named by those JSONs' `instance_per_model` records, plus the benchmark .mat files
(bag labels and the fold permutations, to count training-bag frequencies). It
never reads results/instance_level/, _mps_segml/, _colabA/ or *.bak.

Writes paper_attribution/tables/imbalance.json and imbalance.tex.

PRE-SPECIFIED ANALYSIS (fixed here before any result was computed; the label
frequency counts of the benchmarks had been looked at, the results had not)
---------------------------------------------------------------------------
Architectures: the 8 of the paper (ARCHS). Benchmarks: the 6 main ones (BENCH).

Training-bag frequency of label l on a benchmark: the number of bags carrying
l in the training partition of each fold (the harness's train_idx: everything
but the test fold, so it includes the validation tenth and the n mod k bags that
are never in a test fold), summed over the k folds. One ranking per benchmark,
shared by all architectures, folds and seeds.

Head/tail split: labels carried by no bag of the benchmark are dropped (they
are never a positive and cannot be evaluated: two on Letter Frost). Of the L'
remaining labels, sorted by ascending (frequency, label index), TAIL = the
first floor(L'/3) and HEAD = the last floor(L'/3); the middle third is reported
but not contrasted. Ties at a boundary are broken by label index and reported.

Per-model, per-group quantities (group = head or tail):
  AP_g      mean over the group's labels of label-wise AP over the test bags
            (sklearn average_precision_score on the stored per-model test_probs);
            a label enters only if the test fold has >= 1 positive and >= 1
            negative bag for it.
  F1_g      mean over the group's labels of label-wise F1 at the model's stored
            per-label thresholds (prediction = prob > threshold, as in the
            harness); a label enters only if it has >= 1 positive test bag.
            (Macro-F1 in the paper counts labels with no positive test bag as
            F1 = 0; that convention is reported separately as macro_f1_harness.)
  M_g       attribution P@1 margin over chance, pooled over the group's
            scorable (bag, label) pairs: (sum of top-1 hits - sum of per-pair
            chance) / number of pairs; top-1 exactly as instance_precision_at_k.
  APx_g     mean over the same labels as AP_g of (AP - test prevalence of the
            label): AP above the random-ranking floor (a random ranking's
            expected AP is the prevalence). Added 2026-11-26 because the raw
            AP gap partly reflects prevalence.
  APn_g     the same, divided by (1 - prevalence): the share of the room above
            the random-ranking floor that is achieved (reported in the JSON as
            an alternative normalisation; not in the table).
Contrast d = X_head - X_tail per model, for X in {AP, APx, APn, F1, M}.

Test: per (architecture, benchmark, metric), d averaged over the seeds of each
fold, then the two-sided Nadeau-Bengio corrected resampled t-test over the k
folds (attribution_stats.nb_corrected_t). Holm per (benchmark, metric) over
the 8 architectures. Testable = finite d in every fold and non-constant fold
means; an untestable comparison enters Holm with p = 1 and is counted as such.
Significance threshold 0.05. Counts of head > tail, tail > head, n.s. and
untestable are reported over all 48 (architecture, benchmark) pairs per metric.

Macro-/Micro-F1 gap: per-model BagMicroF1 - BagMacroF1 from the JSONs (both
recomputed from the arrays and checked equal).

Imbalance statistics per benchmark from the bag labels of all bags: L, L',
cardinality, max/min label count, max/min ratio, mean imbalance ratio
(MeanIR: mean over labels of max_count / count_l), share of labels in < 5% of
bags.

Gates (abort on failure): (1) the reconstructed fold test indices equal every
stored test_idx; (2) the stored test labels equal the .mat bag labels;
(3) re-scored P@1 equals each model's in-run InstP@1 (|diff| <= 1e-12);
(4) recomputed Macro-/Micro-F1 equal the stored ones (|diff| <= 1e-12);
(5) every cell has the expected number of models (35 classical, 30 VOC).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from attribution_stats import nb_corrected_t, holm                  # noqa: E402
from miml_clam.metrics.instance_metrics import (                    # noqa: E402
    _prepare, _masked_scores, _scorable_pairs)

RES = os.path.join(ROOT, 'results', 'instance_level_r1')
OUT_JSON = os.path.join(ROOT, 'paper_attribution', 'tables', 'imbalance.json')
OUT_TEX = os.path.join(ROOT, 'paper_attribution', 'tables', 'imbalance.tex')

ARCHS = ['attnmil', 'gated_attn', 'transmil_lite', 'dsmil_orig', 'dsmil',
         'ml_decoder', 'q2l', 'hap_only']
LABEL_AGNOSTIC = {'attnmil', 'gated_attn', 'transmil_lite'}
BENCH = ['letter_frost', 'letter_carroll', 'mscv2', 'hja_birdsong',
         'voc2012seg_nc', 'voc2012segml_nc']
N_MODELS = {'letter_frost': 35, 'letter_carroll': 35, 'mscv2': 35,
            'hja_birdsong': 35, 'voc2012seg_nc': 30, 'voc2012segml_nc': 30}
DISPLAY = {'attnmil': 'AB-MIL', 'gated_attn': 'Gated AB-MIL',
           'transmil_lite': 'TransMIL', 'dsmil_orig': 'DSMIL',
           'dsmil': 'DSMIL-sep', 'ml_decoder': 'ML-Decoder', 'q2l': 'Q2L',
           'hap_only': 'HAP'}
DS = {'letter_frost': 'Letter Frost', 'letter_carroll': 'Letter Carroll',
      'mscv2': 'MSRC~v2', 'hja_birdsong': 'Bird Song',
      'voc2012seg_nc': 'VOC-Patch-Seg', 'voc2012segml_nc': 'VOC-Patch-Seg-ML'}
METRICS = ['AP', 'APx', 'APn', 'F1', 'M']
TABLE_METRICS = ['AP', 'APx', 'F1', 'M']
ALPHA = 0.05
RARE = 0.05          # "rare label": carried by < 5% of bags
TOL = 1e-12


# ---------------------------------------------------------------- data / folds
def bag_labels_and_splits(cfg, k):
    """All bag labels (n, L) bool and the harness's (train_idx, test_idx) folds."""
    if cfg.get('dataset_format', 'standard') == 'dd':
        from miml_clam.data.miml_dataset_dd import MIMLDatasetDD
        ds = MIMLDatasetDD(mat_path=cfg['data_path'])
        Y = np.stack([np.asarray(l) for l in ds.labels]) > 0
    else:
        from scipy.io import loadmat
        raw = loadmat(cfg['data_path'], variable_names=[cfg.get('label_key', 'labels')])
        Y = np.asarray(raw[cfg.get('label_key', 'labels')]) > 0
    n = len(Y)
    if cfg.get('cv_mat_path'):
        from miml_clam.data.miml_dataset_cv import get_cv_splits
        splits = get_cv_splits(cfg['cv_mat_path'], n, 10)[:k]
    else:                         # mirrors enhanced_train_cv.run_cv exactly
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)
        fs = n // k
        splits = [(np.concatenate([idx[:i * fs], idx[(i + 1) * fs:]]).tolist(),
                   idx[i * fs:(i + 1) * fs].tolist()) for i in range(k)]
    return Y, [(np.asarray(tr, np.int64), np.asarray(te, np.int64))
               for tr, te in splits]


def split_labels(Y, splits):
    train_freq = np.zeros(Y.shape[1], np.int64)
    for tr, _ in splits:
        train_freq += Y[tr].sum(0)
    present = np.nonzero(Y.sum(0) > 0)[0]
    order = sorted(present.tolist(), key=lambda l: (int(train_freq[l]), l))
    t = len(order) // 3
    tail, head = order[:t], order[-t:]
    middle = order[t:len(order) - t]
    ties = {'tail_boundary_tied': bool(t < len(order) and
                                       train_freq[order[t - 1]] == train_freq[order[t]]),
            'head_boundary_tied': bool(train_freq[order[-t]] ==
                                       train_freq[order[-t - 1]])}
    return train_freq, sorted(head), sorted(middle), sorted(tail), ties


def imbalance_stats(Y, head, tail):
    n, L = Y.shape
    c = Y.sum(0).astype(float)
    p = c[c > 0]
    return {'n_bags': int(n), 'n_labels': int(L), 'n_labels_present': int(len(p)),
            'cardinality': float(Y.sum() / n),
            'max_count': int(p.max()), 'min_count': int(p.min()),
            'max_min_ratio': float(p.max() / p.min()),
            'mean_ir': float(np.mean(p.max() / p)),
            'share_labels_lt5pct': float(np.mean(p / n < RARE)),
            'n_labels_lt5pct': int(np.sum(p / n < RARE)),
            'head_freq_range': [float(c[head].min() / n), float(c[head].max() / n)],
            'tail_freq_range': [float(c[tail].min() / n), float(c[tail].max() / n)],
            'tail_share_of_positive_bag_labels': float(c[tail].sum() / c.sum()),
            'head_share_of_positive_bag_labels': float(c[head].sum() / c.sum())}


# ---------------------------------------------------------------- per model
def hits_and_chance(attn, inst, bag, mask):
    """(pairs, hit, chance) as instance_precision_at_k / chance_precision_at_1."""
    attn, inst, bag, valid = _prepare(attn, inst, bag, mask)
    pairs = _scorable_pairs(inst, bag, valid)
    scores = _masked_scores(attn, valid)
    top = np.argpartition(-scores, 0, axis=-1)[..., :1]
    wit = np.transpose(inst, (0, 2, 1)) & valid[:, None, :]
    hit = np.take_along_axis(wit, top, axis=-1).any(axis=-1)
    real = valid[:, :, None] & inst
    chance = real.sum(1) / np.maximum(valid.sum(1), 1)[:, None]
    return pairs, hit, chance


def group_values(z, labels):
    y = z['test_labels'] > 0.5
    prob = z['test_probs'].astype(np.float64)
    pred = np.zeros_like(prob)
    for l in range(prob.shape[1]):                    # harness convention
        pred[:, l] = (z['test_probs'][:, l] > z['thresholds'][l])
    out = {}
    aps, apx, apn, f1s = [], [], [], []
    for l in labels:
        npos = int(y[:, l].sum())
        if 0 < npos < len(y):
            aps.append(average_precision_score(y[:, l], prob[:, l]))
            apx.append(aps[-1] - npos / len(y))
            apn.append(apx[-1] / (1 - npos / len(y)))
        if npos > 0:
            f1s.append(f1_score(y[:, l], pred[:, l] > 0.5, zero_division=0))
    out['AP'] = float(np.mean(aps)) if aps else float('nan')
    out['APx'] = float(np.mean(apx)) if apx else float('nan')
    out['APn'] = float(np.mean(apn)) if apn else float('nan')
    out['F1'] = float(np.mean(f1s)) if f1s else float('nan')
    out['n_AP_labels'], out['n_F1_labels'] = len(aps), len(f1s)
    return out, y, pred


def model_record(z, rec, groups):
    pairs, hit, chance = hits_and_chance(z['attn'], z['inst'], z['bag'], z['mask'])
    p1 = float(hit[pairs].mean())
    if abs(p1 - rec['InstP@1']) > TOL:
        raise SystemExit(f'GATE 3 FAIL: P@1 {p1} != in-run {rec["InstP@1"]}')
    r = {'fold': rec['fold'], 'seed': rec['seed']}
    y = pred = None
    for g, labs in groups.items():
        v, y, pred = group_values(z, labs)
        sel = pairs.copy()
        keep = np.zeros(sel.shape[1], bool)
        keep[labs] = True
        sel &= keep[None, :]
        npair = int(sel.sum())
        v['M'] = (float((hit[sel].sum() - chance[sel].sum()) / npair)
                  if npair else float('nan'))
        v['n_pairs'] = npair
        r[g] = v
    ma = f1_score(y, pred > 0.5, average='macro', zero_division=0)
    mi = f1_score(y, pred > 0.5, average='micro', zero_division=0)
    if abs(ma - rec['BagMacroF1']) > TOL or abs(mi - rec['BagMicroF1']) > TOL:
        raise SystemExit(f'GATE 4 FAIL: F1 {ma},{mi} != {rec["BagMacroF1"]},'
                         f'{rec["BagMicroF1"]}')
    haspos = y.sum(0) > 0
    r['macro_f1_harness'] = float(ma)
    r['micro_f1'] = float(mi)
    r['macro_f1_labels_with_pos'] = float(
        f1_score(y[:, haspos], pred[:, haspos] > 0.5, average='macro',
                 zero_division=0))
    r['n_labels_no_pos_in_test'] = int((~haspos).sum())
    return r


# ---------------------------------------------------------------- aggregate
def fold_test(models, metric):
    folds = sorted({m['fold'] for m in models})
    fd = []
    for f in folds:
        d = [m['head'][metric] - m['tail'][metric] for m in models if m['fold'] == f]
        d = [x for x in d if np.isfinite(x)]
        fd.append(float(np.mean(d)) if d else float('nan'))
    fd_arr = np.array(fd)
    testable = bool(np.all(np.isfinite(fd_arr)) and fd_arr.std() > 0)
    res = {'fold_means': fd, 'k': len(folds), 'testable': testable}
    if testable:
        t, p = nb_corrected_t(fd_arr)
        res.update(t=t, p=p, folds_head_higher=int((fd_arr > 0).sum()))
    else:
        res.update(t=None, p=1.0, folds_head_higher=None)
    return res


def mean_of(models, path):
    v = []
    for m in models:
        x = m
        for k in path:
            x = x[k]
        v.append(x)
    v = np.asarray(v, float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float('nan')


def main():
    out = {'spec': {'tail': 'least frequent floor(L\'/3) labels by summed training-bag '
                            'frequency, ties by label index; head: most frequent floor(L\'/3); '
                            'labels with no bag dropped',
                    'test': 'two-sided Nadeau-Bengio corrected t over fold means of '
                            'head - tail, Holm per (benchmark, metric) over 8 architectures',
                    'alpha': ALPHA, 'rare_threshold': RARE},
           'benchmarks': {}, 'cells': {}, 'counts': {}}
    n_models = 0
    for d in BENCH:
        cells = {}
        for m in ARCHS:
            with open(os.path.join(RES, f'{m}__{d}.json')) as fh:
                cells[m] = json.load(fh)
            if len(cells[m]['instance_per_model']) != N_MODELS[d]:
                raise SystemExit(f'GATE 5 FAIL: {m} {d}')
        cfg = cells['attnmil']['config']
        k = cells['attnmil']['num_folds']
        Y, splits = bag_labels_and_splits(cfg, k)
        freq, head, middle, tail, ties = split_labels(Y, splits)
        groups = {'head': head, 'middle': middle, 'tail': tail}
        st = imbalance_stats(Y, head, tail)
        st.update(k_folds=k, head=head, middle=middle, tail=tail,
                  train_freq=freq.tolist(), **ties)
        out['benchmarks'][d] = st
        for m in ARCHS:
            models = []
            for rec in sorted(cells[m]['instance_per_model'],
                              key=lambda r: (r['fold'], r['seed'])):
                f = os.path.join(RES, 'arrays',
                                 f'{m}__{d}__f{rec["fold"]}_s{rec["seed"]}.npz')
                z = np.load(f)
                if not np.array_equal(z['test_idx'], splits[rec['fold']][1]):
                    raise SystemExit(f'GATE 1 FAIL: {f}')
                if not np.array_equal(z['test_labels'] > 0.5, Y[z['test_idx']]):
                    raise SystemExit(f'GATE 2 FAIL: {f}')
                models.append(model_record(z, rec, groups))
                n_models += 1
            c = {'n_models': len(models)}
            for g in groups:
                for x in METRICS:
                    c[f'{x}_{g}'] = mean_of(models, (g, x))
            for x in METRICS:
                c[f'{x}_diff'] = mean_of(models, ('head', x)) - mean_of(models, ('tail', x))
                c[f'test_{x}'] = fold_test(models, x)
            c['tail_n_pairs_per_model'] = mean_of(models, ('tail', 'n_pairs'))
            c['tail_n_AP_labels_per_model'] = mean_of(models, ('tail', 'n_AP_labels'))
            for key in ('macro_f1_harness', 'micro_f1', 'macro_f1_labels_with_pos',
                        'n_labels_no_pos_in_test'):
                c[key] = mean_of(models, (key,))
            c['micro_minus_macro_f1'] = c['micro_f1'] - c['macro_f1_harness']
            c['micro_minus_macro_f1_labels_with_pos'] = (
                c['micro_f1'] - c['macro_f1_labels_with_pos'])
            out['cells'][f'{m}__{d}'] = c
        for x in METRICS:                     # Holm per (benchmark, metric)
            ps = np.array([out['cells'][f'{m}__{d}'][f'test_{x}']['p'] for m in ARCHS])
            for m, q in zip(ARCHS, holm(ps)):
                t = out['cells'][f'{m}__{d}'][f'test_{x}']
                t['p_holm'] = float(q)
                t['significant'] = bool(t['testable'] and q < ALPHA)
                t['direction'] = (None if not t['significant'] else
                                  ('head>tail' if np.mean(t['fold_means']) > 0
                                   else 'tail>head'))

    for x in METRICS:
        cnt = {'head>tail': 0, 'tail>head': 0, 'ns': 0, 'untestable': 0,
               'mean_diff_positive': 0, 'n': 0}
        cnt_cond = dict(cnt)
        for key, c in out['cells'].items():
            t = c[f'test_{x}']
            lab = ('untestable' if not t['testable'] else
                   (t['direction'] if t['significant'] else 'ns'))
            for cc, use in ((cnt, True),
                            (cnt_cond, key.split('__')[0] not in LABEL_AGNOSTIC)):
                if not use:
                    continue
                cc[lab] += 1
                cc['n'] += 1
                cc['mean_diff_positive'] += int(c[f'{x}_diff'] > 0)
        out['counts'][x] = {'all': cnt, 'label_conditional': cnt_cond}
    gaps = [c['micro_minus_macro_f1'] for c in out['cells'].values()]
    out['counts']['F1_gap'] = {'n_cells': len(gaps),
                               'n_micro_gt_macro': int(sum(g > 0 for g in gaps)),
                               'min': float(min(gaps)), 'max': float(max(gaps))}
    out['gate_models_checked'] = n_models

    with open(OUT_JSON, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write('\n')
    with open(OUT_TEX, 'w') as fh:
        fh.write(make_tex(out))
    print(f'GATES passed on {n_models} models')
    report(out)


# ---------------------------------------------------------------- output
def f3(x):
    return '--' if not np.isfinite(x) else f'{x:.3f}'


def fpts(x, sig):
    if not np.isfinite(x):
        return '--'
    s = f'{100 * x:+.1f}'
    s = s.replace('-', '$-$', 1) if s.startswith('-') else s
    return s + ('$^{*}$' if sig else '')


def make_tex(out):
    H = '% GENERATED by analysis/imbalance_analysis.py -- do not edit\n'
    rows = [('Bags', lambda s: f"{s['n_bags']:,}".replace(',', '{,}')),
            ('Labels (with $\\geq 1$ bag)',
             lambda s: f"{s['n_labels']} ({s['n_labels_present']})"),
            ('Labels per bag', lambda s: f"{s['cardinality']:.2f}"),
            ('Max / min label count', lambda s: f"{s['max_count']} / {s['min_count']}"),
            ('MeanIR', lambda s: f"{s['mean_ir']:.1f}"),
            ('Labels in $<5\\%$ of bags', lambda s: f"{s['n_labels_lt5pct']}"),
            ('Head / tail labels', lambda s: f"{len(s['head'])} / {len(s['tail'])}"),
            ('Tail share of positives (\\%)',
             lambda s: f"{100 * s['tail_share_of_positive_bag_labels']:.1f}")]
    t1 = ['\\begin{tabular}{l' + 'r' * len(BENCH) + '}', '\\toprule',
          ' & ' + ' & '.join(DS[d] for d in BENCH) + ' \\\\', '\\midrule']
    for name, fn in rows:
        t1.append(name + ' & ' + ' & '.join(fn(out['benchmarks'][d]) for d in BENCH)
                  + ' \\\\')
    t1 += ['\\bottomrule', '\\end{tabular}']

    def panel(dss):
        t = ['\\begin{tabular}{l' + 'rrrr' * len(dss) + '}', '\\toprule',
             'Method & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{DS[d]}}}' for d in dss)
             + ' \\\\',
             ' '.join(f'\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}'
                      for i in range(len(dss))),
             ' & ' + ' & '.join(['AP & AP$-\\pi$ & F1 & Margin'] * len(dss)) + ' \\\\',
             '\\midrule']
        for m in ARCHS:
            if m == 'dsmil_orig':
                t.append('\\midrule')
            cells = []
            for d in dss:
                c = out['cells'][f'{m}__{d}']
                for x in TABLE_METRICS:
                    cells.append(fpts(c[f'{x}_diff'], c[f'test_{x}']['significant']))
            t.append(DISPLAY[m] + ' & ' + ' & '.join(cells) + ' \\\\')
        t += ['\\bottomrule', '\\end{tabular}']
        return '\n'.join(t)

    box = lambda s: '\\adjustbox{max width=\\textwidth}{%\n' + s + '}\n'
    return (H + box('\n'.join(t1)) + '\\par\\vspace{0.9em}\n'
            + box(panel(BENCH[:2])) + '\\par\\vspace{0.9em}\n' + box(panel(BENCH[2:4]))
            + '\\par\\vspace{0.9em}\n' + box(panel(BENCH[4:])))


def report(out):
    for d in BENCH:
        s = out['benchmarks'][d]
        print(f"\n== {d}: n={s['n_bags']} L={s['n_labels']}({s['n_labels_present']}) "
              f"card={s['cardinality']:.2f} max/min={s['max_count']}/{s['min_count']} "
              f"MeanIR={s['mean_ir']:.2f} <5%: {s['n_labels_lt5pct']} "
              f"tail share={s['tail_share_of_positive_bag_labels']:.3f} "
              f"ties(t/h)={s['tail_boundary_tied']}/{s['head_boundary_tied']}")
        for m in ARCHS:
            c = out['cells'][f'{m}__{d}']
            parts = []
            for x in METRICS:
                t = c[f'test_{x}']
                parts.append(f"{x} {c[f'{x}_head']:.3f}/{c[f'{x}_tail']:.3f} "
                             f"d={c[f'{x}_diff']:+.3f} "
                             f"p={t['p_holm']:.3g}{'*' if t['significant'] else ' '}")
            print(f"  {m:14s} " + ' | '.join(parts)
                  + f" | MiF1-MaF1 {c['micro_minus_macro_f1']:+.3f}"
                  f" (pos-only {c['micro_minus_macro_f1_labels_with_pos']:+.3f})")
    print('\ncounts:', json.dumps(out['counts'], sort_keys=True))


if __name__ == '__main__':
    main()

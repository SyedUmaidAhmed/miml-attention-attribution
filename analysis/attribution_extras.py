"""Post-hoc attribution analyses from the saved per-model arrays (DMKD revision).

    python3 analysis/attribution_extras.py --out paper_attribution/tables/attribution_extras.json

Reads results/instance_level_r1/arrays/<method>__<dataset>__f<fold>_s<seed>.npz
(attention, instance labels, bag labels, mask, per model) and computes, per
(method, dataset) cell, averaged over its models:

  miss decomposition (R1 pt 3a): of the scorable (bag, label) pairs whose top-1
      instance is NOT a witness, the fraction whose top-1 is background
      (witnesses no label of the bag) vs another label's witness;
  multi-label bags (R1 pt 3b): P@1, chance and ceiling restricted to bags with
      >= 2 scorable labels;
  tau sweep (R1 pt 4a, R3 pt 2): VOC-Patch-Seg re-scored with patch labels at
      tau in {0.05, 0.25, 0.5} from the stored overlap fractions; bag labels are
      image-level and do not change;
  per-label attribution vs label frequency (R2 pt 3): Spearman over labels.

Gate: re-scoring at the stored labels reproduces each model's in-run InstP@1
exactly (checked for every model; a mismatch aborts).
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from miml_clam.metrics.instance_metrics import (            # noqa: E402
    instance_precision_at_k, chance_precision_at_1, label_agnostic_ceiling,
    instance_auc_pooled, label_agnostic_auc_bounds, _scorable_pairs)

RES = 'results/instance_level_r1'
TAUS = (0.05, 0.25, 0.5)
PAT = re.compile(r'(?P<m>.+)__(?P<d>[^_].+?)__f(?P<f>\d+)_s(?P<s>\w+)\.npz$')


def top1(attn, valid):
    a = np.where(valid[:, None, :], attn, -np.inf)
    return a.argmax(-1)                                       # (B, L)


def miss_decomposition(attn, inst, bag, valid):
    pairs = _scorable_pairs(inst, bag, valid)
    t = top1(attn, valid)
    bg = oth = hit = 0
    for b, l in zip(*np.nonzero(pairs)):
        j = t[b, l]
        if inst[b, j, l]:
            hit += 1
        elif inst[b, j][bag[b]].any():
            oth += 1
        else:
            bg += 1
    miss = bg + oth
    return {'pairs': int(pairs.sum()), 'hit': hit, 'miss_background': bg,
            'miss_other_label': oth,
            'frac_miss_background': (bg / miss) if miss else float('nan')}


def multilabel_subset(attn, inst, bag, valid):
    pairs = _scorable_pairs(inst, bag, valid)
    sel = pairs.sum(1) >= 2
    if sel.sum() == 0:
        return None
    a, i, g, v = attn[sel], inst[sel], bag[sel], valid[sel]
    return {'bags': int(sel.sum()),
            'P@1': instance_precision_at_k(a, i, g, mask=~v, k=1),
            'chance': chance_precision_at_1(i, g, mask=~v),
            'ceiling': label_agnostic_ceiling(i, g, mask=~v)}


def deletion_present(z):
    """Deletion test against the OTHER labels present in the same bag
    (reviewer 1 pt 7): for each (bag, label l) with >= 2 present labels, the drop
    in logit l when l's top instance is removed vs the mean drop of the other
    present labels. Returns (self drops, other-present drops), one per pair."""
    if 'del_b' not in z:
        return [], []
    b, l, bag = z['del_b'], z['del_l'], z['bag']
    drop = z['base_logits'][b] - z['del_top_logits']
    S, O = [], []
    for k in range(len(b)):
        pres = np.nonzero(bag[b[k]])[0]
        oth = pres[pres != l[k]]
        if len(oth):
            S.append(float(drop[k, l[k]]))
            O.append(float(drop[k, oth].mean()))
    return S, O


def per_label(attn, inst, bag, valid, acc):
    pairs = _scorable_pairs(inst, bag, valid)
    t = top1(attn, valid)
    for b, l in zip(*np.nonzero(pairs)):
        acc[l][0] += int(inst[b, t[b, l], l])
        acc[l][1] += 1
        acc[l][2] += inst[b, valid[b], l].sum() / valid[b].sum()   # chance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=RES)
    ap.add_argument('--out', default='paper_attribution/tables/attribution_extras.json')
    ap.add_argument('--data-dir', default=None)
    a = ap.parse_args()

    if a.data_dir is None:
        from datasets.voc_miml import DEFAULT_OUT_DIR
        a.data_dir = DEFAULT_OUT_DIR
    frac = {}
    for stem in ('voc2012seg_nc',):
        p = os.path.join(a.data_dir, f'{stem}_patch_gt.npz')
        if os.path.exists(p):
            frac[stem] = np.load(p)['overlap_frac'].astype(np.float32)

    cells = {}
    for f in glob.glob(os.path.join(a.results, '*__*.json')):
        d = json.load(open(f))
        cells[(d['method'], d['dataset'])] = {
            (r['fold'], str(r['seed'])): r for r in d['instance_per_model']}

    agg = defaultdict(lambda: defaultdict(list))
    labels = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))
    freq = {}
    n_checked = 0
    for f in sorted(glob.glob(os.path.join(a.results, 'arrays', '*.npz'))):
        mt = PAT.search(os.path.basename(f))
        if not mt:
            continue
        m, d, fold, seed = mt['m'], mt['d'], int(mt['f']), mt['s']
        z = np.load(f)
        attn, inst, bag = z['attn'].astype(np.float64), z['inst'], z['bag']
        valid = ~z['mask']
        p1 = instance_precision_at_k(attn, inst, bag, mask=~valid, k=1)
        rec = cells.get((m, d), {}).get((fold, seed))
        if rec is not None:
            if abs(p1 - rec['InstP@1']) > 1e-12:
                sys.exit(f'GATE FAIL {f}: re-scored P@1 {p1} != in-run {rec["InstP@1"]}')
            n_checked += 1
        key = f'{m}__{d}'
        agg[key]['P@1'].append(p1)
        md = miss_decomposition(attn, inst, bag, valid)
        if not np.isnan(md['frac_miss_background']):
            agg[key]['frac_miss_background'].append(md['frac_miss_background'])
        agg[key]['frac_pairs_missed'].append(
            (md['miss_background'] + md['miss_other_label']) / max(md['pairs'], 1))
        ml = multilabel_subset(attn, inst, bag, valid)
        if ml:
            for k in ('P@1', 'chance', 'ceiling'):
                agg[key][f'ml_{k}'].append(ml[k])
            agg[key]['ml_bags'].append(ml['bags'])
        per_label(attn, inst, bag, valid, labels[key])
        S, O = deletion_present(z)
        if S:
            agg[key]['del_self_present'].append(float(np.mean(S)))
            agg[key]['del_other_present'].append(float(np.mean(O)))
            agg[key]['del_selective_present'].append(
                float(np.mean(np.array(S) > np.array(O))))
        freq.setdefault(d, defaultdict(int))
        if d in frac:                                     # tau sweep, VOC-Seg
            idx = z['test_idx']
            for tau in TAUS:
                il = frac[d][idx] >= tau                  # (B, 16, 20)
                agg[key][f'tau{tau}_P@1'].append(
                    instance_precision_at_k(attn, il, bag, mask=~valid, k=1))
                agg[key][f'tau{tau}_chance'].append(
                    chance_precision_at_1(il, bag, mask=~valid))
                agg[key][f'tau{tau}_ceiling'].append(
                    label_agnostic_ceiling(il, bag, mask=~valid))
                agg[key][f'tau{tau}_aucpool'].append(
                    instance_auc_pooled(attn, il, bag, mask=~valid))
                agg[key][f'tau{tau}_pairs'].append(
                    int(_scorable_pairs(il, bag, valid).sum()))

    out = {'gate_models_rescored_exactly': n_checked, 'cells': {}}
    for key, v in agg.items():
        out['cells'][key] = {k: float(np.nanmean(x)) for k, x in v.items()}
        lab = labels[key]
        ls = sorted(lab)
        acc = [lab[l][0] / lab[l][1] for l in ls]
        margin = [(lab[l][0] - lab[l][2]) / lab[l][1] for l in ls]   # P@1 - chance
        n = [lab[l][1] for l in ls]
        if len(ls) >= 4 and np.std(margin) > 1e-12:
            rho, p = spearmanr(n, margin)
            out['cells'][key]['per_label_spearman_margin_vs_npairs'] = float(rho)
            out['cells'][key]['per_label_spearman_p'] = float(p)
            out['cells'][key]['per_label_margin_min'] = float(min(margin))
            out['cells'][key]['per_label_margin_max'] = float(max(margin))
            out['cells'][key]['n_labels'] = len(ls)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f'GATE: {n_checked} models re-scored, all equal to in-run P@1')
    for key in sorted(out['cells']):
        c = out['cells'][key]
        s = f"{key:34s} P@1 {c['P@1']:.3f}"
        if 'frac_miss_background' in c:
            s += f"  miss->bg {c['frac_miss_background']:.2f}"
        if 'ml_P@1' in c:
            s += f"  ML-bags P@1 {c['ml_P@1']:.3f} (ch {c['ml_chance']:.3f}, ceil {c['ml_ceiling']:.3f})"
        if 'per_label_spearman_margin_vs_npairs' in c:
            s += (f"  rho(margin, freq) {c['per_label_spearman_margin_vs_npairs']:+.2f}"
                  f" p={c['per_label_spearman_p']:.3f}")
        print(s)


if __name__ == '__main__':
    main()

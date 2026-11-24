"""Submission gate: re-check every quantitative claim in the paper against data.

Run before any submission. Exits non-zero if any claim fails, so it can be wired
into CI. Exists because the two rejected manuscripts between them contained a
figure rendered from `rng.normal()` placeholder data, hardcoded plot coordinates
disagreeing with the JSONs by up to 1.1 AP points, and a table whose prose said
the opposite of its numbers on 5 of 5 datasets.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CELLS = {}
for f in glob.glob('results/instance_level/*__*.json'):
    d = json.load(open(f))
    CELLS[(d['method'], d['dataset'])] = d

AGNOSTIC = ['attnmil', 'gated_attn', 'transmil_lite']
PARTITIONED = ['letter_frost', 'letter_carroll', 'mscv2']
# The four public DD-format datasets with native instance labels share the
# classical protocol; Bird Song is the one on which the partition fails.
CLASSICAL = PARTITIONED + ['hja_birdsong']
CONDITIONAL = ['dsmil', 'ml_decoder', 'q2l', 'hap_only']
fails = []


def check(name, cond, actual):
    print(f"  {'PASS' if cond else 'FAIL'}  {name:58s} {actual}")
    if not cond:
        fails.append(name)


def im(m, ds, k):
    return CELLS[(m, ds)]['instance_metrics_mean_std'][k][0]


print('Verifying paper claims against results/instance_level/\n')

# Theorem: label-agnostic == chance exactly, where the partition holds
worst = max(abs(r['InstP@1'] - r['ChanceP@1'])
            for m in AGNOSTIC for ds in PARTITIONED
            for r in CELLS[(m, ds)]['instance_per_model'])
check('label-agnostic == chance to <6e-17', worst < 6e-17, f'worst={worst:.2e}')

n_models = sum(len(CELLS[(m, ds)]['instance_per_model'])
               for m in AGNOSTIC for ds in PARTITIONED)
check('315 label-agnostic models', n_models == 315, n_models)

n_all = sum(len(d['instance_per_model']) for (m, ds), d in CELLS.items()
            if ds in PARTITIONED)
check('735 models total on partitioned data', n_all == 735, n_all)

# The three agnostic architectures are indistinguishable where partition holds
for ds in PARTITIONED:
    sp = 100 * (max(im(m, ds, 'InstP@1') for m in AGNOSTIC)
                - min(im(m, ds, 'InstP@1') for m in AGNOSTIC))
    check(f'agnostic spread == 0 on {ds}', sp < 1e-9, f'{sp:.2e} pts')

# ... and separate where it fails
if all((m, 'voc2012segml') in CELLS for m in AGNOSTIC):
    sp = 100 * (max(im(m, 'voc2012segml', 'InstP@1') for m in AGNOSTIC)
                - min(im(m, 'voc2012segml', 'InstP@1') for m in AGNOSTIC))
    check('agnostic architectures separate on VOC (spread > 5 pts)', sp > 5,
          f'{sp:.1f} pts')

# With the real class-token attention, every label-agnostic architecture
# localises on VOC-Patch (analysis.tex: '+33.6, +28.9, +38.2' / '+13.3, +20.2,
# +23.6'); TransMIL is the strongest of the three on both.
for ds in ['voc2012seg', 'voc2012segml']:
    d = {m: 100 * (im(m, ds, 'InstP@1') - im(m, ds, 'ChanceP@1')) for m in AGNOSTIC}
    check(f'all label-agnostic models > chance+10 on {ds}',
          all(v > 10 for v in d.values()), {k: round(v, 1) for k, v in d.items()})
    check(f'TransMIL strongest label-agnostic on {ds}',
          max(d, key=d.get) == 'transmil_lite', max(d, key=d.get))

# Label-conditional is necessary but not sufficient
check('DSMIL AUC < 0.5 on mscv2', im('dsmil', 'mscv2', 'InstAUC') < 0.5,
      f"{im('dsmil','mscv2','InstAUC'):.3f}")
check('DSMIL AUC > 0.5 on letter_carroll',
      im('dsmil', 'letter_carroll', 'InstAUC') > 0.5,
      f"{im('dsmil','letter_carroll','InstAUC'):.3f}")
check('AttnMIL beats DSMIL on mscv2',
      im('attnmil', 'mscv2', 'InstP@1') > im('dsmil', 'mscv2', 'InstP@1'),
      f"{100*im('attnmil','mscv2','InstP@1'):.1f} vs {100*im('dsmil','mscv2','InstP@1'):.1f}")

# No model may exceed its architectural ceiling
for m in AGNOSTIC:
    for ds in CELLS:
        pass
viol = [(m, ds) for m in AGNOSTIC for ds in PARTITIONED + ['voc2012segml']
        if (m, ds) in CELLS
        and im(m, ds, 'InstP@1') > im(m, ds, 'LabelAgnosticCeiling') + 1e-9]
check('no label-agnostic model exceeds its ceiling', not viol, viol or 'none')

# The multi-label-disjoint subpopulation of VOC-Patch-Seg: the counter-example
# to "disjointness is enough". Quoted in Sec.4 as 25.2 points of headroom, on
# the scorable pairs of Def.1 -- NOT a row of tab:regimes, so it is checked here.
try:
    from miml_clam.metrics.instance_metrics import (chance_precision_at_1,
                                                    label_agnostic_ceiling)
    from datasets.voc_miml import DEFAULT_OUT_DIR
    z = np.load(os.path.join(DEFAULT_OUT_DIR, 'voc2012seg_patch_gt.npz'),
                allow_pickle=True)
    ps, bl = z['patch_seg'].astype(bool), z['bag_labels'].astype(bool)
    n_inst = ps.shape[1]
    keep = []
    for i in range(bl.shape[0]):
        if bl[i].sum() < 2:
            continue
        ws = [set(np.where(ps[i, :, l])[0]) for l in np.where(bl[i])[0]]
        ws = [w for w in ws if 0 < len(w) < n_inst]          # scorable pairs only
        if len(ws) >= 2 and all(not (ws[a] & ws[b])
                                for a in range(len(ws))
                                for b in range(a + 1, len(ws))):
            keep.append(i)
    keep = np.array(keep)
    ch = 100 * chance_precision_at_1(ps[keep], bl[keep], None)
    ce = 100 * label_agnostic_ceiling(ps[keep], bl[keep], None)
    check('VOC multi-label-disjoint headroom == 25.2 pts',
          abs((ce - ch) - 25.2) < 0.1,
          f'{len(keep)} bags, chance {ch:.1f} -> ceiling {ce:.1f} ({ce-ch:+.1f})')
except Exception as exc:                                     # pragma: no cover
    check('VOC multi-label-disjoint headroom == 25.2 pts', False,
          f'could not compute: {exc}')

# ---------------------------------------------------------------------------
# Setup-section claims (added 2026-09-04 after the Neurocomputing desk
# rejection for an unspecified methodology). These pin the prose of
# sections/setup.tex, architectures.tex and the Benchmarks section to the
# stored run configs and the benchmark files.
# ---------------------------------------------------------------------------
for ds in PARTITIONED:
    for m, d in ((m, d) for (m, dd), d in CELLS.items() if dd == ds):
        pass
ok = all(CELLS[(m, ds)]['num_folds'] == 5
         and len(CELLS[(m, ds)]['instance_per_model']) == 35
         and CELLS[(m, ds)]['ensemble_seeds'] == [42, 123, 456, 789, 1024, 2048, 3072]
         for (m, ds) in CELLS if ds in CLASSICAL)
check('classical cells (incl. Bird Song): 5 folds x 7 seeds = 35 models',
      ok and all((m, 'hja_birdsong') in CELLS for m in AGNOSTIC + CONDITIONAL), '')
ok = all(CELLS[(m, ds)]['num_folds'] == 3
         and len(CELLS[(m, ds)]['instance_per_model']) == 3
         and CELLS[(m, ds)]['ensemble_seeds'] is None
         for (m, ds) in CELLS if ds.startswith('voc'))
check('VOC cells: 3 folds x 1 model', ok, '')
bs = {(m, ds): CELLS[(m, ds)]['config']['batch_size']
      for (m, ds) in CELLS if ds.startswith('voc')}
exc = {k: v for k, v in bs.items() if v != 4}
check('VOC batch size 4 except AB-MIL/VOC-Patch-Seg = 8',
      exc == {('attnmil', 'voc2012seg'): 8}, exc)
check('HAP cells scored on eff_attn, baselines on attn_weights',
      all(d['used_eff_attn'] == (m == 'hap_only') for (m, ds), d in CELLS.items()),
      '')
try:
    from datasets.voc_miml import DEFAULT_OUT_DIR
    zz = np.load(os.path.join(DEFAULT_OUT_DIR, 'voc2012_patch_gt.npz'),
                 allow_pickle=True)
    check('VOC patch-label threshold == 0.05', float(zz['threshold']) == 0.05,
          float(zz['threshold']))
    z = np.load(os.path.join(DEFAULT_OUT_DIR, 'voc2012seg_patch_gt.npz'),
                allow_pickle=True)
    ps, bl = z['patch_seg'].astype(bool), z['bag_labels'].astype(bool)
    unw = int((bl & ~ps.any(1)).sum())
    check('VOC-Patch-Seg: 103 of 3,095 bag labels unwitnessed',
          unw == 103 and int(bl.sum()) == 3095, f'{unw}/{int(bl.sum())}')
    check('VOC-Patch-Seg: 2,273 bags, 27.8% multi-label',
          bl.shape[0] == 2273 and abs(100 * (bl.sum(1) > 1).mean() - 27.8) < 0.05,
          f'{bl.shape[0]} bags, {100*(bl.sum(1)>1).mean():.1f}%')
    z = np.load(os.path.join(DEFAULT_OUT_DIR, 'voc2012segml_patch_gt.npz'),
                allow_pickle=True)
    ps, bl = z['patch_seg'].astype(bool), z['bag_labels'].astype(bool)
    check('VOC-Patch-Seg-ML: 660 bags, OR-consistent, 2.29 labels/bag',
          bl.shape[0] == 660 and (ps.any(1) == bl).all()
          and abs(bl.sum(1).mean() - 2.29) < 0.005,
          f'{bl.shape[0]} bags, {bl.sum(1).mean():.2f}')
except Exception as exc:                                     # pragma: no cover
    check('VOC benchmark files readable', False, f'{exc}')
try:
    from baselines.transmil_lite import TransMILLite
    m = TransMILLite(input_dim=8, num_labels=3)
    check('TransMIL exposes real encoder attention (no attn_proj)',
          not hasattr(m, 'attn_proj'), '')
except Exception as exc:                                     # pragma: no cover
    check('TransMIL importable', False, f'{exc}')
tex = 'paper_attribution/tables/attribution_stats.tex'
if os.path.exists(tex):
    rows = [line for line in open(tex) if line.rstrip().endswith('\\\\')
            and (' $>$ ' in line or ' vs ' in line)]
    n_vs = sum(' vs ' in line for line in rows)
    check('stats appendix: 84 comparisons, 9 tied, 75 tested',
          len(rows) == 84 and n_vs == 9, f'{len(rows)} rows, {n_vs} tied')

# ---------------------------------------------------------------------------
# Claims added in the 2026-10-05 pre-submission review (DMKD build).
# ---------------------------------------------------------------------------
check('42 result cells', len(CELLS) == 42, len(CELLS))
n_total = sum(len(d['instance_per_model']) for d in CELLS.values())
check('1,022 trained models in total', n_total == 1022, n_total)

def ap(m, ds):
    return CELLS[(m, ds)]['bag_metrics_mean_std']['AveragePrecision'][0]

def same_order(ds):
    by_ap = sorted(CONDITIONAL, key=lambda m: -ap(m, ds))
    by_p1 = sorted(CONDITIONAL, key=lambda m: -im(m, ds, 'InstP@1'))
    return by_ap == by_p1

for ds in ['letter_frost', 'letter_carroll']:
    check(f'bag AP orders label-conditional methods as P@1 on {ds}',
          same_order(ds), '')
for ds in ['mscv2', 'hja_birdsong', 'voc2012seg', 'voc2012segml']:
    check(f'bag AP order differs from P@1 order on {ds}', not same_order(ds), '')

mags = []
for ds in CLASSICAL + ['voc2012seg', 'voc2012segml']:
    sa = max(ap(m, ds) for m in CONDITIONAL) - min(ap(m, ds) for m in CONDITIONAL)
    sp = (max(im(m, ds, 'InstP@1') for m in CONDITIONAL)
          - min(im(m, ds, 'InstP@1') for m in CONDITIONAL))
    mags.append(sp / sa)
check('magnification spans 5.3 to 22.6',
      abs(min(mags) - 5.3) < 0.05 and abs(max(mags) - 22.6) < 0.05,
      f'{min(mags):.1f}..{max(mags):.1f}')

d_ag = [100 * (im(m, 'hja_birdsong', 'InstP@1') - im(m, 'hja_birdsong', 'ChanceP@1'))
        for m in AGNOSTIC]
check('Bird Song: label-agnostic +13.3/+15.4/+14.4 over chance',
      [round(x, 1) for x in d_ag] == [13.3, 15.4, 14.4], [round(x, 1) for x in d_ag])
try:
    from miml_clam.metrics.instance_metrics import (
        chance_precision_at_1, label_agnostic_ceiling, witness_disjointness)
    from analysis.instance_level_stats import load_padded
    _, inst, bag, mask = load_padded('hja_birdsong')
    ch = chance_precision_at_1(inst, bag, mask)
    ce = label_agnostic_ceiling(inst, bag, mask)
    orc = all((inst[i][~mask[i]].any(0) == bag[i]).all() for i in range(len(bag)))
    check('Bird Song: 548 bags, OR-consistent, disjoint, chance 24.5 / ceiling 48.4',
          bag.shape[0] == 548 and orc
          and abs(witness_disjointness(inst, bag, mask) - 1.0) < 1e-12
          and round(100 * ch, 1) == 24.5 and round(100 * ce, 1) == 48.4,
          f'{bag.shape[0]} bags, {100*ch:.1f} / {100*ce:.1f}')
    check('Bird Song: label-agnostic models below the ceiling',
          all(im(m, 'hja_birdsong', 'InstP@1') < ce for m in AGNOSTIC), '')
except Exception as exc:                                     # pragma: no cover
    check('Bird Song data readable', False, f'{exc}')

hours = {k: 0.0 for k in ('all', 'voc', 'bird', 'tvoc')}
for (m, ds), d in CELLS.items():
    w = d['wall_clock_seconds'] / 3600
    hours['all'] += w
    if ds.startswith('voc'):
        hours['voc'] += w
        if m == 'transmil_lite':
            hours['tvoc'] += w
    if ds == 'hja_birdsong':
        hours['bird'] += w
check('compute: 27.0 h total, 20.2 VOC (16.1 TransMIL), 4.2 Bird Song',
      [round(hours[k], 1) for k in ('all', 'voc', 'tvoc', 'bird')]
      == [27.0, 20.2, 16.1, 4.2], {k: round(v, 1) for k, v in hours.items()})

st = json.load(open('paper_attribution/tables/attribution_stats.json'))['per_dataset']
sig = [c for ds in CLASSICAL for c in st[ds]['comparisons'] if c.get('significant')]
check('every significant comparison holds in 5/5 folds',
      all(c['folds_same_sign'] == 5 for c in sig), f'{len(sig)} significant')
mq = [c for ds in CLASSICAL for c in st[ds]['comparisons']
      if {c['a'], c['b']} == {'ml_decoder', 'q2l'}]
check('ML-Decoder > Q2L significant on all four classical benchmarks',
      len(mq) == 4 and all(c['a'] == 'ml_decoder' and c['significant'] for c in mq),
      [round(100 * c['mean_diff'], 1) for c in mq])

print()
if fails:
    print(f'{len(fails)} CHECK(S) FAILED')
    sys.exit(1)
print('all checks passed')

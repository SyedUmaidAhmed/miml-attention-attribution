"""Submission gate for the DMKD revision: every numeric claim of the revised
manuscript, re-derived from results/instance_level_r1 and the generated
analysis files. Exits non-zero on any failure.

Each check names the manuscript location of the claim. When a sentence in the
paper changes, its check changes in the same commit.

    python3 analysis/verify_paper_claims_r1.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R = 'results/instance_level_r1'
T = 'paper_attribution/tables'
# The manuscript source is private: checks of its wording run only where it is present.
PAPER = os.path.exists('paper_attribution/main.tex')
CELLS = {}
for f in glob.glob(os.path.join(R, '*__*.json')):
    d = json.load(open(f))
    CELLS[(d['method'], d['dataset'])] = d
EX = json.load(open(os.path.join(T, 'attribution_extras.json')))['cells']
ST = json.load(open(os.path.join(T, 'attribution_stats_r1.json')))['per_dataset']
SA = json.load(open(os.path.join(T, 'bagap_stats_r1.json')))['per_dataset']
RS = json.load(open(os.path.join(T, 'revision_summary.json')))

AG = ['attnmil', 'gated_attn', 'transmil_lite']
PART = ['letter_frost', 'letter_carroll', 'mscv2']
CLS = PART + ['hja_birdsong']
COND = ['dsmil_orig', 'dsmil', 'ml_decoder', 'q2l', 'hap_only']
MAIN = AG + COND
fails = []


def check(where, name, cond, actual=''):
    print(f"  {'PASS' if cond else 'FAIL'}  [{where:12s}] {name:62s} {actual}")
    if not cond:
        fails.append(f'{where}: {name}')


def im(m, d, k='InstP@1'):
    return CELLS[(m, d)]['instance_metrics_mean_std'][k][0]


def delta(m, d, k='InstP@1'):
    return 100 * (im(m, d, k) - im(m, d, 'ChanceP@1'))


def r1(x, n=1):
    return round(x, n)


def comp(st, d, a, b):
    for c in st[d]['comparisons']:
        if {c['a'], c['b']} == {a, b}:
            return c
    raise KeyError((d, a, b))


print('Verifying the revised manuscript against', R, '\n')

# ---- completeness -----------------------------------------------------------
n_cls = sum(len(CELLS[(m, d)]['instance_per_model']) for m in MAIN for d in CLS)
check('setup', 'classical cells complete: 8 archs x 4 x 35 models', n_cls == 8 * 4 * 35, n_cls)
for m in MAIN:
    for d in CLS:
        c = CELLS[(m, d)]
        ok = (c['num_folds'] == 5 and len(c['instance_per_model']) == 35
              and c.get('protocol_rev') == 'dmkd_r1')
        if not ok:
            check('setup', f'protocol {m}/{d}', False)

# ---- partitioned identity (analysis 9.1, intro, abstract) ---------------------
worst = max(max(abs(r['InstP@1'] - r['ChanceP@1']), abs(r['InstAUCPooled'] - 0.5))
            for m in AG for d in PART for r in CELLS[(m, d)]['instance_per_model'])
n = sum(len(CELLS[(m, d)]['instance_per_model']) for m in AG for d in PART)
check('analysis 9.1', '315 agnostic models: P@1=chance, pooled AUC=0.5 (<6e-17)',
      n == 315 and worst < 6e-17, f'n={n} worst={worst:.2e}')
check('analysis 9.1', 'agnostic chance 26.1, 25.4, 34.2',
      [r1(100 * im('attnmil', d, 'ChanceP@1')) for d in PART] == [26.1, 25.4, 34.2])
mac = [im(m, d, 'InstAUC') for m in AG for d in PART]
check('analysis 9.1', 'agnostic macro AUC within 0.498..0.506',
      round(min(mac), 3) >= 0.498 and round(max(mac), 3) <= 0.506,
      f'{min(mac):.3f}..{max(mac):.3f}')

# ---- regimes table: AUC ceiling (theory, T3) --------------------------------
reg = json.load(open(os.path.join(T, 'attribution_summary.json')))['regimes'] \
    if os.path.exists(os.path.join(T, 'attribution_summary.json')) else {}
if reg and 'auc_pool_ceiling' in next(iter(reg.values())):
    got = {k: round(v['auc_pool_ceiling'], 3) for k, v in reg.items()}
    check('T3', 'Gamma_AUC: 0.500 x3, Bird Song 0.843',
          [got.get(x) for x in ('Letter Frost', 'Letter Carroll', 'MSRC~v2', 'Bird Song')]
          == [0.5, 0.5, 0.5, 0.843], got)

# ---- additive read-out (theory 4.4, analysis, abstract) -----------------------
add = {m: [r1(delta(m, d, 'AddP@1')) for d in PART] for m in ('attnmil', 'gated_attn')}
check('analysis add.', 'AB-MIL additive +29.3 +31.9 +26.4', add['attnmil'] == [29.3, 31.9, 26.4], add['attnmil'])
check('analysis add.', 'Gated additive +33.7 +34.2 +28.6', add['gated_attn'] == [33.7, 34.2, 28.6], add['gated_attn'])
pa = [im(m, d, 'AddAUCPooled') for m in ('attnmil', 'gated_attn') for d in PART]
check('analysis add.', 'additive pooled AUC 0.747..0.785',
      [round(min(pa), 3), round(max(pa), 3)] == [0.747, 0.785], f'{min(pa):.3f}..{max(pa):.3f}')
mn = min(100 * (r['AddP@1'] - r['ChanceP@1']) for m in ('attnmil', 'gated_attn')
         for d in PART for r in CELLS[(m, d)]['instance_per_model'])
nadd = sum(len(CELLS[(m, d)]['instance_per_model']) for m in ('attnmil', 'gated_attn') for d in PART)
check('analysis add.', '210 additive models all >= +17 pts', nadd == 210 and mn >= 17, f'min {mn:.1f}')
err = max(r['AddIdentityMaxErr'] for m in ('attnmil', 'gated_attn') for d in CLS
          for r in CELLS[(m, d)]['instance_per_model'])
check('analysis add.', 'decomposition reproduces logits to <2e-5', err < 2e-5, f'{err:.2e}')
check('analysis add.', 'ML-Decoder +29.0 +33.5 +37.5',
      [r1(delta('ml_decoder', d)) for d in PART] == [29.0, 33.5, 37.5])
check('analysis add.', 'Bird Song attention +13.7/+15.0, additive +25.7/+31.8',
      [r1(delta('attnmil', 'hja_birdsong')), r1(delta('gated_attn', 'hja_birdsong')),
       r1(delta('attnmil', 'hja_birdsong', 'AddP@1')), r1(delta('gated_attn', 'hja_birdsong', 'AddP@1'))]
      == [13.7, 15.0, 25.7, 31.8])
lo = min(min(add['attnmil']), min(add['gated_attn']))
hi = max(max(add['attnmil']), max(add['gated_attn']))
check('abstract', 'additive "26 to 34" points', int(lo) == 26 and int(hi) == 34, f'{lo}..{hi}')

# ---- deletion test (analysis, response) ------------------------------------
eq = max(abs(EX[f'{m}__{d}']['del_self_present'] - EX[f'{m}__{d}']['del_other_present'])
         for m in AG for d in CLS)
check('analysis del.', 'agnostic: delta_l == delta_l\' on every classical benchmark', eq < 1e-6, f'{eq:.1e}')
ml_s = [EX[f'ml_decoder__{d}']['del_self_present'] for d in PART]
ml_o = [EX[f'ml_decoder__{d}']['del_other_present'] for d in PART]
check('analysis del.', 'ML-Dec delta_l 1.50..2.05, delta_l\' 0.13..0.47 (partitioned)',
      [round(min(ml_s), 2), round(max(ml_s), 2), round(min(ml_o), 2), round(max(ml_o), 2)]
      == [1.50, 2.05, 0.13, 0.47], (ml_s, ml_o))
e = EX['dsmil__mscv2']
check('analysis del.', 'DSMIL-sep MSRC: other 0.94 > self 0.84',
      [round(e['del_other_present'], 2), round(e['del_self_present'], 2)] == [0.94, 0.84])
bad = []
for m in MAIN:
    for d in CLS:
        s, rnd = EX[f'{m}__{d}']['del_self_present'], im(m, d, 'DelRndSelf')
        if not s > rnd:
            bad.append((m, d))
        elif not (s >= 2 * rnd or rnd <= 0) and (m, d) != ('dsmil', 'mscv2'):
            bad.append((m, d, 'factor<2'))
check('analysis del.', 'delta_l > delta_rnd everywhere; >=2x except DSMIL-sep MSRC', not bad, bad)

# ---- DSMIL pair (analysis, abstract, conclusion) --------------------------------
check('analysis nec.', 'DSMIL +12.8 +16.6 +23.4 +26.0',
      [r1(delta('dsmil_orig', d)) for d in CLS] == [12.8, 16.6, 23.4, 26.0])
R_sep = CELLS[('dsmil', 'mscv2')]['instance_per_model']
check('analysis nec.', 'DSMIL-sep MSRC P@1 30.1, chance 34.2, pooled 0.460, 34/35 AUC<0.5',
      [r1(100 * im('dsmil', 'mscv2')), r1(100 * im('dsmil', 'mscv2', 'ChanceP@1')),
       round(im('dsmil', 'mscv2', 'InstAUCPooled'), 3), sum(r['InstAUC'] < 0.5 for r in R_sep)]
      == [30.1, 34.2, 0.460, 34])
c = comp(ST, 'mscv2', 'attnmil', 'dsmil')
check('analysis nec.', 'AB-MIL > DSMIL-sep MSRC: 4.0 in the mean, NOT significant two-sided (p 0.091), delta 0.74, 5/5',
      c['a'] == 'attnmil' and [r1(100 * c['mean_diff']), round(c['p_fold_nb_holm'], 3),
                               round(c['cliffs_delta'], 2), c['folds_same_sign']] == [4.0, 0.091, 0.74, 5]
      and not c['significant_fold'])
c = comp(ST, 'mscv2', 'dsmil_orig', 'dsmil')
check('analysis nec.', 'DSMIL vs DSMIL-sep MSRC: 27.4, p 6.9e-4, delta 0.998',
      c['a'] == 'dsmil_orig' and r1(100 * c['mean_diff']) == 27.4
      and float(f"{c['p_fold_nb_holm']:.1e}") == 6.9e-4 and round(c['cliffs_delta'], 3) == 0.998)
ap_o, ap_s = im('dsmil_orig', 'mscv2', 'BagAveragePrecision'), im('dsmil', 'mscv2', 'BagAveragePrecision')
check('analysis nec.', 'DSMIL pair: 3.1 pts AP (0.773 vs 0.743)',
      [round(ap_o, 3), round(ap_s, 3), r1(100 * (ap_o - ap_s))] == [0.773, 0.743, 3.1])

# ---- label-query designs ----------------------------------------------------
mla = [im('ml_decoder', d, 'InstAUCPooled') for d in CLS]
qa = [im('q2l', d, 'InstAUCPooled') for d in CLS]
check('analysis nec.', 'pooled AUC ML-Dec 0.735..0.808, Q2L 0.678..0.710',
      [round(min(mla), 3), round(max(mla), 3), round(min(qa), 3), round(max(qa), 3)]
      == [0.735, 0.808, 0.678, 0.710])
cs = [comp(ST, d, 'ml_decoder', 'q2l') for d in CLS]
check('analysis nec.', 'ML-Dec > Q2L +5.4 +9.0 +12.4 (p<=0.018) and +12.4 BS (p 0.29, 5/5)',
      all(c['a'] == 'ml_decoder' for c in cs)
      and [r1(100 * c['mean_diff']) for c in cs] == [5.4, 9.0, 12.4, 12.4]
      and all(c['p_fold_nb_holm'] <= 0.018 for c in cs[:3])
      and round(cs[3]['p_fold_nb_holm'], 2) == 0.29 and cs[3]['folds_same_sign'] == 5)
check('analysis nec.', 'ML-Dec > Q2L delta 0.57..0.94 on partitioned',
      [round(c['cliffs_delta'], 2) for c in cs[:3]] == [0.57, 0.89, 0.94])
check('analysis nec.', 'ML-Dec vs TransMIL Cliff delta = 1.00 on all four',
      all(comp(ST, d, 'ml_decoder', 'transmil_lite')['cliffs_delta'] == 1.0 for d in CLS))

# ---- bag AP vs attribution ------------------------------------------------------
def nsig(st, d):
    return sum(c['significant_fold'] for c in st[d]['comparisons']
               if c['a'] in COND and c['b'] in COND)
na, npp = sum(nsig(SA, d) for d in CLS), sum(nsig(ST, d) for d in CLS)
check('analysis bag', 'fold-sig label-conditional pairs: attribution 26/40, AP 5/40',
      (npp, na) == (26, 5), (npp, na))
opp = 0
for d in CLS:
    for ca in SA[d]['comparisons']:
        if ca['a'] in COND and ca['b'] in COND and ca['significant_fold']:
            cp = comp(ST, d, ca['a'], ca['b'])
            if cp['significant_fold'] and cp['a'] != ca['a']:
                opp += 1
check('analysis bag', 'no pair separated by both in opposite directions', opp == 0, opp)
span_ap = [r1(100 * (max(im(m, d, 'BagAveragePrecision') for m in COND)
                     - min(im(m, d, 'BagAveragePrecision') for m in COND))) for d in CLS]
span_p = [r1(100 * (max(im(m, d) for m in COND) - min(im(m, d) for m in COND))) for d in CLS]
check('analysis bag', 'AP spans 6.3 7.4 6.3 2.2; P@1 spans 27.1 30.2 41.6 23.9',
      span_ap == [6.3, 7.4, 6.3, 2.2] and span_p == [27.1, 30.2, 41.6, 23.9], (span_ap, span_p))
bs = sorted(round(im(m, 'hja_birdsong', 'BagAveragePrecision'), 3) for m in COND)
check('analysis bag', 'Bird Song AP 0.931..0.936 for four of five', bs[1:] == sorted(bs[1:])
      and bs[1] == 0.931 and bs[-1] == 0.936, bs)
for d in ('mscv2', 'hja_birdsong'):
    order = sorted(COND, key=lambda m: -im(m, d, 'BagAveragePrecision'))
    check('analysis bag', f'{d}: Q2L top AP, ML-Dec second, ML-Dec +12.4 attribution',
          order[:2] == ['q2l', 'ml_decoder'] and r1(100 * (im('ml_decoder', d) - im('q2l', d))) == 12.4,
          order[:2])
check('analysis bag', 'MSRC: HAP AP 0.784 > DSMIL 0.773, attribution 15.3 lower',
      [round(im('hap_only', 'mscv2', 'BagAveragePrecision'), 3),
       round(im('dsmil_orig', 'mscv2', 'BagAveragePrecision'), 3),
       r1(100 * (im('dsmil_orig', 'mscv2') - im('hap_only', 'mscv2')))] == [0.784, 0.773, 15.3])
W = {k: v for k, v in RS['within_architecture_spearman_ap_p1'].items()
     if k.split('__')[0] in MAIN and k.split('__')[1] in CLS}
rh = [v['spearman_rho_fold_centred'] for v in W.values()]
check('analysis bag', 'within-arch: 23 cells, 9 sig, rho -0.49..+0.88',
      len(W) == 23 and sum(v['p'] < 0.05 for v in W.values()) == 9
      and [round(min(rh), 2), round(max(rh), 2)] == [-0.49, 0.88])
check('analysis bag', 'DSMIL-sep MSRC within-arch rho -0.49',
      round(W['dsmil__mscv2']['spearman_rho_fold_centred'], 2) == -0.49)

# ---- Bird Song decomposition ---------------------------------------------------
from miml_clam.metrics.instance_metrics import label_agnostic_ceiling, chance_precision_at_1
from analysis.instance_level_stats import load_padded
_, _i, _b, _m = load_padded('hja_birdsong')
check('analysis 9.1', 'Bird Song chance 24.5, dataset ceiling 48.4 (T3)',
      [r1(100 * chance_precision_at_1(_i, _b, _m)), r1(100 * label_agnostic_ceiling(_i, _b, _m))]
      == [24.5, 48.4])
check('analysis 9.1', 'Bird Song agnostic +13.7 +15.0 +14.6, pooled 0.638..0.672',
      [r1(delta(m, 'hja_birdsong')) for m in AG] == [13.7, 15.0, 14.6]
      and [round(min(im(m, 'hja_birdsong', 'InstAUCPooled') for m in AG), 3),
           round(max(im(m, 'hja_birdsong', 'InstAUCPooled') for m in AG), 3)] == [0.638, 0.672])
check('analysis 9.1', 'Bird Song agnostic models never above ceiling',
      all(r['InstP@1'] <= r['LabelAgnosticCeiling'] + 1e-12 for m in AG
          for r in CELLS[(m, 'hja_birdsong')]['instance_per_model']))
bg = [100 * EX[f'{m}__hja_birdsong']['frac_miss_background'] for m in AG]
check('analysis 9.1', 'Bird Song agnostic misses on silence 28..29%',
      round(min(bg)) == 28 and round(max(bg)) == 29, [round(x, 1) for x in bg])
mlp = [100 * EX[f'{m}__hja_birdsong']['ml_P@1'] for m in AG]
e = EX['attnmil__hja_birdsong']
n_bags = CELLS[('attnmil', 'hja_birdsong')]
check('analysis 9.1', 'multi-species: agnostic 31.0..32.5, chance 18.1, ceiling 38.5',
      [r1(min(mlp)), r1(max(mlp)), r1(100 * e['ml_chance']), r1(100 * e['ml_ceiling'])]
      == [31.0, 32.5, 18.1, 38.5])
# 5 test folds of floor(548/5) = 109 recordings: 545 are ever tested
check('analysis 9.1', '64% of tested recordings have >=2 scorable species',
      round(100 * e['ml_bags'] * 5 / (5 * (548 // 5))) == 64, f"{100 * e['ml_bags'] / (548 // 5):.1f}")
check('analysis 9.1', 'multi-species ML-Dec 63.4, DSMIL 45.0',
      [r1(100 * EX['ml_decoder__hja_birdsong']['ml_P@1']),
       r1(100 * EX['dsmil_orig__hja_birdsong']['ml_P@1'])] == [63.4, 45.0])

# ---- HAP ablation --------------------------------------------------------------
check('analysis HAP', 'soft-only +4.6 +5.4 +11.6; HAP +1.9 +3.3 +8.1',
      [r1(delta('hap_soft_only', d)) for d in PART] == [4.6, 5.4, 11.6]
      and [r1(delta('hap_only', d)) for d in PART] == [1.9, 3.3, 8.1])
hw = max(abs(r['InstP@1'] - r['ChanceP@1']) for d in PART
         for r in CELLS[('hap_hard_only', d)]['instance_per_model'])
check('analysis HAP', 'top-k only: all 105 models within 0.8 pts of chance', 100 * hw < 0.8, f'{100 * hw:.2f}')
check('analysis HAP', 'k=5 > k=3 > k=1 on every partitioned benchmark',
      all(im('hap_k5', d) > im('hap_only', d) > im('hap_k1', d) for d in PART))
al = np.array([a for m in ('hap_only', 'hap_k1', 'hap_k5') for d in PART
               for r in CELLS[(m, d)]['instance_per_model'] for a in r['HapAlpha']])
check('analysis HAP', 'alpha 0.51 +- 0.03, range 0.42..0.65, n=7875',
      [len(al), round(al.mean(), 2), round(al.std(), 2), round(al.min(), 2), round(al.max(), 2)]
      == [7875, 0.51, 0.03, 0.42, 0.65])
check('analysis HAP', 'Bird Song: soft +18.3, HAP +16.5, top-k +8.9 (lowest), alpha 0.53',
      [r1(delta(m, 'hja_birdsong')) for m in ('hap_soft_only', 'hap_only', 'hap_hard_only')]
      == [18.3, 16.5, 8.9]
      and min(delta(m, 'hja_birdsong') for m in ('hap_only', 'hap_soft_only', 'hap_hard_only', 'hap_k1', 'hap_k5'))
      == delta('hap_hard_only', 'hja_birdsong')
      and round(RS['hap_alpha']['hap_only__hja_birdsong']['mean'], 2) == 0.53)
check('analysis HAP', 'AP HAP vs soft: 0.705/0.708, 0.784/0.785, 0.932/0.935',
      [round(im(m, d, 'BagAveragePrecision'), 3) for d in ('letter_frost', 'mscv2', 'hja_birdsong')
       for m in ('hap_only', 'hap_soft_only')] == [0.705, 0.708, 0.784, 0.785, 0.932, 0.935])

# ---- per-label (R2.3) ----------------------------------------------------------
pl = [(EX[f'{m}__{d}']['per_label_spearman_margin_vs_npairs'], EX[f'{m}__{d}']['per_label_spearman_p'])
      for m in COND for d in PART]
check('analysis lab.', 'rho -0.06..+0.81; 10/15 significant, all positive',
      [round(min(r for r, _ in pl), 2), round(max(r for r, _ in pl), 2),
       sum(p < 0.05 for _, p in pl), all(r > 0 for r, p in pl if p < 0.05)] == [-0.06, 0.81, 10, True])
check('analysis lab.', 'Bird Song: no label-conditional cell significant',
      all(EX[f'{m}__hja_birdsong']['per_label_spearman_p'] >= 0.05 for m in COND))

# ---- statistics / ensemble ----------------------------------------------------
pm = sum(c['significant'] for d in CLS for c in ST[d]['comparisons'])
pf = sum(c['significant_fold'] for d in CLS for c in ST[d]['comparisons'])
check('response R3.3', '90 per-model significant, 63 fold-level (classical)', (pm, pf) == (90, 63), (pm, pf))
G = {k: 100 * v for k, v in RS['ensemble_minus_model_ap'].items()
     if k.split('__')[0] in MAIN and k.split('__')[1] in CLS}
check('appendix', 'ensemble lift positive in all 32 classical cells, 1.2..7.9',
      len(G) == 32 and min(G.values()) > 0
      and [r1(min(G.values())), r1(max(G.values()))] == [1.2, 7.9])
check('appendix', 'AB-MIL lift up to 7.9; ML-Decoder at most 3.3',
      r1(max(v for k, v in G.items() if k.startswith('attnmil'))) == 7.9
      and r1(max(v for k, v in G.items() if k.startswith('ml_decoder'))) == 3.3)

# ---- VOC-Patch grid sensitivity (analysis voc-sensitivity) -------------------
if all((m, d) in CELLS for m in ('attnmil', 'transmil_lite', 'ml_decoder', 'q2l', 'dsmil_orig')
       for d in ('voc2012segml_nc', 'voc2012segml_g8')):
    check('analysis voc', 'grid: 8 archs x 10 folds x 1 model (as stated)',
          all(CELLS[(m, 'voc2012segml_g8')]['num_folds'] == 10 and len(CELLS[(m, 'voc2012segml_g8')]['instance_per_model']) == 10
              for m in MAIN))
    check('analysis voc', 'grid: chance 34.0 -> 25.6; fold ceiling 84.0 / 82.8',
          [r1(100 * im('attnmil', d, k)) for d in ('voc2012segml_nc', 'voc2012segml_g8')
           for k in ('ChanceP@1', 'LabelAgnosticCeiling')] == [34.0, 84.0, 25.6, 82.8])
    g = {m: [r1(delta(m, d)) for d in ('voc2012segml_nc', 'voc2012segml_g8')]
         for m in ('ml_decoder', 'q2l', 'dsmil_orig', 'attnmil', 'transmil_lite')}
    check('analysis voc', 'grid margins (ML-Dec, Q2L, DSMIL, AB-MIL, TransMIL)',
          g == {'ml_decoder': [45.3, 44.2], 'q2l': [44.5, 43.0], 'dsmil_orig': [35.0, 33.8],
                'attnmil': [17.2, 13.4], 'transmil_lite': [24.1, 15.6]}, g)
    allg = [m for m in MAIN if (m, 'voc2012segml_g8') in CELLS and (m, 'voc2012segml_nc') in CELLS]
    cond_min = {d: min(delta(m, d) for m in ('dsmil_orig', 'ml_decoder', 'q2l')) for d in ('voc2012segml_nc', 'voc2012segml_g8')}
    ag_max = {d: max(delta(m, d) for m in AG) for d in ('voc2012segml_nc', 'voc2012segml_g8')}
    check('analysis voc', 'grid: class gap widens (min cond - max agnostic)',
          cond_min['voc2012segml_g8'] - ag_max['voc2012segml_g8'] > cond_min['voc2012segml_nc'] - ag_max['voc2012segml_nc'])

# ---- VOC-Patch-Seg-ML results (analysis) ----------------------------------------
V = 'voc2012segml_nc'
if all((m, V) in CELLS for m in MAIN):
    check('analysis voc', 'Seg-ML complete: 8 archs x 30 models, 10 folds',
          all(len(CELLS[(m, V)]['instance_per_model']) == 30 and CELLS[(m, V)]['num_folds'] == 10 for m in MAIN))
    check('analysis voc', 'Seg-ML agnostic +17.2 +18.1 +24.1; ML-Dec +45.3, Q2L +44.5',
          [r1(delta(m, V)) for m in AG + ['ml_decoder', 'q2l']] == [17.2, 18.1, 24.1, 45.3, 44.5])
    oth = [100 * (1 - EX[f'{m}__{V}']['frac_miss_background']) for m in AG]
    check('analysis voc', 'Seg-ML agnostic misses on other objects 74..81%',
          [round(min(oth)), round(max(oth))] == [74, 81], [round(x, 1) for x in oth])
    check('analysis voc', 'Seg-ML additive +41.9 / +41.5; DSMIL +35.0',
          [r1(delta('attnmil', V, 'AddP@1')), r1(delta('gated_attn', V, 'AddP@1')), r1(delta('dsmil_orig', V))]
          == [41.9, 41.5, 35.0])
    e1, e2, e3 = EX[f'ml_decoder__{V}'], EX[f'dsmil_orig__{V}'], EX[f'dsmil__{V}']
    check('analysis del.', 'Seg-ML deletion: ML-Dec 0.72/-0.01, DSMIL 0.56/0.05, DSMIL-sep 0.13/0.13',
          [round(x, 2) for x in (e1['del_self_present'], e1['del_other_present'], e2['del_self_present'],
                                 e2['del_other_present'], e3['del_self_present'], e3['del_other_present'])]
          == [0.72, -0.01, 0.56, 0.05, 0.13, 0.13])
    check('analysis del.', 'Seg-ML agnostic delta_l == delta_l\'',
          max(abs(EX[f'{m}__{V}']['del_self_present'] - EX[f'{m}__{V}']['del_other_present']) for m in AG) < 1e-6)
    c = comp(ST, V, 'dsmil_orig', 'dsmil')
    check('analysis nec.', 'Seg-ML DSMIL vs sep: 32.2, p 2.4e-8, delta 1.00, 10/10 folds',
          c['a'] == 'dsmil_orig' and r1(100 * c['mean_diff']) == 32.2
          and float(f"{c['p_fold_nb_holm']:.1e}") == 2.4e-8 and c['cliffs_delta'] == 1.0 and c['folds_same_sign'] == 10)
    apv = sorted(MAIN, key=lambda m: -im(m, V, 'BagAveragePrecision'))
    check('analysis nec.', 'Seg-ML: DSMIL-sep +2.8, pooled 0.486, AP 0.892 second, +0.5 over DSMIL',
          [r1(delta('dsmil', V)), round(im('dsmil', V, 'InstAUCPooled'), 3),
           round(im('dsmil', V, 'BagAveragePrecision'), 3), apv[1],
           r1(100 * (im('dsmil', V, 'BagAveragePrecision') - im('dsmil_orig', V, 'BagAveragePrecision')))]
          == [2.8, 0.486, 0.892, 'dsmil', 0.5])
    ds5 = CLS + [V]
    check('analysis voc', 'Seg-ML chance 34.0, dataset ceiling 84.0 (T3)',
          reg.get('VOC-Patch-Seg-ML', {}).get('chance') is None or
          [r1(100 * reg['VOC-Patch-Seg-ML']['chance']), r1(100 * reg['VOC-Patch-Seg-ML']['ceiling'])] == [34.0, 84.0])

# ---- VOC-Patch-Seg results (analysis; Mac CPU, 10 folds x 3 seeds) ---------------
S = 'voc2012seg_nc'
if all((m, S) in CELLS for m in MAIN) and all((m, V) in CELLS for m in MAIN):
    check('analysis voc', 'Seg complete: 8 archs x 30 models, 10 folds',
          all(len(CELLS[(m, S)]['instance_per_model']) == 30 and CELLS[(m, S)]['num_folds'] == 10 for m in MAIN))
    check('analysis voc', 'Seg chance 40.4, ceiling 93.2 (fold means and T3)',
          [r1(100 * im('attnmil', S, 'ChanceP@1')), r1(100 * im('attnmil', S, 'LabelAgnosticCeiling'))] == [40.4, 93.2]
          and (reg.get('VOC-Patch-Seg', {}).get('chance') is None or
               [r1(100 * reg['VOC-Patch-Seg']['chance']), r1(100 * reg['VOC-Patch-Seg']['ceiling'])] == [40.4, 93.2]))
    g = [r1(delta(m, S)) for m in AG + ['ml_decoder', 'q2l']]
    check('analysis voc', 'Seg agnostic +32.6 +30.4 +35.3; ML-Dec +47.9, Q2L +46.6', g == [32.6, 30.4, 35.3, 47.9, 46.6], g)
    e = {m: EX[f'{m}__{S}'] for m in MAIN}
    ag_ml = [r1(100 * e[m]['ml_P@1']) for m in AG]
    check('analysis voc', 'Seg multi-label bags: agnostic 55.1..59.5, chance 33.5, ceil 84.7, ML-Dec 81.0, Q2L 79.2',
          [min(ag_ml), max(ag_ml), r1(100 * e['attnmil']['ml_chance']), r1(100 * e['attnmil']['ml_ceiling']),
           r1(100 * e['ml_decoder']['ml_P@1']), r1(100 * e['q2l']['ml_P@1'])] == [55.1, 59.5, 33.5, 84.7, 81.0, 79.2], ag_ml)
    # tau sweep (tab:tau)
    TAU = ['0.05', '0.25', '0.5']
    t0 = e['attnmil']
    check('analysis voc', 'tau: pairs 294/255/204, chance 40.4/31.6/25.6, ceiling 93.2/89.5/87.2',
          [round(t0[f'tau{t}_pairs']) for t in TAU] == [294, 255, 204]
          and [r1(100 * t0[f'tau{t}_chance']) for t in TAU] == [40.4, 31.6, 25.6]
          and [r1(100 * t0[f'tau{t}_ceiling']) for t in TAU] == [93.2, 89.5, 87.2])
    tm = {m: [100 * (e[m][f'tau{t}_P@1'] - e[m][f'tau{t}_chance']) for t in TAU] for m in MAIN}
    check('analysis voc', 'tau: AB-MIL +32.6 -> +14.2, ML-Dec +47.9 -> +35.2; every margin shrinks',
          [r1(tm['attnmil'][0]), r1(tm['attnmil'][2]), r1(tm['ml_decoder'][0]), r1(tm['ml_decoder'][2])] == [32.6, 14.2, 47.9, 35.2]
          and all(tm[m][0] > tm[m][1] > tm[m][2] for m in MAIN))
    check('analysis voc', 'tau: agnostic margins shrink most (AB-MIL, gated > every label-query drop)',
          min(tm[m][0] - tm[m][2] for m in ('attnmil', 'gated_attn')) > max(tm[m][0] - tm[m][2] for m in ('ml_decoder', 'q2l')))
    from itertools import combinations as _comb
    sw = [(a, b) for a, b in _comb(MAIN, 2)
          if len({np.sign(e[a][f'tau{t}_P@1'] - e[b][f'tau{t}_P@1']) for t in TAU}) > 1]
    check('analysis voc', 'tau: only DSMIL/HAP change places; within 1 pt; not sig at 0.05',
          sw == [('dsmil_orig', 'hap_only')]
          and max(abs(100 * (e['dsmil_orig'][f'tau{t}_P@1'] - e['hap_only'][f'tau{t}_P@1'])) for t in TAU) <= 1.05
          and not comp(ST, S, 'dsmil_orig', 'hap_only')['significant_fold'], sw)
    check('analysis voc', 'tau: ML-Dec, Q2L top two and DSMIL-sep last at every tau',
          all(set(sorted(MAIN, key=lambda m: -tm[m][i])[:2]) == {'ml_decoder', 'q2l'}
              and sorted(MAIN, key=lambda m: -tm[m][i])[-1] == 'dsmil' for i in range(3)))
    # additive (subsec:additive)
    ad = [r1(delta('attnmil', S, 'AddP@1')), r1(delta('gated_attn', S, 'AddP@1'))]
    check('analysis add.', 'Seg additive +49.3 / +49.2, above every label-conditional attention',
          ad == [49.3, 49.2] and min(delta(m, S, 'AddP@1') for m in ('attnmil', 'gated_attn'))
          > max(delta(m, S) for m in COND), ad)
    # deletion (subsec:deletion)
    dl = [round(e[m][k], 2) for m in ('ml_decoder', 'dsmil_orig', 'dsmil') for k in ('del_self_present', 'del_other_present')]
    check('analysis del.', 'Seg deletion: ML-Dec 0.93/0.01, DSMIL 0.87/0.03, DSMIL-sep 0.11 self < 0.19 other',
          dl == [0.93, 0.01, 0.87, 0.03, 0.11, 0.19], dl)
    check('analysis del.', 'Seg agnostic delta_l == delta_l\'',
          max(abs(e[m]['del_self_present'] - e[m]['del_other_present']) for m in AG) < 1e-6)
    bad = []
    for m in MAIN:
        for d in CLS + [V, S]:
            sv, rnd = EX[f'{m}__{d}']['del_self_present'], im(m, d, 'DelRndSelf')
            if not sv > rnd:
                bad.append((m, d))
            elif not (sv >= 2 * rnd or rnd <= 0):
                bad.append((m, d, 'factor<2'))
    rat = [EX[f'dsmil__{d}']['del_self_present'] / im('dsmil', d, 'DelRndSelf') for d in ('mscv2', V, S)]
    check('analysis del.', 'delta_l > delta_rnd in all 48 cells; >=2x except DSMIL-sep MSRC, Seg-ML, Seg (1.2..1.9)',
          sorted(bad) == sorted([('dsmil', 'mscv2', 'factor<2'), ('dsmil', V, 'factor<2'), ('dsmil', S, 'factor<2')])
          and [r1(min(rat)), r1(max(rat))] == [1.2, 1.9], (bad, rat))
    # DSMIL pair (subsec:necessary)
    c = comp(ST, S, 'dsmil_orig', 'dsmil')
    check('analysis nec.', 'Seg DSMIL vs sep: 28.8, p 5.0e-8, delta 1.00, 10/10 folds',
          c['a'] == 'dsmil_orig' and r1(100 * c['mean_diff']) == 28.8
          and float(f"{c['p_fold_nb_holm']:.1e}") == 5.0e-8 and c['cliffs_delta'] == 1.0 and c['folds_same_sign'] == 10)
    aps = sorted(MAIN, key=lambda m: -im(m, S, 'BagAveragePrecision'))
    check('analysis nec.', 'Seg DSMIL-sep +12.1 weakest; AP 0.926 highest, = ML-Dec to 3 dp',
          r1(delta('dsmil', S)) == 12.1 and min(MAIN, key=lambda m: delta(m, S)) == 'dsmil'
          and aps[0] == 'dsmil' and round(im('dsmil', S, 'BagAveragePrecision'), 3) == 0.926
          == round(im('ml_decoder', S, 'BagAveragePrecision'), 3))
    # bag AP vs attribution over six benchmarks (subsec:bagap)
    ds6 = CLS + [V, S]
    check('analysis bag', 'six benchmarks: attribution 43/60, AP 6/60, AP 0 on Seg-ML',
          (sum(nsig(ST, d) for d in ds6), sum(nsig(SA, d) for d in ds6), nsig(SA, V)) == (43, 6, 0),
          (sum(nsig(ST, d) for d in ds6), sum(nsig(SA, d) for d in ds6), nsig(SA, V)))
    opp6 = []
    for d in ds6:
        for ca in SA[d]['comparisons']:
            if ca['a'] in COND and ca['b'] in COND and ca['significant_fold']:
                cp = comp(ST, d, ca['a'], ca['b'])
                if cp['significant_fold'] and cp['a'] != ca['a']:
                    opp6.append((d, ca['a'], ca['b'], ca, cp))
    ok = len(opp6) == 1 and opp6[0][:3] == (S, 'dsmil', 'q2l')
    if ok:
        ca, cp = opp6[0][3], opp6[0][4]
        ok = (r1(100 * ca['mean_diff']) == 1.0 and round(ca['p_fold_nb_holm'], 3) == 0.034
              and r1(100 * cp['mean_diff']) == 34.5 and float(f"{cp['p_fold_nb_holm']:.1e}") == 1.1e-8)
    check('analysis bag', 'exactly one opposite pair: Seg DSMIL-sep vs Q2L, AP +1.0 p .034 / P@1 34.5 p 1.1e-8',
          ok, [o[:3] for o in opp6])
    spv = [r1(100 * (max(im(m, d, k) for m in COND) - min(im(m, d, k) for m in COND)))
           for d in (V, S) for k in ('BagAveragePrecision', 'InstP@1')]
    check('analysis bag', 'VOC spans: Seg-ML AP 2.4 / P@1 42.5; Seg AP 1.0 / P@1 35.9', spv == [2.4, 42.5, 1.0, 35.9], spv)
    # model totals (setup) and Figure 1 caption
    nmain = sum(len(CELLS[(m, d)]['instance_per_model']) for m in MAIN for d in CLS + [V, S])
    ncls = sum(len(CELLS[(m, d)]['instance_per_model']) for m in MAIN for d in CLS)
    nall = sum(len(c['instance_per_model']) for c in CELLS.values())
    check('setup', 'models: 1,120 classical, 480 VOC, 1,600 main, 2,480 total',
          (ncls, nmain - ncls, nmain, nall) == (1120, 480, 1600, 2480), (ncls, nmain - ncls, nmain, nall))
    if PAPER:
        cap = open('paper_attribution/figures/overview_caption.tex').read()
        check('figure 1', 'caption model count = 1,600', '$1{,}600$' in cap and '1{,}022' not in cap)
    devs = sorted({(m, d) for (m, d), c in CELLS.items() if not str(c.get('device', '')).startswith('cpu')})
    check('setup', 'one platform: every r1 cell trained on CPU', not devs, devs)

# ---- compute and cross-platform check (setup subsec:compute) --------------------
_w = {}
for (m, d), c in CELLS.items():
    _w[d] = _w.get(d, 0) + c['wall_clock_seconds']
_cl = sum(_w[d] for d in CLS) / 3600
_tot = sum(_w.values()) / 3600
check('setup', 'compute: 48.4 h summed (16.0 classical, 32.4 VOC)',
      [r1(_tot), r1(_cl), r1(_tot - _cl)] == [48.4, 16.0, 32.4], [_tot, _cl])
_pm = [c['wall_clock_seconds'] / len(c['instance_per_model']) / 60 for (m, d), c in CELLS.items() if d.startswith('voc')]
check('setup', 'VOC per-model 0.7..9.1 min', [r1(min(_pm)), r1(max(_pm))] == [0.7, 9.1], [min(_pm), max(_pm)])
_cx = {}
for f in glob.glob('results/instance_level_r1_colabA/*__voc2012seg_nc.json'):
    a = json.load(open(f)); m = a['method']
    _cx[m] = (100 * (a['instance_metrics_mean_std']['InstP@1'][0] - im(m, 'voc2012seg_nc')),
              a['instance_metrics_mean_std']['BagAveragePrecision'][0] - im(m, 'voc2012seg_nc', 'BagAveragePrecision'),
              str(a.get('device', '')))
_st = [m for m in _cx if m != 'dsmil']
_sep_cu = im('dsmil', 'voc2012seg_nc') + _cx['dsmil'][0] / 100 if 'dsmil' in _cx else None
check('setup', 'CUDA check: 6 cells; 5 within 1.6 P@1 / 0.001 AP; DSMIL-sep 5.1, weakest on both',
      len(_cx) == 6 and 'hap_only' in _cx and all(v[2].startswith('cuda') for v in _cx.values())
      and max(abs(_cx[m][0]) for m in _st) < 1.6 and max(abs(_cx[m][1]) for m in _st) < 0.001
      and r1(abs(_cx['dsmil'][0])) == 5.1
      and all(_sep_cu < im(m, 'voc2012seg_nc') + _cx[m][0] / 100 for m in _st)
      and min(_st + ['dsmil'], key=lambda m: im(m, 'voc2012seg_nc')) == 'dsmil', {k: v[:2] for k, v in _cx.items()})

import re as _re
# ---- defects found by the explainer audit (2026-11-25), pinned so they stay fixed --
_pool = lambda m: [x['InstAUCPooled'] for x in CELLS[(m, 'mscv2')]['instance_per_model']]
check('analysis nec.', 'DSMIL-sep MSRC: pooled AUC 0.460, 31/35 models below 0.5; DSMIL 0.677, 0/35',
      [round(np.mean(_pool('dsmil')), 3), sum(v < 0.5 for v in _pool('dsmil')),
       round(np.mean(_pool('dsmil_orig')), 3), sum(v < 0.5 for v in _pool('dsmil_orig'))] == [0.46, 31, 0.677, 0])
_gap = max(100 * (max(im(m, d, 'BagAveragePrecision') for m in AG) - min(im(m, d, 'BagAveragePrecision') for m in AG))
           for d in PART)
check('analysis 9.1', 'label-agnostic bag AP differs by up to 3.5 pts on partitioned data', r1(_gap) == 3.5, _gap)
DS6 = CLS + ['voc2012seg_nc', 'voc2012segml_nc']
_span = lambda d, k: 100 * (max(im(m, d, k) for m in COND) - min(im(m, d, k) for m in COND))
_ma, _mi = [_span(d, 'BagMacroF1') for d in DS6], [_span(d, 'BagMicroF1') for d in DS6]
check('response R2.2', 'per-model F1 spans: Macro 2.6..8.9, Micro 2.3..4.9',
      [r1(min(_ma)), r1(max(_ma)), r1(min(_mi)), r1(max(_mi))] == [2.6, 8.9, 2.3, 4.9], (_ma, _mi))
_top = sum(max(COND, key=lambda m: im(m, d, 'BagAveragePrecision')) != max(COND, key=lambda m: im(m, d))
           for d in DS6)
check('analysis 9.8', 'top bag-AP method is not the top attributor on 4 of 6 benchmarks', _top == 4, _top)
_n = {'letter_frost': (144, 5), 'letter_carroll': (166, 5), 'mscv2': (591, 5), 'hja_birdsong': (548, 5),
      'voc2012seg_nc': (2273, 10), 'voc2012segml_nc': (660, 10)}
_ds = {r['name']: (r['bags'], r['folds']) for r in json.load(open(os.path.join(T, 'dataset_stats.json')))}
_nb = [_ds.get(k) for k in ('Letter Frost', 'Letter Carroll', 'MSRC~v2', 'Bird Song', 'VOC-Patch-Seg', 'VOC-Patch-Seg-ML')]
check('setup 7.1', 'never-tested remainder n mod k: 4,1,1,3,3,0 (bag counts and folds from dataset_stats)',
      _nb == list(_n.values()) and [n % k for n, k in _n.values()] == [4, 1, 1, 3, 3, 0]
      and all(len(CELLS[('attnmil', d)]['instance_per_model']) == k * (7 if k == 5 else 3) for d, (n, k) in _n.items()), _nb)
_figs = [] if not PAPER else ['paper_attribution/figures/ap_vs_attribution.pdf', 'paper_attribution/figures/ceiling_vs_achieved.pdf',
         'paper_attribution/figures/overview.pdf']
_newest = max(os.path.getmtime(f) for f in glob.glob(os.path.join(R, '*__*.json')))
if PAPER:
  check('figures', 'every data figure is newer than every result file',
      all(os.path.getmtime(f) >= _newest for f in _figs), [os.path.basename(f) for f in _figs if os.path.getmtime(f) < _newest])
try:
    from datasets.voc_miml import DEFAULT_OUT_DIR as _D
    from miml_clam.metrics.instance_metrics import (label_agnostic_auc_bounds, chance_precision_at_1,
                                                    label_agnostic_ceiling)
    _att = []
    for stem in ('voc2012seg_nc', 'voc2012segml_nc'):
        z = np.load(os.path.join(_D, f'{stem}_patch_gt.npz'), allow_pickle=True)
        ps, bl = z['patch_seg'].astype(bool), z['bag_labels'].astype(bool)
        lo, hi = label_agnostic_auc_bounds(ps, bl)
        num = den = 0.0                          # pooled AUC of the coverage ranking
        for i in range(bl.shape[0]):
            n_i = ps.shape[1]
            ls = [l for l in np.where(bl[i])[0] if 0 < ps[i, :, l].sum() < n_i]
            if not ls:
                continue
            W = ps[i][:, ls]
            kap = W.sum(1)
            for l in range(len(ls)):
                w, nw = kap[W[:, l]], kap[~W[:, l]]
                num += (w[:, None] > nw[None, :]).sum() + 0.5 * (w[:, None] == nw[None, :]).sum()
                den += len(w) * len(nw)
        _att.append((round(hi, 4), round(num / den, 4), round(lo + hi, 6)))
    check('theory remark', 'Gamma_AUC attained by the coverage ranking; lower bound = 1 - Gamma_AUC (VOC x2)',
          all(a == b and c == 1.0 for a, b, c in _att), _att)
    z = np.load(os.path.join(_D, 'voc2012seg_nc_patch_gt.npz'), allow_pickle=True)
    ps, bl = z['patch_seg'].astype(bool), z['bag_labels'].astype(bool)
    keep = []
    for i in range(bl.shape[0]):
        if bl[i].sum() < 2:
            continue
        ws = [set(np.where(ps[i, :, l])[0]) for l in np.where(bl[i])[0]]
        ws = [w for w in ws if 0 < len(w) < ps.shape[1]]
        if len(ws) >= 2 and all(not (ws[a] & ws[b]) for a in range(len(ws)) for b in range(a + 1, len(ws))):
            keep.append(i)
    ch = 100 * chance_precision_at_1(ps[keep], bl[keep], None)
    ce = 100 * label_agnostic_ceiling(ps[keep], bl[keep], None)
    check('theory 4.2', 'VOC-Seg multi-label-disjoint bags: 95, chance 22.58, ceiling 47.74, +25.2',
          [len(keep), round(ch, 2), round(ce, 2), r1(ce - ch)] == [95, 22.58, 47.74, 25.2], (len(keep), ch, ce))
except Exception as exc:                                       # pragma: no cover
    check('theory', 'GT-based checks computable', False, exc)
_RESP = 'paper_attribution/revision_dmkd_r1/response/response.tex'
# withdrawn wording that must not return; kept in the private revision folder, which
# the public mirror does not carry (the phrases quote the confidential manuscript)
_PHRF = 'paper_attribution/revision_dmkd_r1/stale_phrases.json'
_PHR = json.load(open(_PHRF)) if os.path.exists(_PHRF) else {'response': [], 'manuscript': []}
if os.path.exists(_RESP):                      # the letter is not part of the public mirror
    _resp = open(_RESP).read()
    _resp = '\n'.join(l for l in _resp.splitlines() if not l.lstrip().startswith('%'))
    _bad = [p for p in _PHR['response'] if _re.search(p, _resp)]
_sig = {d: sorted({c['folds_same_sign'] for c in ST[d]['comparisons'] if c['significant_fold']}) for d in DS6}
check('appendix C', '28 comparisons per benchmark; significant ones keep sign in 5/5 (classical), >=9/10 (VOC)',
      all(len(ST[d]['comparisons']) == 28 for d in DS6) and all(_sig[d] == [5] for d in CLS)
      and all(min(_sig[d]) >= 9 for d in DS6[4:]), _sig)
if PAPER:
    _hp = open(os.path.join(T, 'hyperparams.tex')).read()
    check('appendix A', 'Table 9: attention width row excludes DSMIL (official DSMIL q-dim 128)',
          'DSMIL-sep, HAP; DSMIL uses $128$' in _hp and "q_dim=128" in open('baselines/dsmil_orig.py').read(), '')
if os.path.exists(_RESP):
    check('response', 'letter free of the corrected values (AUC averages, F1 spans, section refs, ...)', not _bad, _bad)

# ---- stale v1 phrases (captions and prose are not otherwise checked) --------------
import re as _re
if PAPER:
    _src = ''.join(open(f).read() for f in glob.glob('paper_attribution/sections/*.tex')
                   + ['paper_attribution/main.tex', 'paper_attribution/figures/overview_caption.tex'])
    _src = '\n'.join(l for l in _src.splitlines() if not l.lstrip().startswith('%'))
    _stale = [pat for pat in _PHR['manuscript'] if _re.search(pat, _src)]
    check('manuscript', 'no stale v1 phrases (seven archs, 3-model VOC, 1,022, ...)', not _stale, _stale)

# ---- crop effect (analysis voc-sensitivity, R1.4b) ------------------------------
if all((m, 'voc2012segml_crop') in CELLS for m in MAIN):
    dP = [delta(m, 'voc2012segml_nc') - delta(m, 'voc2012segml_crop') for m in MAIN]
    dA = [100 * (im(m, 'voc2012segml_nc', 'BagAveragePrecision') - im(m, 'voc2012segml_crop', 'BagAveragePrecision')) for m in MAIN]
    check('analysis voc', 'crop: AP +1.7..+2.9 for every arch; margins -0.5..+3.0',
          [r1(min(dA)), r1(max(dA)), r1(min(dP)), r1(max(dP))] == [1.7, 2.9, -0.5, 3.0], (dA, dP))
    check('analysis voc', 'crop: DSMIL-sep +3.2 (crop) / +2.8 (no crop)',
          [r1(delta('dsmil', 'voc2012segml_crop')), r1(delta('dsmil', 'voc2012segml_nc'))] == [3.2, 2.8])
    from itertools import combinations
    flips = [(a, b) for a, b in combinations(MAIN, 2)
             if (delta(a, 'voc2012segml_nc') - delta(b, 'voc2012segml_nc'))
             * (delta(a, 'voc2012segml_crop') - delta(b, 'voc2012segml_crop')) < 0]
    st_nc = {frozenset((c['a'], c['b'])): c for c in ST['voc2012segml_nc']['comparisons']}
    sig_flips = [f for f in flips if st_nc[frozenset(f)]['significant_fold']]
    check('analysis voc', 'crop: no significant comparison changes direction', not sig_flips,
          f'{len(flips)} mean-order swaps, {len(sig_flips)} among fold-significant')
    top = sorted(MAIN, key=lambda m: -delta(m, 'voc2012segml_crop'))[:2]
    check('analysis voc', 'crop: ML-Dec and Q2L strongest attributors on both',
          set(top) == {'ml_decoder', 'q2l'} and set(sorted(MAIN, key=lambda m: -delta(m, 'voc2012segml_nc'))[:2]) == {'ml_decoder', 'q2l'})

# ---- harness does not favour HAP (setup, R1.6c) ----------------------------------
dsets = CLS + [d for d in ('voc2012segml_nc', 'voc2012seg_nc') if all((m, d) in CELLS for m in MAIN)]
nb = {d: sum(im(m, d, 'BagAveragePrecision') > im('hap_only', d, 'BagAveragePrecision')
             for m in MAIN if m != 'hap_only') for d in dsets}
check('setup', '>= 2 other archs beat HAP per-model AP on every benchmark', min(nb.values()) >= 2, nb)


# ---- DMKD R1 panel review (2026-11-25): two-sided tests and new claims -------------
check('setup stats', 'all tests two-sided (no one-sided alternative in attribution_stats.py)',
      "alternative='greater'" not in open('analysis/attribution_stats.py').read()
      and '2 * t_dist.sf(abs(tval)' in open('analysis/attribution_stats.py').read())
_hk = json.load(open(os.path.join(T, 'hap_topk_identity.json')))
_hs = [100 * _hk[d]['share_identical_mean'] for d in PART]
check('analysis HAP', 'alpha~0: top-k set label-identical in 99.4..99.8% of bags > k; alpha = sigma(-10)',
      [r1(min(_hs)), r1(max(_hs))] == [99.4, 99.8]
      and all(abs(_hk[d]['alpha_max'] - 1 / (1 + np.exp(10))) < 1e-9 for d in PART), _hs)
DS6 = CLS + ['voc2012seg_nc', 'voc2012segml_nc']
_sel = lambda m: [EX[f'{m}__{d}']['del_selective_present'] for d in DS6]
_ag = [x for m in AG for x in _sel(m)]
check('analysis del.', 'per-pair selectivity: agnostic 0.43..0.49; ML-Dec 0.66..0.77; Q2L 0.61..0.74; DSMIL 0.61..0.76',
      [round(min(_ag), 2), round(max(_ag), 2)] == [0.43, 0.49]
      and [round(min(_sel('ml_decoder')), 2), round(max(_sel('ml_decoder')), 2)] == [0.66, 0.77]
      and [round(min(_sel('q2l')), 2), round(max(_sel('q2l')), 2)] == [0.61, 0.74]
      and [round(min(_sel('dsmil_orig')), 2), round(max(_sel('dsmil_orig')), 2)] == [0.61, 0.76])
_hsel = {d: EX[f'hap_only__{d}']['del_selective_present'] for d in DS6}
check('analysis del.', 'HAP selectivity 0.54..0.64 on MSRC, BS, VOC; ~0.5 on Letters; DSMIL-sep MSRC 0.43',
      [round(min(_hsel[d] for d in DS6[2:]), 2), round(max(_hsel[d] for d in DS6[2:]), 2)] == [0.54, 0.64]
      and all(abs(_hsel[d] - 0.5) < 0.03 for d in PART[:2])
      and round(EX['dsmil__mscv2']['del_selective_present'], 2) == 0.43, _hsel)
_ms = {m: r1(100 * EX[f'{m}__hja_birdsong']['ml_P@1']) for m in COND}
check('analysis 9.1', 'multi-species: Q2L 49.4 above 38.5; DSMIL-sep 34.0 and HAP 34.4 below',
      (_ms['q2l'], _ms['dsmil'], _ms['hap_only']) == (49.4, 34.0, 34.4)
      and r1(100 * EX['attnmil__hja_birdsong']['ml_ceiling']) == 38.5, _ms)
_bg = [100 * EX[f'{m}__voc2012seg_nc']['frac_miss_background'] for m in AG]
check('analysis voc', 'VOC-Seg agnostic misses: 36..46% background, 54..64% other object',
      [round(min(_bg)), round(max(_bg)), round(100 - max(_bg)), round(100 - min(_bg))] == [36, 46, 54, 64], _bg)
_tm = {m: (EX[f'{m}__voc2012seg_nc']['tau0.05_P@1'] - EX[f'{m}__voc2012seg_nc']['tau0.05_chance'])
          - (EX[f'{m}__voc2012seg_nc']['tau0.5_P@1'] - EX[f'{m}__voc2012seg_nc']['tau0.5_chance']) for m in MAIN} \
    if 'tau0.05_P@1' in EX['attnmil__voc2012seg_nc'] else None
if _tm:
    check('analysis voc', 'tau shrink: agnostic 17..18 points, ML-Dec/Q2L 12..13',
          [round(100 * min(_tm[m] for m in AG)), round(100 * max(_tm[m] for m in AG)),
           round(100 * min(_tm['ml_decoder'], _tm['q2l'])), round(100 * max(_tm['ml_decoder'], _tm['q2l']))] == [17, 18, 12, 13], _tm)
_best = {d: max(delta(m, d) for m in COND) for d in DS6}
_add = {d: max(delta('attnmil', d, 'AddP@1'), delta('gated_attn', d, 'AddP@1')) for d in DS6}
_addmin = {d: min(delta('attnmil', d, 'AddP@1'), delta('gated_attn', d, 'AddP@1')) for d in DS6}
check('analysis add.', 'additive: level with best label-cond. on LF, LC, VOC-Seg; behind on MSRC (9..11), BS, Seg-ML',
      all(_add[d] >= _best[d] - 2 for d in ('letter_frost', 'letter_carroll', 'voc2012seg_nc'))
      and all(_add[d] < _best[d] - 2 for d in ('mscv2', 'hja_birdsong', 'voc2012segml_nc'))
      and [round(_best['mscv2'] - _add['mscv2']), round(_best['mscv2'] - _addmin['mscv2'])] == [9, 11], (_best, _add))
check('discussion', 'additive 26..49 points over chance; up to 15 behind (conclusion); ML-Dec BS 40.3',
      [round(min(_addmin.values())), round(max(_add.values())),
       round(max(_best[d] - _addmin[d] for d in DS6)), r1(_best['hja_birdsong'])] == [26, 49, 15, 40.3])
_bb = json.load(open(os.path.join(T, 'voc_bbox_agreement.json')))
check('limitations', 'box-derived labels: 2,913 images, recall 95.7, precision 74.4 at tau 0.05',
      [_bb['images'], r1(100 * _bb['recall']), r1(100 * _bb['precision']), _bb['threshold']] == [2913, 95.7, 74.4, 0.05], _bb)
_pm = lambda S: sum(c['significant'] for d in DS6 for c in S[d]['comparisons'] if c['a'] in COND and c['b'] in COND)
check('analysis bag', 'secondary per-model test: attribution 54/60, bag AP 35/60', (_pm(ST), _pm(SA)) == (54, 35), (_pm(ST), _pm(SA)))
if PAPER and os.path.exists('paper_attribution/main.pdf') and os.path.exists(_RESP):
    import subprocess as _sp
    _np = int(_re.search(r'Pages:\s+(\d+)', _sp.run(['pdfinfo', 'paper_attribution/main.pdf'], capture_output=True, text=True).stdout).group(1))
    _decl = next(p for p in range(1, _np + 1) if _re.search(r'^Statements and Declarations', _sp.run(
        ['pdftotext', '-f', str(p), '-l', str(p), 'paper_attribution/main.pdf', '-'], capture_output=True, text=True).stdout, _re.M))
    _rt = open(_RESP).read()
    check('response R1.1f', 'letter page counts: whole PDF 58 -> N, main text 46 -> page of Declarations',
          f'from $58$ to ${_np}$' in _rt and f'from $46$ to ${_decl}$' in _rt, (_np, _decl))
if PAPER:
    _abs = open('paper_attribution/sections/abstract.tex').read()
    _abs = _abs[_abs.index('Attention maps'):]
    _nw = len(_re.sub(r'[$][^$]*[$]', 'N', _abs).split())
    check('abstract', 'abstract within 250 words, AUC and DSMIL spelled out', _nw <= 250
          and '(AUC)' in _abs and '(DSMIL)' in _abs, _nw)


# ---- R2.3: label imbalance, head vs tail (analysis/imbalance_analysis.py) -------------
_IMB = json.load(open(os.path.join(T, 'imbalance.json')))
from scipy.stats import t as _tdist
def _nb2(fm):
    fm = np.asarray(fm, float); k = len(fm); v = fm.var(ddof=1)
    return 1.0 if v == 0 else float(min(1.0, 2 * _tdist.sf(abs(fm.mean() / np.sqrt((1 / k + 1 / (k - 1)) * v)), k - 1)))
_bad_p = [(c, m) for c, r in _IMB['cells'].items() for m in ('test_AP', 'test_APx', 'test_APn', 'test_F1', 'test_M')
          if r[m].get('testable') and abs(_nb2(r[m]['fold_means']) - r[m]['p']) > 1e-9]
check('imbalance', '1,600 models; every head-tail NB p recomputed from fold means', _IMB['gate_models_checked'] == 1600 and not _bad_p, _bad_p[:3])
_B = _IMB['benchmarks']
check('imbalance', 'MeanIR 3.1 (BS) .. 15.8 (Seg-ML); 2..7 labels < 5%; VOC-Seg min count 89; classical tail freq 0.5..10.6%',
      [r1(min(v['mean_ir'] for v in _B.values())), r1(_B['hja_birdsong']['mean_ir']), r1(max(v['mean_ir'] for v in _B.values())), r1(_B['voc2012segml_nc']['mean_ir'])] == [3.1, 3.1, 15.8, 15.8]
      and [min(v['n_labels_lt5pct'] for v in _B.values()), max(v['n_labels_lt5pct'] for v in _B.values())] == [2, 7]
      and _B['voc2012seg_nc']['min_count'] == 89
      and [r1(100 * min(_B[d]['tail_freq_range'][0] for d in CLS)), r1(100 * max(_B[d]['tail_freq_range'][1] for d in CLS))] == [0.5, 10.6])
_C = _IMB['counts']
_apc = [100 * _IMB['cells'][f'{m}__{d}']['AP_diff'] for m in MAIN for d in CLS]
_aps = [100 * _IMB['cells'][f'{m}__voc2012seg_nc']['AP_diff'] for m in MAIN]
check('imbalance', 'AP tail lower 48/48, significant 33 (all 32 classical, 0 VOC-Seg); 17.6..51.3 classical; VOC-Seg 1.7..2.5',
      (_C['AP']['all']['mean_diff_positive'], _C['AP']['all']['head>tail'], _C['AP']['all']['tail>head']) == (48, 33, 0)
      and all(_IMB['cells'][f'{m}__{d}']['test_AP']['significant'] for m in MAIN for d in CLS)
      and not any(_IMB['cells'][f'{m}__voc2012seg_nc']['test_AP']['significant'] for m in MAIN)
      and [r1(min(_apc)), r1(max(_apc)), r1(min(_aps)), r1(max(_aps))] == [17.6, 51.3, 1.7, 2.5])
check('imbalance', 'F1 tail significantly lower 41/48; Micro > Macro in 48/48 by 3.1..19.0',
      _C['F1']['all']['head>tail'] == 41 and _C['F1']['all']['tail>head'] == 0 and _C['F1_gap']['n_micro_gt_macro'] == 48
      and [r1(100 * _C['F1_gap']['min']), r1(100 * _C['F1_gap']['max'])] == [3.1, 19.0])
_sig = sorted(c for c in (f'{m}__{d}' for m in COND for d in DS6) if _IMB['cells'][c]['test_M']['significant'] and _IMB['cells'][c]['test_M']['direction'] == 'head>tail')
check('imbalance', 'attribution (label-cond.): tail lower 26/30, significant 5 (DSMIL/ML-Dec/Q2L MSRC, Q2L LC, DSMIL BS), 1 reversed',
      (_C['M']['label_conditional']['mean_diff_positive'], _C['M']['label_conditional']['n'], _C['M']['label_conditional']['head>tail'], _C['M']['label_conditional']['tail>head']) == (26, 30, 5, 1)
      and _sig == sorted(['dsmil_orig__mscv2', 'ml_decoder__mscv2', 'q2l__mscv2', 'q2l__letter_carroll', 'dsmil_orig__hja_birdsong']), _sig)
_mt = [100 * _IMB['cells'][f'{m}__mscv2']['M_tail'] for m in COND]
check('imbalance', 'MSRC: no label-conditional tail margin above +3 (-4.7..+2.8); ML-Dec head +44.4',
      [r1(min(_mt)), r1(max(_mt)), r1(100 * _IMB['cells']['ml_decoder__mscv2']['M_head'])] == [-4.7, 2.8, 44.4], _mt)
_cl = [f'{m}__{d}' for m in MAIN for d in CLS]
_vc = [f'{m}__{d}' for m in MAIN for d in ('voc2012seg_nc', 'voc2012segml_nc')]
_ci = _IMB['cells']
_fl = sorted({r1(100 * (_ci[c]['AP_diff'] - _ci[c]['APx_diff'])) for c in _cl})
check('imbalance', 'prevalence floor head-tail 18.5..22.9 (classical); raw AP significant 1/16 on VOC',
      [_fl[0], _fl[-1]] == [18.5, 22.9] and sum(_ci[c]['test_AP']['significant'] for c in _vc) == 1, _fl)
_xs = sorted(c for c in _cl if _ci[c]['test_APx']['significant'])
check('imbalance', 'AP-pi: classical tail lower 31/32, significant 7 all Letter; VOC tail higher 16/16, 3 significant',
      sum(_ci[c]['APx_diff'] > 0 for c in _cl) == 31 and len(_xs) == 7
      and all(_ci[c]['test_APx']['direction'] == 'head>tail' and 'letter' in c for c in _xs)
      and sum(_ci[c]['APx_diff'] < 0 for c in _vc) == 16
      and [_ci[c]['test_APx']['direction'] for c in _vc if _ci[c]['test_APx']['significant']] == ['tail>head'] * 3, _xs)
check('imbalance', '(AP-pi)/(1-pi): tail lower 48/48, significant 26/32 classical (all on LF, LC, MSRC), 0 VOC',
      _C['APn']['all']['mean_diff_positive'] == 48
      and sum(_ci[c]['test_APn']['significant'] for c in _cl) == 26
      and all(_ci[f'{m}__{d}']['test_APn']['significant'] and _ci[f'{m}__{d}']['APx_diff'] > 0
              for m in MAIN for d in ('letter_frost', 'letter_carroll', 'mscv2'))
      and not any(_ci[c]['test_APn']['significant'] for c in _vc) and _C['APn']['all']['tail>head'] == 0)
_pos = [100 * c['micro_minus_macro_f1_labels_with_pos'] for c in _ci.values()]
_zl = [100 * (_ci[f'{m}__{d}']['micro_minus_macro_f1'] - _ci[f'{m}__{d}']['micro_minus_macro_f1_labels_with_pos'])
       for m in MAIN for d in ('letter_frost', 'letter_carroll')]
_zs = [z / (100 * _ci[f'{m}__{d}']['micro_minus_macro_f1']) for z, (m, d) in
       zip(_zl, [(m, d) for m in MAIN for d in ('letter_frost', 'letter_carroll')])]
check('imbalance', 'Letter: labels without a positive test bag account for 4.7..9.1 points (a quarter to a half) of Micro-Macro',
      [r1(min(_zl)), r1(max(_zl))] == [4.7, 9.1] and 0.25 <= min(_zs) and max(_zs) <= 0.55, [r1(min(_zs), 2), r1(max(_zs), 2)])
check('imbalance', 'Micro-Macro over labels with a positive: 3.1..15.3; F1 tail lower 48/48',
      [r1(min(_pos)), r1(max(_pos))] == [3.1, 15.3] and _C['F1']['all']['mean_diff_positive'] == 48)
_rev = sorted(c for c, r in _ci.items() if r['test_M']['significant'] and r['test_M']['direction'] == 'tail>head')
check('imbalance', 'attribution reversals: DSMIL-sep (head margin +3.4) and Gated AB-MIL on VOC-Seg',
      _rev == ['dsmil__voc2012seg_nc', 'gated_attn__voc2012seg_nc'] and r1(100 * _ci['dsmil__voc2012seg_nc']['M_head']) == 3.4, _rev)
if PAPER:
    check('imbalance', 'tab:imbalance in the manuscript', '\\input{tables/imbalance}' in open('paper_attribution/main.tex').read())


# ---- R1.4c: COCO-Patch-ML 8x8 (analysis/coco_patch_summary.py, subsec:coco) ------
_RC = 'results/instance_level_r1_coco'
_CC = {}
for _f in glob.glob(os.path.join(_RC, '*__coco2017val_g8.json')):
    _d = json.load(open(_f))
    _CC[_d['method']] = _d
_CO = json.load(open(os.path.join(T, 'coco_patch.json')))
_CX = json.load(open(os.path.join(T, 'coco_extras.json')))['cells']
_PM, _RG = _CO['per_method'], _CO['regime']
_LC4 = ['dsmil_orig', 'ml_decoder', 'q2l', 'hap_only']


def _cp(m, k='InstP@1'):
    return r1(100 * _PM[m][k][0])


def _ct(key, a, b):
    for r in _CO[key]:
        if {r['a'], r['b']} == {a, b}:
            return r


_bad = [m for m in MAIN if m not in _CC or sorted(r['fold'] for r in _CC[m]['instance_per_model']) != list(range(10))
        or _CC[m]['config'].get('batch_size') != 8 or _CC[m]['config'].get('embed_dim') != 128
        or not str(_CC[m].get('device', '')).startswith('cpu')
        or any(isinstance(v, float) and v != v for r in _CC[m]['instance_per_model'] for v in r.values())]
check('coco', '8 cells x 10 models (folds 0..9), batch 8, VOC widths, CPU, no NaN', not _bad, _bad)
_viol = [(m, r['fold']) for m in AG for r in _CC[m]['instance_per_model'] if r['InstP@1'] > r['LabelAgnosticCeiling'] + 1e-12]
check('coco', 'Theorem 1: every label-agnostic model P@1 <= its ceiling', not _viol, _viol)
check('coco', '3,766 bags, 80 labels, 64 patches, 3.3 labels/bag (2..13), 12.1 witnesses, all pairs witnessed',
      (_RG['bags'], _RG['labels'], _RG['patches_per_bag'], r1(_RG['labels_per_bag']), _RG['min_labels_per_bag'],
       _RG['max_labels_per_bag'], r1(_RG['witnesses_per_pair_mean']), _RG['pairs_without_witness'])
      == (3766, 80, 64, 3.3, 2, 13, 12.1, 0))
check('coco', 'disjoint witness sets 13.6%; chance 18.5, ceiling 65.2 (fold avg); Gamma_AUC 0.877',
      r1(100 * _RG['share_bags_disjoint_witnesses']) == 13.6 and _cp('attnmil', 'ChanceP@1') == 18.5
      and _cp('attnmil', 'LabelAgnosticCeiling') == 65.2 and round(_RG['dataset_gamma_auc'][1], 3) == 0.877)
check('coco', 'label-agnostic P@1 30.7..32.9, margins +12.3..+14.4, > 30 below ceiling',
      [min(_cp(m) for m in AG), max(_cp(m) for m in AG)] == [30.7, 32.9]
      and [min(_cp(m, 'margin') for m in AG), max(_cp(m, 'margin') for m in AG)] == [12.3, 14.4]
      and all(_PM[m]['LabelAgnosticCeiling'][0] - _PM[m]['InstP@1'][0] > 0.30 for m in AG))
check('coco', 'ML-Dec 57.6 (+39.1), Q2L 57.2 (+38.8), DSMIL 43.4, HAP 41.2, DSMIL-sep 20.0 (+1.5)',
      (_cp('ml_decoder'), _cp('ml_decoder', 'margin'), _cp('q2l'), _cp('q2l', 'margin'), _cp('dsmil_orig'),
       _cp('hap_only'), _cp('dsmil'), _cp('dsmil', 'margin')) == (57.6, 39.1, 57.2, 38.8, 43.4, 41.2, 20.0, 1.5))
_x = [_ct('tests_P1', a, b) for a in _LC4 for b in AG]
_dd = [abs(100 * r['mean_diff']) for r in _x]
check('coco', 'each of DSMIL/ML-Dec/Q2L/HAP > each label-agnostic: 10/10 folds, sig., 8.3..26.9',
      all(r['significant'] and r['folds_same_sign'] == 10 for r in _x)
      and all((r['mean_diff'] > 0) == (r['a'] in _LC4) for r in _x) and [r1(min(_dd)), r1(max(_dd))] == [8.3, 26.9])
check('coco', 'DSMIL-sep significantly below every label-agnostic architecture',
      all(_ct('tests_P1', 'dsmil', m)['significant'] and _PM['dsmil']['InstP@1'][0] < _PM[m]['InstP@1'][0] for m in AG))
check('coco', 'Holm over 28 pairs', len(_CO['tests_P1']) == len(_CO['tests_AUC']) == len(_CO['tests_BagAP']) == 28)
_ds = {m: _CX[f'{m}__coco2017val_g8'] for m in MAIN}
check('coco', 'deletion per-pair: label-agnostic 0.47..0.48; ML-Dec 0.61, Q2L 0.63, DSMIL 0.71; ML-Dec 0.47 vs 0.03',
      [r1(min(_ds[m]['del_selective_present'] for m in AG), 2), r1(max(_ds[m]['del_selective_present'] for m in AG), 2)] == [0.47, 0.48]
      and [r1(_ds[m]['del_selective_present'], 2) for m in ('ml_decoder', 'q2l', 'dsmil_orig')] == [0.61, 0.63, 0.71]
      and (r1(_ds['ml_decoder']['del_self_present'], 2), r1(_ds['ml_decoder']['del_other_present'], 2)) == (0.47, 0.03))
check('coco', 'additive P@1 AB-MIL 55.7, gated 53.7, within 4 points of ML-Dec and Q2L',
      (_cp('attnmil', 'AddP@1'), _cp('gated_attn', 'AddP@1')) == (55.7, 53.7)
      and max(_PM[c]['InstP@1'][0] for c in ('ml_decoder', 'q2l')) - min(_PM[a]['AddP@1'][0] for a in ('attnmil', 'gated_attn')) < 0.04)
_ap = {m: _PM[m]['BagAveragePrecision'][0] for m in MAIN}
check('coco', 'bag AP 0.648..0.701, 12/28 significant, top three ML-Dec/Q2L/HAP; P@1 20.0..57.6',
      [round(min(_ap.values()), 3), round(max(_ap.values()), 3)] == [0.648, 0.701]
      and sum(r['significant'] for r in _CO['tests_BagAP']) == 12
      and set(sorted(_ap, key=_ap.get)[-3:]) == {'ml_decoder', 'q2l', 'hap_only'}
      and [min(_cp(m) for m in MAIN), max(_cp(m) for m in MAIN)] == [20.0, 57.6])
_au = {m: round(_PM[m]['InstAUCPooled'][0], 3) for m in MAIN}
check('coco', 'pooled AUC: agnostic .553..640; ML-Dec/Q2L/HAP .635..679, none sig.; DSMIL .507, DSMIL-sep .420 lowest',
      [min(_au[m] for m in AG), max(_au[m] for m in AG)] == [0.553, 0.640]
      and [min(_au[m] for m in ('ml_decoder', 'q2l', 'hap_only')), max(_au[m] for m in ('ml_decoder', 'q2l', 'hap_only'))] == [0.635, 0.679]
      and not any(_ct('tests_AUC', a, b)['significant'] for a in AG for b in ('ml_decoder', 'q2l', 'hap_only'))
      and (_au['dsmil_orig'], _au['dsmil']) == (0.507, 0.420) and sorted(_au, key=_au.get)[:2] == ['dsmil', 'dsmil_orig'])
_fa = {m: [r['InstAUCPooled'] for r in _CC[m]['instance_per_model']] for m in ('attnmil', 'gated_attn', 'transmil_lite')}
_lo = [x for m in ('attnmil', 'gated_attn') for x in _fa[m] if x < 0.6]
_hi = [x for m in ('attnmil', 'gated_attn') for x in _fa[m] if x >= 0.6]
check('coco', 'AUC per fold: AB-MIL, gated 7 folds 0.48..0.57, 3 folds 0.68..0.72; TransMIL 0.61..0.66; diff to ML-Dec 0.10..0.13',
      all(sum(x < 0.6 for x in _fa[m]) == 7 for m in ('attnmil', 'gated_attn'))
      and [round(min(_lo), 2), round(max(_lo), 2), round(min(_hi), 2), round(max(_hi), 2)] == [0.48, 0.57, 0.68, 0.72]
      and [round(min(_fa['transmil_lite']), 2), round(max(_fa['transmil_lite']), 2)] == [0.61, 0.66]
      and sorted(round(abs(_ct('tests_AUC', m, 'ml_decoder')['mean_diff']), 2) for m in ('attnmil', 'gated_attn')) == [0.1, 0.13])
check('coco', 'AUC: DSMIL, DSMIL-sep sig. below TransMIL (0.132, 0.219), ML-Dec, Q2L, HAP',
      all(_ct('tests_AUC', d, m)['significant'] for d in ('dsmil_orig', 'dsmil') for m in ('transmil_lite', 'ml_decoder', 'q2l', 'hap_only'))
      and [round(abs(_ct('tests_AUC', 'transmil_lite', d)['mean_diff']), 3) for d in ('dsmil_orig', 'dsmil')] == [0.132, 0.219])
check('coco', '12,466 present pairs, 70 all-witness (not scorable); deletion HAP 0.51, DSMIL-sep 0.53',
      _RG['pairs_all_patches_witness'] == 70 and int(round(_RG['labels_per_bag'] * _RG['bags'])) == 12466
      and (r1(_ds['hap_only']['del_selective_present'], 2), r1(_ds['dsmil']['del_selective_present'], 2)) == (0.51, 0.53))
check('coco', 'ML-Dec, Q2L margins 4..5 below VOC 8x8 (+44.2, +43.0); AB-MIL +13.4 there',
      all(4 <= delta(m, 'voc2012segml_g8') - 100 * _PM[m]['margin'][0] <= 5.2 for m in ('ml_decoder', 'q2l'))
      and (r1(delta('ml_decoder', 'voc2012segml_g8')), r1(delta('q2l', 'voc2012segml_g8')), r1(delta('attnmil', 'voc2012segml_g8'))) == (44.2, 43.0, 13.4))
check('coco', 'models: 2,480 + 80 COCO = 2,560', sum(len(c['instance_per_model']) for c in CELLS.values())
      + sum(len(c['instance_per_model']) for c in _CC.values()) == 2560)
_GTP = next((p for p in [os.path.join(d, 'coco2017val_g8_patch_gt.npz') for d in
             (os.environ.get('COCO_PATCH_DIR', ''), os.path.expanduser('~/data/coco_patch'), 'data/coco_patch') if d]
             if os.path.exists(p)), '')
if os.path.exists(_GTP):
    _z = np.load(_GTP)
    _ps, _bl = _z['patch_seg'], _z['bag_labels']
    check('coco', 'patch labels = overlap >= 0.05 (float16 store); bag labels = OR of patch labels; >= 2 labels per bag',
          np.array_equal(_ps, _z['overlap_frac'] >= np.float16(0.05)) and np.array_equal(_bl, _ps.any(1))
          and int(_bl.sum(1).min()) >= 2 and int(_z['grid']) == 8)
if PAPER:
    check('coco', 'tab:coco in the manuscript', '\\input{tables/coco_patch}' in open('paper_attribution/sections/analysis.tex').read())

# ---- data provenance -------------------------------------------------------------
try:
    from scipy.io import loadmat
    from datasets.voc_miml import DEFAULT_OUT_DIR as D
    for stem in ('voc2012seg', 'voc2012segml'):
        a = loadmat(os.path.join(D, f'{stem}.mat'), variable_names=['labels'])['labels']
        b = loadmat(os.path.join(D, f'{stem}_nc.mat'), variable_names=['labels'])['labels']
        za, zb = np.load(os.path.join(D, f'{stem}_patch_gt.npz')), np.load(os.path.join(D, f'{stem}_nc_patch_gt.npz'))
        check('benchmarks', f'{stem}: no-crop file keeps labels and patch GT',
              np.array_equal(a, b) and np.array_equal(za['patch_seg'], zb['patch_seg'])
              and np.array_equal(za['patch_seg'], zb['overlap_frac'].astype(np.float32) >= 0.05))
except Exception as exc:                                       # pragma: no cover
    check('benchmarks', 'VOC files readable', False, exc)

print()
if fails:
    print(f'{len(fails)} CHECK(S) FAILED')
    for f in fails:
        print('   -', f)
    sys.exit(1)
print('all checks passed')

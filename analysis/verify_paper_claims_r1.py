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
check('analysis nec.', 'AB-MIL > DSMIL-sep MSRC: 4.0, fold p 0.046, delta 0.74',
      c['a'] == 'attnmil' and [r1(100 * c['mean_diff']), round(c['p_fold_nb_holm'], 3),
                               round(c['cliffs_delta'], 2)] == [4.0, 0.046, 0.74])
c = comp(ST, 'mscv2', 'dsmil_orig', 'dsmil')
check('analysis nec.', 'DSMIL vs DSMIL-sep MSRC: 27.4, p 3.4e-4, delta 0.998',
      c['a'] == 'dsmil_orig' and r1(100 * c['mean_diff']) == 27.4
      and float(f"{c['p_fold_nb_holm']:.1e}") == 3.4e-4 and round(c['cliffs_delta'], 3) == 0.998)
ap_o, ap_s = im('dsmil_orig', 'mscv2', 'BagAveragePrecision'), im('dsmil', 'mscv2', 'BagAveragePrecision')
check('abstract', 'DSMIL pair: 3 pts AP (0.773 vs 0.743), 27 pts attribution',
      [round(ap_o, 3), round(ap_s, 3), round(100 * (ap_o - ap_s))] == [0.773, 0.743, 3])

# ---- label-query designs ----------------------------------------------------
mla = [im('ml_decoder', d, 'InstAUCPooled') for d in CLS]
qa = [im('q2l', d, 'InstAUCPooled') for d in CLS]
check('analysis nec.', 'pooled AUC ML-Dec 0.735..0.808, Q2L 0.678..0.710',
      [round(min(mla), 3), round(max(mla), 3), round(min(qa), 3), round(max(qa), 3)]
      == [0.735, 0.808, 0.678, 0.710])
cs = [comp(ST, d, 'ml_decoder', 'q2l') for d in CLS]
check('analysis nec.', 'ML-Dec > Q2L +5.4 +9.0 +12.4 (p<=0.009) and +12.4 BS (p 0.15, 5/5)',
      all(c['a'] == 'ml_decoder' for c in cs)
      and [r1(100 * c['mean_diff']) for c in cs] == [5.4, 9.0, 12.4, 12.4]
      and all(c['p_fold_nb_holm'] <= 0.009 for c in cs[:3])
      and round(cs[3]['p_fold_nb_holm'], 2) == 0.15 and cs[3]['folds_same_sign'] == 5)
check('analysis nec.', 'ML-Dec > Q2L delta 0.57..0.94 on partitioned',
      [round(c['cliffs_delta'], 2) for c in cs[:3]] == [0.57, 0.89, 0.94])
check('analysis nec.', 'ML-Dec vs TransMIL Cliff delta = 1.00 on all four',
      all(comp(ST, d, 'ml_decoder', 'transmil_lite')['cliffs_delta'] == 1.0 for d in CLS))

# ---- bag AP vs attribution ------------------------------------------------------
def nsig(st, d):
    return sum(c['significant_fold'] for c in st[d]['comparisons']
               if c['a'] in COND and c['b'] in COND)
na, npp = sum(nsig(SA, d) for d in CLS), sum(nsig(ST, d) for d in CLS)
check('analysis bag', 'fold-sig label-conditional pairs: attribution 28/40, AP 9/40',
      (npp, na) == (28, 9), (npp, na))
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
check('analysis 9.1', '63% of recordings have >=2 scorable species',
      round(100 * e['ml_bags'] * 5 / 548) == 63, f"{100 * e['ml_bags'] * 5 / 548:.1f}")
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
check('response R3.3', '90 per-model significant, 74 fold-level (classical)', (pm, pf) == (90, 74), (pm, pf))
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
    check('analysis nec.', 'Seg-ML DSMIL vs sep: 32.2, p 1.2e-8, delta 1.00, 10/10 folds',
          c['a'] == 'dsmil_orig' and r1(100 * c['mean_diff']) == 32.2
          and float(f"{c['p_fold_nb_holm']:.1e}") == 1.2e-8 and c['cliffs_delta'] == 1.0 and c['folds_same_sign'] == 10)
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
    check('analysis nec.', 'Seg DSMIL vs sep: 28.8, p 2.5e-8, delta 1.00, 10/10 folds',
          c['a'] == 'dsmil_orig' and r1(100 * c['mean_diff']) == 28.8
          and float(f"{c['p_fold_nb_holm']:.1e}") == 2.5e-8 and c['cliffs_delta'] == 1.0 and c['folds_same_sign'] == 10)
    aps = sorted(MAIN, key=lambda m: -im(m, S, 'BagAveragePrecision'))
    check('analysis nec.', 'Seg DSMIL-sep +12.1 weakest; AP 0.926 highest, = ML-Dec to 3 dp',
          r1(delta('dsmil', S)) == 12.1 and min(MAIN, key=lambda m: delta(m, S)) == 'dsmil'
          and aps[0] == 'dsmil' and round(im('dsmil', S, 'BagAveragePrecision'), 3) == 0.926
          == round(im('ml_decoder', S, 'BagAveragePrecision'), 3))
    # bag AP vs attribution over six benchmarks (subsec:bagap)
    ds6 = CLS + [V, S]
    check('analysis bag', 'six benchmarks: attribution 45/60, AP 11/60, AP 0 on Seg-ML',
          (sum(nsig(ST, d) for d in ds6), sum(nsig(SA, d) for d in ds6), nsig(SA, V)) == (45, 11, 0),
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
        ok = (r1(100 * ca['mean_diff']) == 1.0 and round(ca['p_fold_nb_holm'], 3) == 0.017
              and r1(100 * cp['mean_diff']) == 34.5 and float(f"{cp['p_fold_nb_holm']:.1e}") == 5.6e-9)
    check('analysis bag', 'exactly one opposite pair: Seg DSMIL-sep vs Q2L, AP +1.0 p .017 / P@1 34.5 p 5.6e-9',
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

# ---- stale v1 phrases (captions and prose are not otherwise checked) --------------
import re as _re
_src = ''.join(open(f).read() for f in glob.glob('paper_attribution/sections/*.tex')
               + ['paper_attribution/main.tex', 'paper_attribution/figures/overview_caption.tex'])
_src = '\n'.join(l for l in _src.splitlines() if not l.lstrip().startswith('%'))
_stale = [pat for pat in (r'seven architectures', r'\$3\$ of a VOC', r'VOC-Patch cells hold three',
                          r'1\{,\}022', r'one to two minutes', r'R1-VOC', r'PENDING',
                          r'six baseline', r'\$42\$ stored', r'two VOC-Patch feature files')
          if _re.search(pat, _src)]
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

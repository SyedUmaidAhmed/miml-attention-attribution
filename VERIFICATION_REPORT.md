# Verification report

Generated 2026-09-05 18:47 by `analysis/make_verification_report.py`. Every number below is read from the stored result files or produced by running the checks; none is typed in.

## 1. Environment

- Python 3.14.6 on macOS-26.5.2-arm64-arm-64bit-Mach-O
- torch 2.12.0
- numpy 2.4.2
- scipy 1.17.1
- sklearn 1.8.0
- matplotlib 3.10.8
- default device: mps
- git revision: `64947eb`

## 2. Claim verifier

`python3 analysis/verify_paper_claims.py` — **26 passed, 0 failed** (exit code 0).

```
  PASS  label-agnostic == chance to <6e-17                         worst=5.55e-17
  PASS  315 label-agnostic models                                  315
  PASS  735 models total on partitioned data                       735
  PASS  agnostic spread == 0 on letter_frost                       0.00e+00 pts
  PASS  agnostic spread == 0 on letter_carroll                     0.00e+00 pts
  PASS  agnostic spread == 0 on mscv2                              0.00e+00 pts
  PASS  agnostic architectures separate on VOC (spread > 5 pts)    10.3 pts
  PASS  all label-agnostic models > chance+10 on voc2012seg        {'attnmil': 33.6, 'gated_attn': 28.9, 'transmil_lite': 38.2}
  PASS  TransMIL strongest label-agnostic on voc2012seg            transmil_lite
  PASS  all label-agnostic models > chance+10 on voc2012segml      {'attnmil': 13.3, 'gated_attn': 20.2, 'transmil_lite': 23.6}
  PASS  TransMIL strongest label-agnostic on voc2012segml          transmil_lite
  PASS  DSMIL AUC < 0.5 on mscv2                                   0.452
  PASS  DSMIL AUC > 0.5 on letter_carroll                          0.598
  PASS  AttnMIL beats DSMIL on mscv2                               34.2 vs 30.1
  PASS  no label-agnostic model exceeds its ceiling                none
  PASS  VOC multi-label-disjoint headroom == 25.2 pts              95 bags, chance 22.6 -> ceiling 47.7 (+25.2)
  PASS  classical cells: 5 folds x 7 seeds = 35 models             
  PASS  VOC cells: 3 folds x 1 model                               
  PASS  VOC batch size 4 except AB-MIL/VOC-Patch-Seg = 8           {('attnmil', 'voc2012seg'): 8}
  PASS  HAP cells scored on eff_attn, baselines on attn_weights    
  PASS  VOC patch-label threshold == 0.05                          0.05
  PASS  VOC-Patch-Seg: 103 of 3,095 bag labels unwitnessed         103/3095
  PASS  VOC-Patch-Seg: 2,273 bags, 27.8% multi-label               2273 bags, 27.8%
  PASS  VOC-Patch-Seg-ML: 660 bags, OR-consistent, 2.29 labels/bag 660 bags, 2.29
  PASS  TransMIL exposes real encoder attention (no attn_proj)     
  PASS  stats appendix lists 63 pairwise tests                     63
all checks passed
```

## 3. Result cells

One JSON per (architecture, benchmark). `models` is the number of independently trained networks whose attribution was scored; `P@1`, `chance`, `AUC` are means over them; `AP` is the bag-level average precision of the cell (ensemble on the classical benchmarks, single model per fold on VOC-Patch). `sha256` is the first 12 hex digits of the file digest.

| Architecture | Benchmark | folds | models | AP | P@1 | chance | Δ | AUC | max \|P@1−chance\| per model | wall min | sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AB-MIL | Letter Frost | 5 | 35 | 0.676 | 26.1 | 26.1 | +0.0 | 0.502 | 5.6e-17 | 1 | `5c2533ccf881` |
| Gated AB-MIL | Letter Frost | 5 | 35 | 0.665 | 26.1 | 26.1 | +0.0 | 0.503 | 5.6e-17 | 1 | `a6738a05bd94` |
| TransMIL | Letter Frost | 5 | 35 | 0.741 | 26.1 | 26.1 | +0.0 | 0.503 | 5.6e-17 | 8 | `cb4eaf57d9bf` |
| DSMIL | Letter Frost | 5 | 35 | 0.770 | 35.7 | 26.1 | +9.6 | 0.594 | 2.4e-01 | 2 | `3f8b2c95c0d3` |
| ML-Decoder | Letter Frost | 5 | 35 | 0.801 | 55.0 | 26.1 | +29.0 | 0.751 | 3.7e-01 | 2 | `91f8d0290c8f` |
| Q2L | Letter Frost | 5 | 35 | 0.790 | 49.7 | 26.1 | +23.6 | 0.692 | 3.4e-01 | 6 | `84c1c0ed414d` |
| HAP | Letter Frost | 5 | 35 | 0.750 | 27.9 | 26.1 | +1.9 | 0.518 | 9.9e-02 | 6 | `4d34c9c665fd` |
| AB-MIL | Letter Carroll | 5 | 35 | 0.677 | 25.4 | 25.4 | +0.0 | 0.499 | 5.6e-17 | 1 | `ebba9305cec7` |
| Gated AB-MIL | Letter Carroll | 5 | 35 | 0.675 | 25.4 | 25.4 | +0.0 | 0.498 | 5.6e-17 | 2 | `9ddb87d6c9b0` |
| TransMIL | Letter Carroll | 5 | 35 | 0.733 | 25.4 | 25.4 | +0.0 | 0.502 | 5.6e-17 | 10 | `3be25afe9954` |
| DSMIL | Letter Carroll | 5 | 35 | 0.787 | 35.0 | 25.4 | +9.6 | 0.598 | 2.3e-01 | 2 | `ed88ae894601` |
| ML-Decoder | Letter Carroll | 5 | 35 | 0.801 | 58.9 | 25.4 | +33.5 | 0.778 | 4.0e-01 | 3 | `e09e5a6248e7` |
| Q2L | Letter Carroll | 5 | 35 | 0.797 | 49.9 | 25.4 | +24.5 | 0.697 | 3.3e-01 | 6 | `2c9c9968f40c` |
| HAP | Letter Carroll | 5 | 35 | 0.754 | 28.7 | 25.4 | +3.3 | 0.527 | 9.7e-02 | 7 | `2fd1a0d2555a` |
| AB-MIL | MSRC v2 | 5 | 35 | 0.715 | 34.2 | 34.2 | +0.0 | 0.500 | 0.0e+00 | 5 | `611549893089` |
| Gated AB-MIL | MSRC v2 | 5 | 35 | 0.714 | 34.2 | 34.2 | +0.0 | 0.500 | 0.0e+00 | 5 | `ab2a3f01f823` |
| TransMIL | MSRC v2 | 5 | 35 | 0.770 | 34.2 | 34.2 | +0.0 | 0.506 | 0.0e+00 | 29 | `b5c83650f74e` |
| DSMIL | MSRC v2 | 5 | 35 | 0.771 | 30.1 | 34.2 | -4.1 | 0.452 | 1.0e-01 | 5 | `5ca8e91be89f` |
| ML-Decoder | MSRC v2 | 5 | 35 | 0.816 | 71.7 | 34.2 | +37.5 | 0.804 | 4.5e-01 | 9 | `dddcfc5a205a` |
| Q2L | MSRC v2 | 5 | 35 | 0.835 | 59.3 | 34.2 | +25.1 | 0.714 | 3.6e-01 | 22 | `3323aa859c0f` |
| HAP | MSRC v2 | 5 | 35 | 0.817 | 42.3 | 34.2 | +8.1 | 0.568 | 2.0e-01 | 19 | `36ddc2ce572b` |
| AB-MIL | VOC-Patch-Seg | 3 | 3 | 0.892 | 73.6 | 39.9 | +33.6 | 0.745 | 3.5e-01 | 21 | `eae9b739c240` |
| Gated AB-MIL | VOC-Patch-Seg | 3 | 3 | 0.891 | 68.9 | 39.9 | +28.9 | 0.736 | 3.0e-01 | 25 | `20432deed2e9` |
| TransMIL | VOC-Patch-Seg | 3 | 3 | 0.900 | 78.1 | 39.9 | +38.2 | 0.819 | 3.9e-01 | 356 | `eb507edc5ca7` |
| DSMIL | VOC-Patch-Seg | 3 | 3 | 0.907 | 44.4 | 39.9 | +4.5 | 0.452 | 6.8e-02 | 28 | `64b6a3a132a6` |
| ML-Decoder | VOC-Patch-Seg | 3 | 3 | 0.890 | 88.4 | 39.9 | +48.5 | 0.795 | 5.0e-01 | 28 | `feab5423f5ef` |
| Q2L | VOC-Patch-Seg | 3 | 3 | 0.845 | 69.6 | 39.9 | +29.7 | 0.654 | 3.1e-01 | 39 | `7632d17e123d` |
| HAP | VOC-Patch-Seg | 3 | 3 | 0.902 | 82.7 | 39.9 | +42.8 | 0.817 | 4.5e-01 | 59 | `ae29fa54275c` |
| AB-MIL | VOC-Patch-Seg-ML | 3 | 3 | 0.832 | 47.4 | 34.1 | +13.3 | 0.590 | 1.6e-01 | 8 | `8815af4c6005` |
| Gated AB-MIL | VOC-Patch-Seg-ML | 3 | 3 | 0.840 | 54.2 | 34.1 | +20.2 | 0.620 | 2.3e-01 | 8 | `91d001c991c2` |
| TransMIL | VOC-Patch-Seg-ML | 3 | 3 | 0.871 | 57.7 | 34.1 | +23.6 | 0.724 | 2.6e-01 | 611 | `405ecef684d5` |
| DSMIL | VOC-Patch-Seg-ML | 3 | 3 | 0.872 | 36.6 | 34.1 | +2.5 | 0.471 | 4.2e-02 | 6 | `91614b2d6049` |
| ML-Decoder | VOC-Patch-Seg-ML | 3 | 3 | 0.865 | 79.2 | 34.1 | +45.1 | 0.753 | 4.7e-01 | 6 | `dcb2d10a1e79` |
| Q2L | VOC-Patch-Seg-ML | 3 | 3 | 0.894 | 76.9 | 34.1 | +42.9 | 0.720 | 4.4e-01 | 10 | `c41903dbb262` |
| HAP | VOC-Patch-Seg-ML | 3 | 3 | 0.856 | 61.5 | 34.1 | +27.4 | 0.718 | 3.0e-01 | 12 | `181729f5a3f9` |

Total trained models scored: **777** (315 label-agnostic on the three partitioned benchmarks).

## 4. Theorem check, model by model

Corollary 1 predicts that every label-agnostic model on a partitioned benchmark scores exactly at its fold's chance rate. The table gives, per cell, the largest deviation over all trained models; the theorem is confirmed if every entry is at floating-point noise.

| Architecture | Letter Frost | Letter Carroll | MSRC v2 |
|---|---|---|---|
| AB-MIL | 5.55e-17 | 5.55e-17 | 0.00e+00 |
| Gated AB-MIL | 5.55e-17 | 5.55e-17 | 0.00e+00 |
| TransMIL | 5.55e-17 | 5.55e-17 | 0.00e+00 |

## 5. Benchmark files

Data directory: `/Users/umaidahmed/Desktop/Research/Research_Writer/ML_Papers/MIML_CLAM_Enhanced/../VIT_MIL/Code/MIML-SELFVIT`

| File | bytes | sha256 |
|---|---|---|
| `MSCV2.mat` | 377,938 | `e50e5a15ddef` |
| `letter_frost.mat` | 6,774 | `767818da0dd7` |
| `letter_carroll.mat` | 8,329 | `a29b0eed4e38` |
| `voc2012seg.mat` | 298,236,024 | `140ed4f30ffc` |
| `voc2012seg_10CV.mat` | 4,744 | `52e822593175` |
| `voc2012seg_patch_gt.npz` | 27,549 | `0f04a518e63a` |
| `voc2012segml.mat` | 86,597,520 | `ebf4b63c43b6` |
| `voc2012segml_10CV.mat` | 1,512 | `ba764076b6dc` |
| `voc2012segml_patch_gt.npz` | 11,711 | `1dcbebc14a72` |
| `voc2012_patch_gt.npz` | 151,497 | `3bbc512ddd0b` |

Patch-label overlap threshold stored in `voc2012_patch_gt.npz`: **0.05**.

## 6. Configuration uniformity

All 35 cells were compared on 18 training and architecture settings against the reference configuration of their column (Table 6 of the manuscript).

- `batch_size` = 8 for AB-MIL on VOC-Patch-Seg (reference 4)

## 7. Attention read-out audit

- TransMIL parameters that receive no gradient from the bag loss: none
- TransMIL returned attention: shape (2, 3, 5), rows sum to 1.000000, padded entries zero: True, identical across labels: True
- tensor scored per architecture (`used_eff_attn` flag in the result files): AB-MIL=attn_weights, DSMIL=attn_weights, Gated AB-MIL=attn_weights, HAP=eff_attn, ML-Decoder=attn_weights, Q2L=attn_weights, TransMIL=attn_weights

## 8. How to regenerate this report and the manuscript

```bash
python3 analysis/make_attribution_tables.py --outdir paper_attribution/tables
python3 analysis/make_attribution_figures.py
python3 analysis/make_overview_figure.py
python3 analysis/attribution_stats.py --out paper_attribution/tables/attribution_stats.json
python3 analysis/make_setup_tables.py --outdir paper_attribution/tables
python3 analysis/verify_paper_claims.py
python3 analysis/make_verification_report.py
```


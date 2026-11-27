# Verification report: release v1.1

Generated 2026-11-27 12:45 by `python3 analysis/verify_paper_claims_r1.py` (145 checks).
The reported numbers are re-derived from `results/instance_level_r1/` (80 cells, 2,480 trained models, all CPU), `results/instance_level_r1_coco/` (COCO-Patch-ML: 8 cells, 80 models, CPU), the generated analysis files in `paper_attribution/tables/` and the VOC-Patch and COCO-Patch-ML ground truth; the run exits non-zero if any check fails. All pairwise tests are two-sided (fold-level Nadeau-Bengio corrected resampled t-test, Holm per benchmark).

```
Verifying the revised manuscript against results/instance_level_r1 

  PASS  [setup       ] classical cells complete: 8 archs x 4 x 35 models              1120
  PASS  [analysis 9.1] 315 agnostic models: P@1=chance, pooled AUC=0.5 (<6e-17)       n=315 worst=5.55e-17
  PASS  [analysis 9.1] agnostic chance 26.1, 25.4, 34.2                               
  PASS  [analysis 9.1] agnostic macro AUC within 0.498..0.506                         0.498..0.506
  PASS  [T3          ] Gamma_AUC: 0.500 x3, Bird Song 0.843                           {'Letter Frost': 0.5, 'Letter Carroll': 0.5, 'MSRC~v2': 0.5, 'Bird Song': 0.843, 'VOC-Patch-Seg': 0.956, 'VOC-Patch-Seg-ML': 0.891}
  PASS  [analysis add.] AB-MIL additive +29.3 +31.9 +26.4                              [29.3, 31.9, 26.4]
  PASS  [analysis add.] Gated additive +33.7 +34.2 +28.6                               [33.7, 34.2, 28.6]
  PASS  [analysis add.] additive pooled AUC 0.747..0.785                               0.747..0.785
  PASS  [analysis add.] 210 additive models all >= +17 pts                             min 17.3
  PASS  [analysis add.] decomposition reproduces logits to <2e-5                       1.53e-05
  PASS  [analysis add.] ML-Decoder +29.0 +33.5 +37.5                                   
  PASS  [analysis add.] Bird Song attention +13.7/+15.0, additive +25.7/+31.8          
  PASS  [abstract    ] additive "26 to 34" points                                     26.4..34.2
  PASS  [analysis del.] agnostic: delta_l == delta_l' on every classical benchmark     1.3e-09
  PASS  [analysis del.] ML-Dec delta_l 1.50..2.05, delta_l' 0.13..0.47 (partitioned)   ([1.4961134762076274, 1.8029660502679878, 2.053804448549571], [0.13037550493623984, 0.16880043733898756, 0.469352021575887])
  PASS  [analysis del.] DSMIL-sep MSRC: other 0.94 > self 0.84                         
  PASS  [analysis del.] delta_l > delta_rnd everywhere; >=2x except DSMIL-sep MSRC     []
  PASS  [analysis nec.] DSMIL +12.8 +16.6 +23.4 +26.0                                  
  PASS  [analysis nec.] DSMIL-sep MSRC P@1 30.1, chance 34.2, pooled 0.460, 34/35 AUC<0.5 
  PASS  [analysis nec.] AB-MIL > DSMIL-sep MSRC: 4.0 in the mean, NOT significant two-sided (p 0.091), delta 0.74, 5/5 
  PASS  [analysis nec.] DSMIL vs DSMIL-sep MSRC: 27.4, p 6.9e-4, delta 0.998           
  PASS  [analysis nec.] DSMIL pair: 3.1 pts AP (0.773 vs 0.743)                        
  PASS  [analysis nec.] pooled AUC ML-Dec 0.735..0.808, Q2L 0.678..0.710               
  PASS  [analysis nec.] ML-Dec > Q2L +5.4 +9.0 +12.4 (p<=0.018) and +12.4 BS (p 0.29, 5/5) 
  PASS  [analysis nec.] ML-Dec > Q2L delta 0.57..0.94 on partitioned                   
  PASS  [analysis nec.] ML-Dec vs TransMIL Cliff delta = 1.00 on all four              
  PASS  [analysis bag] fold-sig label-conditional pairs: attribution 26/40, AP 5/40   (26, 5)
  PASS  [analysis bag] no pair separated by both in opposite directions               0
  PASS  [analysis bag] AP spans 6.3 7.4 6.3 2.2; P@1 spans 27.1 30.2 41.6 23.9        ([6.3, 7.4, 6.3, 2.2], [27.1, 30.2, 41.6, 23.9])
  PASS  [analysis bag] Bird Song AP 0.931..0.936 for four of five                     [0.914, 0.931, 0.932, 0.933, 0.936]
  PASS  [analysis bag] mscv2: Q2L top AP, ML-Dec second, ML-Dec +12.4 attribution     ['q2l', 'ml_decoder']
  PASS  [analysis bag] hja_birdsong: Q2L top AP, ML-Dec second, ML-Dec +12.4 attribution ['q2l', 'ml_decoder']
  PASS  [analysis bag] MSRC: HAP AP 0.784 > DSMIL 0.773, attribution 15.3 lower       
  PASS  [analysis bag] within-arch: 23 cells, 9 sig, rho -0.49..+0.88                 
  PASS  [analysis bag] DSMIL-sep MSRC within-arch rho -0.49                           
  PASS  [analysis 9.1] Bird Song chance 24.5, dataset ceiling 48.4 (T3)               
  PASS  [analysis 9.1] Bird Song agnostic +13.7 +15.0 +14.6, pooled 0.638..0.672      
  PASS  [analysis 9.1] Bird Song agnostic models never above ceiling                  
  PASS  [analysis 9.1] Bird Song agnostic misses on silence 28..29%                   [29.5, 28.0, 27.8]
  PASS  [analysis 9.1] multi-species: agnostic 31.0..32.5, chance 18.1, ceiling 38.5  
  PASS  [analysis 9.1] 63% of recordings have >=2 scorable species                    63.3
  PASS  [analysis 9.1] multi-species ML-Dec 63.4, DSMIL 45.0                          
  PASS  [analysis HAP] soft-only +4.6 +5.4 +11.6; HAP +1.9 +3.3 +8.1                  
  PASS  [analysis HAP] top-k only: all 105 models within 0.8 pts of chance            0.75
  PASS  [analysis HAP] k=5 > k=3 > k=1 on every partitioned benchmark                 
  PASS  [analysis HAP] alpha 0.51 +- 0.03, range 0.42..0.65, n=7875                   
  PASS  [analysis HAP] Bird Song: soft +18.3, HAP +16.5, top-k +8.9 (lowest), alpha 0.53 
  PASS  [analysis HAP] AP HAP vs soft: 0.705/0.708, 0.784/0.785, 0.932/0.935          
  PASS  [analysis lab.] rho -0.06..+0.81; 10/15 significant, all positive              
  PASS  [analysis lab.] Bird Song: no label-conditional cell significant               
  PASS  [response R3.3] 90 per-model significant, 63 fold-level (classical)            (90, 63)
  PASS  [appendix    ] ensemble lift positive in all 32 classical cells, 1.2..7.9     
  PASS  [appendix    ] AB-MIL lift up to 7.9; ML-Decoder at most 3.3                  
  PASS  [analysis voc] grid: 8 archs x 10 folds x 1 model (as stated)                 
  PASS  [analysis voc] grid: chance 34.0 -> 25.6; fold ceiling 84.0 / 82.8            
  PASS  [analysis voc] grid margins (ML-Dec, Q2L, DSMIL, AB-MIL, TransMIL)            {'ml_decoder': [45.3, 44.2], 'q2l': [44.5, 43.0], 'dsmil_orig': [35.0, 33.8], 'attnmil': [17.2, 13.4], 'transmil_lite': [24.1, 15.6]}
  PASS  [analysis voc] grid: class gap widens (min cond - max agnostic)               
  PASS  [analysis voc] Seg-ML complete: 8 archs x 30 models, 10 folds                 
  PASS  [analysis voc] Seg-ML agnostic +17.2 +18.1 +24.1; ML-Dec +45.3, Q2L +44.5     
  PASS  [analysis voc] Seg-ML agnostic misses on other objects 74..81%                [74.1, 75.7, 81.0]
  PASS  [analysis voc] Seg-ML additive +41.9 / +41.5; DSMIL +35.0                     
  PASS  [analysis del.] Seg-ML deletion: ML-Dec 0.72/-0.01, DSMIL 0.56/0.05, DSMIL-sep 0.13/0.13 
  PASS  [analysis del.] Seg-ML agnostic delta_l == delta_l'                            
  PASS  [analysis nec.] Seg-ML DSMIL vs sep: 32.2, p 2.4e-8, delta 1.00, 10/10 folds   
  PASS  [analysis nec.] Seg-ML: DSMIL-sep +2.8, pooled 0.486, AP 0.892 second, +0.5 over DSMIL 
  PASS  [analysis voc] Seg-ML chance 34.0, dataset ceiling 84.0 (T3)                  
  PASS  [analysis voc] Seg complete: 8 archs x 30 models, 10 folds                    
  PASS  [analysis voc] Seg chance 40.4, ceiling 93.2 (fold means and T3)              
  PASS  [analysis voc] Seg agnostic +32.6 +30.4 +35.3; ML-Dec +47.9, Q2L +46.6        [32.6, 30.4, 35.3, 47.9, 46.6]
  PASS  [analysis voc] Seg multi-label bags: agnostic 55.1..59.5, chance 33.5, ceil 84.7, ML-Dec 81.0, Q2L 79.2 [58.0, 55.1, 59.5]
  PASS  [analysis voc] tau: pairs 294/255/204, chance 40.4/31.6/25.6, ceiling 93.2/89.5/87.2 
  PASS  [analysis voc] tau: AB-MIL +32.6 -> +14.2, ML-Dec +47.9 -> +35.2; every margin shrinks 
  PASS  [analysis voc] tau: agnostic margins shrink most (AB-MIL, gated > every label-query drop) 
  PASS  [analysis voc] tau: only DSMIL/HAP change places; within 1 pt; not sig at 0.05 [('dsmil_orig', 'hap_only')]
  PASS  [analysis voc] tau: ML-Dec, Q2L top two and DSMIL-sep last at every tau       
  PASS  [analysis add.] Seg additive +49.3 / +49.2, above every label-conditional attention [49.3, 49.2]
  PASS  [analysis del.] Seg deletion: ML-Dec 0.93/0.01, DSMIL 0.87/0.03, DSMIL-sep 0.11 self < 0.19 other [0.93, 0.01, 0.87, 0.03, 0.11, 0.19]
  PASS  [analysis del.] Seg agnostic delta_l == delta_l'                               
  PASS  [analysis del.] delta_l > delta_rnd in all 48 cells; >=2x except DSMIL-sep MSRC, Seg-ML, Seg (1.2..1.9) ([('dsmil', 'mscv2', 'factor<2'), ('dsmil', 'voc2012segml_nc', 'factor<2'), ('dsmil', 'voc2012seg_nc', 'factor<2')], [1.1929435832439033, 1.9171530160036865, 1.5725988550967578])
  PASS  [analysis nec.] Seg DSMIL vs sep: 28.8, p 5.0e-8, delta 1.00, 10/10 folds      
  PASS  [analysis nec.] Seg DSMIL-sep +12.1 weakest; AP 0.926 highest, = ML-Dec to 3 dp 
  PASS  [analysis bag] six benchmarks: attribution 43/60, AP 6/60, AP 0 on Seg-ML     (43, 6, 0)
  PASS  [analysis bag] exactly one opposite pair: Seg DSMIL-sep vs Q2L, AP +1.0 p .034 / P@1 34.5 p 1.1e-8 [('voc2012seg_nc', 'dsmil', 'q2l')]
  PASS  [analysis bag] VOC spans: Seg-ML AP 2.4 / P@1 42.5; Seg AP 1.0 / P@1 35.9     [2.4, 42.5, 1.0, 35.9]
  PASS  [setup       ] models: 1,120 classical, 480 VOC, 1,600 main, 2,480 total      (1120, 480, 1600, 2480)
  PASS  [setup       ] one platform: every r1 cell trained on CPU                     []
  PASS  [setup       ] compute: 48.4 h summed (16.0 classical, 32.4 VOC)              [48.44630278540982, 16.021961557269098]
  PASS  [setup       ] VOC per-model 0.7..9.1 min                                     [0.6817472700277964, 9.073870103359221]
  PASS  [setup       ] CUDA check: 6 cells; 5 within 1.6 P@1 / 0.001 AP; DSMIL-sep 5.1, weakest on both {'ml_decoder': (-0.057360797883898584, -0.0003333877874150959), 'dsmil_orig': (1.5317698452278372, 0.0006651097869508904), 'dsmil': (-5.056023115130492, -0.004645070900312742), 'gated_attn': (-0.22762810746886908, -9.402772839783857e-06), 'attnmil': (-0.4527916463207271, -0.00023369612241919135), 'hap_only': (0.18345986713544926, -0.0003270462879755298)}
  PASS  [analysis nec.] DSMIL-sep MSRC: pooled AUC 0.460, 31/35 models below 0.5; DSMIL 0.677, 0/35 
  PASS  [analysis 9.1] label-agnostic bag AP differs by up to 3.5 pts on partitioned data 3.5313798395607554
  PASS  [response R2.2] per-model F1 spans: Macro 2.6..8.9, Micro 2.3..4.9             ([3.5225835453440157, 4.208307179738313, 8.862227841237335, 6.353132745107248, 2.624904904480807, 7.015595220544957], [2.9463357997317416, 4.370551600324724, 4.916460733980221, 4.424499581119202, 2.3413517931381134, 2.998723082377275])
  PASS  [analysis 9.8] top bag-AP method is not the top attributor on 4 of 6 benchmarks 4
  PASS  [setup 7.1   ] never-tested remainder n mod k: 4,1,1,3,3,0 (bag counts and folds from dataset_stats) [(144, 5), (166, 5), (591, 5), (548, 5), (2273, 10), (660, 10)]
  PASS  [theory remark] Gamma_AUC attained by the coverage ranking; lower bound = 1 - Gamma_AUC (VOC x2) [(0.9557, np.float64(0.9557), 1.0), (0.8908, np.float64(0.8908), 1.0)]
  PASS  [theory 4.2  ] VOC-Seg multi-label-disjoint bags: 95, chance 22.58, ceiling 47.74, +25.2 (95, 22.581658291457288, 47.73869346733669)
  PASS  [appendix C  ] 28 comparisons per benchmark; significant ones keep sign in 5/5 (classical), >=9/10 (VOC) {'letter_frost': [5], 'letter_carroll': [5], 'mscv2': [5], 'hja_birdsong': [5], 'voc2012seg_nc': [9, 10], 'voc2012segml_nc': [10]}
  PASS  [analysis voc] crop: AP +1.7..+2.9 for every arch; margins -0.5..+3.0         ([2.8952522671341074, 2.1291247674614766, 2.2306283717386077, 2.4416280338194984, 2.127659178713426, 2.1304259195049746, 1.9301454183332178, 1.729913914107839], [1.5093431146814833, 2.2598896923528162, 0.943803553988598, 2.8713575070373523, -0.4612054832384249, -0.11762745786804629, 1.4551145839164974, 3.0097542096507723])
  PASS  [analysis voc] crop: DSMIL-sep +3.2 (crop) / +2.8 (no crop)                   
  PASS  [analysis voc] crop: no significant comparison changes direction              0 mean-order swaps, 0 among fold-significant
  PASS  [analysis voc] crop: ML-Dec and Q2L strongest attributors on both             
  PASS  [setup       ] >= 2 other archs beat HAP per-model AP on every benchmark      {'letter_frost': 4, 'letter_carroll': 4, 'mscv2': 2, 'hja_birdsong': 2, 'voc2012segml_nc': 5, 'voc2012seg_nc': 2}
  PASS  [setup stats ] all tests two-sided (no one-sided alternative in attribution_stats.py) 
  PASS  [analysis HAP] alpha~0: top-k set label-identical in 99.4..99.8% of bags > k; alpha = sigma(-10) [99.78021978021977, 99.53399541634835, 99.4465106257172]
  PASS  [analysis del.] per-pair selectivity: agnostic 0.43..0.49; ML-Dec 0.66..0.77; Q2L 0.61..0.74; DSMIL 0.61..0.76 
  PASS  [analysis del.] HAP selectivity 0.54..0.64 on MSRC, BS, VOC; ~0.5 on Letters; DSMIL-sep MSRC 0.43 {'letter_frost': 0.4798847296511245, 'letter_carroll': 0.48086771031736736, 'mscv2': 0.5467851502829577, 'hja_birdsong': 0.5369901763179868, 'voc2012seg_nc': 0.643646687569765, 'voc2012segml_nc': 0.5417130255769678}
  PASS  [analysis 9.1] multi-species: Q2L 49.4 above 38.5; DSMIL-sep 34.0 and HAP 34.4 below {'dsmil_orig': 45.0, 'dsmil': 34.0, 'ml_decoder': 63.4, 'q2l': 49.4, 'hap_only': 34.4}
  PASS  [analysis voc] VOC-Seg agnostic misses: 36..46% background, 54..64% other object [42.20459902418067, 45.69372995370931, 36.030094819737116]
  PASS  [analysis voc] tau shrink: agnostic 17..18 points, ML-Dec/Q2L 12..13          {'attnmil': 0.18426465887945603, 'gated_attn': 0.1763995329616419, 'transmil_lite': 0.17049882182869575, 'dsmil_orig': 0.1550820292369084, 'dsmil': 0.07076434374473367, 'ml_decoder': 0.12701147324388695, 'q2l': 0.11657723400956355, 'hap_only': 0.17252346742458446}
  PASS  [analysis add.] additive: level with best label-cond. on LF, LC, VOC-Seg; behind on MSRC (9..11), BS, Seg-ML ({'letter_frost': 28.96358786536734, 'letter_carroll': 33.497234682568575, 'mscv2': 37.54253158867064, 'hja_birdsong': 40.316023813348345, 'voc2012seg_nc': 47.93222946219819, 'voc2012segml_nc': 45.267389645361526}, {'letter_frost': 33.69971556100702, 'letter_carroll': 34.244436070472474, 'mscv2': 28.645351007367896, 'hja_birdsong': 31.75671474180146, 'voc2012seg_nc': 49.26951602254122, 'voc2012segml_nc': 41.91859920482715})
  PASS  [discussion  ] additive 26..49 points over chance; up to 15 behind (conclusion); ML-Dec BS 40.3 
  PASS  [limitations ] box-derived labels: 2,913 images, recall 95.7, precision 74.4 at tau 0.05 {'images': 2913, 'threshold': 0.05, 'recall': 0.9567334919016877, 'precision': 0.7437485325193708}
  PASS  [analysis bag] secondary per-model test: attribution 54/60, bag AP 35/60      (54, 35)
  PASS  [imbalance   ] 1,600 models; every head-tail NB p recomputed from fold means  []
  PASS  [imbalance   ] MeanIR 3.1 (BS) .. 15.8 (Seg-ML); 2..7 labels < 5%; VOC-Seg min count 89; classical tail freq 0.5..10.6% 
  PASS  [imbalance   ] AP tail lower 48/48, significant 33 (all 32 classical, 0 VOC-Seg); 17.6..51.3 classical; VOC-Seg 1.7..2.5 
  PASS  [imbalance   ] F1 tail significantly lower 41/48; Micro > Macro in 48/48 by 3.1..19.0 
  PASS  [imbalance   ] attribution (label-cond.): tail lower 26/30, significant 5 (DSMIL/ML-Dec/Q2L MSRC, Q2L LC, DSMIL BS), 1 reversed ['dsmil_orig__hja_birdsong', 'dsmil_orig__mscv2', 'ml_decoder__mscv2', 'q2l__letter_carroll', 'q2l__mscv2']
  PASS  [imbalance   ] MSRC: no label-conditional tail margin above +3 (-4.7..+2.8); ML-Dec head +44.4 [2.8374971028356284, -3.036420646082121, 0.5891204794590047, -1.462518357894118, -4.685307473540377]
  PASS  [imbalance   ] prevalence floor head-tail 18.5..22.9 (classical); raw AP significant 1/16 on VOC [18.5, 18.6, 22.5, 22.9]
  PASS  [imbalance   ] AP-pi: classical tail lower 31/32, significant 7 all Letter; VOC tail higher 16/16, 3 significant ['attnmil__letter_carroll', 'attnmil__letter_frost', 'dsmil__letter_carroll', 'dsmil_orig__letter_carroll', 'gated_attn__letter_frost', 'ml_decoder__letter_carroll', 'q2l__letter_carroll']
  PASS  [imbalance   ] (AP-pi)/(1-pi): tail lower 48/48, significant 26/32 classical (all on LF, LC, MSRC), 0 VOC 
  PASS  [imbalance   ] Micro-Macro over labels with a positive: 3.1..15.3; F1 tail lower 48/48 
  PASS  [imbalance   ] attribution reversals: DSMIL-sep (head margin +3.4) and Gated AB-MIL on VOC-Seg ['dsmil__voc2012seg_nc', 'gated_attn__voc2012seg_nc']
  PASS  [coco        ] 8 cells x 10 models (folds 0..9), batch 8, VOC widths, CPU, no NaN []
  PASS  [coco        ] Theorem 1: every label-agnostic model P@1 <= its ceiling       []
  PASS  [coco        ] 3,766 bags, 80 labels, 64 patches, 3.3 labels/bag (2..13), 12.1 witnesses, all pairs witnessed 
  PASS  [coco        ] disjoint witness sets 13.6%; chance 18.5, ceiling 65.2 (fold avg); Gamma_AUC 0.877 
  PASS  [coco        ] label-agnostic P@1 30.7..32.9, margins +12.3..+14.4, > 30 below ceiling 
  PASS  [coco        ] ML-Dec 57.6 (+39.1), Q2L 57.2 (+38.8), DSMIL 43.4, HAP 41.2, DSMIL-sep 20.0 (+1.5) 
  PASS  [coco        ] each of DSMIL/ML-Dec/Q2L/HAP > each label-agnostic: 10/10 folds, sig., 8.3..26.9 
  PASS  [coco        ] DSMIL-sep significantly below every label-agnostic architecture 
  PASS  [coco        ] Holm over 28 pairs                                             
  PASS  [coco        ] deletion per-pair: label-agnostic 0.47..0.48; ML-Dec 0.61, Q2L 0.63, DSMIL 0.71; ML-Dec 0.47 vs 0.03 
  PASS  [coco        ] additive P@1 AB-MIL 55.7, gated 53.7, within 4 points of ML-Dec and Q2L 
  PASS  [coco        ] bag AP 0.648..0.701, 12/28 significant, top three ML-Dec/Q2L/HAP; P@1 20.0..57.6 
  PASS  [coco        ] pooled AUC: agnostic .553..640; ML-Dec/Q2L/HAP .635..679, none sig.; DSMIL .507, DSMIL-sep .420 lowest 
  PASS  [coco        ] AUC per fold: AB-MIL, gated 7 folds 0.48..0.57, 3 folds 0.68..0.72; TransMIL 0.61..0.66; diff to ML-Dec 0.10..0.13 
  PASS  [coco        ] AUC: DSMIL, DSMIL-sep sig. below TransMIL (0.132, 0.219), ML-Dec, Q2L, HAP 
  PASS  [coco        ] 12,466 present pairs, 70 all-witness (not scorable); deletion HAP 0.51, DSMIL-sep 0.53 
  PASS  [coco        ] ML-Dec, Q2L margins 4..5 below VOC 8x8 (+44.2, +43.0); AB-MIL +13.4 there 
  PASS  [coco        ] models: 2,480 + 80 COCO = 2,560                                
  PASS  [coco        ] patch labels = overlap >= 0.05 (float16 store); bag labels = OR of patch labels; >= 2 labels per bag 
  PASS  [benchmarks  ] voc2012seg: no-crop file keeps labels and patch GT             
  PASS  [benchmarks  ] voc2012segml: no-crop file keeps labels and patch GT           

all checks passed
```

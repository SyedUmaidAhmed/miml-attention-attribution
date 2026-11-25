# Verification report: release v1.1

Generated 2026-11-25 11:06 by `python3 analysis/verify_paper_claims_r1.py` (104 checks).
The reported numbers are re-derived from `results/instance_level_r1/` (80 cells, 2,480 trained models, all CPU), the generated analysis files in `paper_attribution/tables/` and the VOC-Patch ground truth; the run exits non-zero if any check fails.

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
  PASS  [analysis nec.] AB-MIL > DSMIL-sep MSRC: 4.0, fold p 0.046, delta 0.74         
  PASS  [analysis nec.] DSMIL vs DSMIL-sep MSRC: 27.4, p 3.4e-4, delta 0.998           
  PASS  [analysis nec.] DSMIL pair: 3.1 pts AP (0.773 vs 0.743)                        
  PASS  [analysis nec.] pooled AUC ML-Dec 0.735..0.808, Q2L 0.678..0.710               
  PASS  [analysis nec.] ML-Dec > Q2L +5.4 +9.0 +12.4 (p<=0.009) and +12.4 BS (p 0.15, 5/5) 
  PASS  [analysis nec.] ML-Dec > Q2L delta 0.57..0.94 on partitioned                   
  PASS  [analysis nec.] ML-Dec vs TransMIL Cliff delta = 1.00 on all four              
  PASS  [analysis bag] fold-sig label-conditional pairs: attribution 28/40, AP 9/40   (28, 9)
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
  PASS  [response R3.3] 90 per-model significant, 74 fold-level (classical)            (90, 74)
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
  PASS  [analysis nec.] Seg-ML DSMIL vs sep: 32.2, p 1.2e-8, delta 1.00, 10/10 folds   
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
  PASS  [analysis nec.] Seg DSMIL vs sep: 28.8, p 2.5e-8, delta 1.00, 10/10 folds      
  PASS  [analysis nec.] Seg DSMIL-sep +12.1 weakest; AP 0.926 highest, = ML-Dec to 3 dp 
  PASS  [analysis bag] six benchmarks: attribution 45/60, AP 11/60, AP 0 on Seg-ML    (45, 11, 0)
  PASS  [analysis bag] exactly one opposite pair: Seg DSMIL-sep vs Q2L, AP +1.0 p .017 / P@1 34.5 p 5.6e-9 [('voc2012seg_nc', 'dsmil', 'q2l')]
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
  PASS  [appendix C  ] 28 comparisons per benchmark; significant ones keep sign in 5/5 (classical), >=9/10 (VOC) {'letter_frost': [5], 'letter_carroll': [5], 'mscv2': [5], 'hja_birdsong': [5], 'voc2012seg_nc': [9, 10], 'voc2012segml_nc': [9, 10]}
  PASS  [analysis voc] crop: AP +1.7..+2.9 for every arch; margins -0.5..+3.0         ([2.8952522671341074, 2.1291247674614766, 2.2306283717386077, 2.4416280338194984, 2.127659178713426, 2.1304259195049746, 1.9301454183332178, 1.729913914107839], [1.5093431146814833, 2.2598896923528162, 0.943803553988598, 2.8713575070373523, -0.4612054832384249, -0.11762745786804629, 1.4551145839164974, 3.0097542096507723])
  PASS  [analysis voc] crop: DSMIL-sep +3.2 (crop) / +2.8 (no crop)                   
  PASS  [analysis voc] crop: no significant comparison changes direction              0 mean-order swaps, 0 among fold-significant
  PASS  [analysis voc] crop: ML-Dec and Q2L strongest attributors on both             
  PASS  [setup       ] >= 2 other archs beat HAP per-model AP on every benchmark      {'letter_frost': 4, 'letter_carroll': 4, 'mscv2': 2, 'hja_birdsong': 2, 'voc2012segml_nc': 5, 'voc2012seg_nc': 2}
  PASS  [benchmarks  ] voc2012seg: no-crop file keeps labels and patch GT             
  PASS  [benchmarks  ] voc2012segml: no-crop file keeps labels and patch GT           

all checks passed
```

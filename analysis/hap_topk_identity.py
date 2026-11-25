"""How often HAP's top-k selection is the same for every label (DMKD R1 panel review).

For the HAP ablation row alpha_l ~ 0 (beta_l fixed at -10, alpha_l = sigmoid(-10)
~ 4.5e-5), the pooled distribution is almost exactly uniform over the k = 3
instances the scorer selects. This script measures, from the stored test-fold
attention arrays, the share of bags with more than k valid instances (where the
selection is not trivially the whole bag) in which the selected top-k set is
identical for all labels, as a mean over the models of each cell. It also
records the stored alpha value.

    /opt/anaconda3/bin/python3 analysis/hap_topk_identity.py
writes paper_attribution/tables/hap_topk_identity.json
"""
import glob
import json
import os

import numpy as np

ARR = 'results/instance_level_r1/arrays'
OUT = 'paper_attribution/tables/hap_topk_identity.json'
K = 3
DATASETS = ['letter_frost', 'letter_carroll', 'mscv2', 'hja_birdsong']


def share_identical(attn, valid, k=K):
    same, n = 0, 0
    for b in range(attn.shape[0]):
        idx = np.nonzero(valid[b])[0]
        if len(idx) <= k:
            continue
        sets = {tuple(sorted(idx[np.argsort(-attn[b, l, idx], kind='stable')[:k]]))
                for l in range(attn.shape[1])}
        same += len(sets) == 1
        n += 1
    return same / n if n else float('nan'), n


def main():
    out = {}
    for d in DATASETS:
        vals, nb, alphas = [], [], []
        for f in sorted(glob.glob(os.path.join(ARR, f'hap_hard_only__{d}__*.npz'))):
            z = np.load(f)
            s, n = share_identical(z['attn'], ~z['mask'])
            vals.append(s)
            nb.append(n)
            if 'hap_alpha' in z:
                alphas.append(float(np.asarray(z['hap_alpha']).max()))
        if vals:
            out[d] = {'models': len(vals), 'share_identical_mean': float(np.mean(vals)),
                      'share_identical_min': float(np.min(vals)),
                      'bags_gt_k_per_model': float(np.mean(nb)),
                      'alpha_max': max(alphas) if alphas else None}
    json.dump(out, open(OUT, 'w'), indent=1)
    for d, v in out.items():
        print(d, v)


if __name__ == '__main__':
    main()

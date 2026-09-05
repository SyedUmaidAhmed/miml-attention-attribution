"""Instance-level attribution sweep.

For each (method, dataset) cell this trains exactly the same models as the
regular sweep — same config, same fold splits, same seeds — and additionally
measures, for every trained model, how well its per-label attention localizes
the true witness instances. Bag-level and instance-level numbers therefore come
from the *same* models on the *same* splits, which is what makes the
"bag accuracy does not predict instance grounding" comparison valid.

Only datasets carrying per-instance ground truth (the DD-format files with a
`YR` matrix) can be used: MSCV2, letter_frost, letter_carroll.

Usage
-----
    # HAP + the four alpha limit cases
    python3 run_instance_attribution.py \
        --methods hap_only,hap_soft_only,hap_hard_only,hap_fixed_half \
        --datasets mscv2,letter_frost,letter_carroll --device mps

    # external baselines
    python3 run_instance_attribution.py \
        --methods attnmil,gated_attn,dsmil,transmil_lite,ml_decoder,q2l \
        --datasets mscv2,letter_frost,letter_carroll --device mps

Output: results/instance_level/<method>__<dataset>.json (idempotent; pass
--rerun to overwrite).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_all_enhanced import CONFIGS                                # noqa: E402
from run_alfapp_ablation import ROW_OVERRIDES, _patch_hap_beta      # noqa: E402
from baselines.run_baselines import (                                # noqa: E402
    MODEL_FACTORIES, BASELINE_OVERRIDES,
)
from miml_clam.training import enhanced_train_cv as _etcv           # noqa: E402
from miml_clam.training.enhanced_train_cv import run_cv             # noqa: E402
from miml_clam.metrics.instance_metrics import (                     # noqa: E402
    all_instance_metrics, check_or_consistency,
)

# Datasets carrying per-instance ground truth. The three DD-format files have it
# natively in their `YR` matrix; voc2012seg carries it in a side-car .npz built
# from VOC's pixel-exact segmentation masks.
INSTANCE_DATASETS = ['mscv2', 'letter_frost', 'letter_carroll',
                     'voc2012seg', 'voc2012segml']

# Datasets whose instance labels come from a side-car file rather than the
# dataset object. Maps dataset name -> (npz path, array key).
#
# voc2012segml is the multi-label-only subset of voc2012seg (633 of 2,273 bags).
# It is preferred for the attribution comparison on two grounds: (i) single-label
# bags cannot penalise a label-agnostic method at all, so they dilute the
# contrast the study is measuring, and (ii) it holds ~83 MB in memory rather
# than ~298 MB, which matters because run_fold() deepcopies the dataset per
# fold and the full set thrashes a memory-constrained machine.
SIDECAR_GT = {
    'voc2012seg':   ('voc2012seg_patch_gt.npz', 'patch_seg'),
    'voc2012segml': ('voc2012segml_patch_gt.npz', 'patch_seg'),
}


def _voc_seg_config(name='voc2012seg'):
    """Config for a pixel-exact VOC subset, mirroring _voc_config()."""
    from run_voc_experiment import _voc_config
    from datasets.voc_miml import DEFAULT_OUT_DIR
    c = dict(_voc_config())
    c.update({
        'data_path': os.path.join(DEFAULT_OUT_DIR, f'{name}.mat'),
        'cv_mat_path': os.path.join(DEFAULT_OUT_DIR, f'{name}_10CV.mat'),
        'dataset': name,
        # Smaller batches: the 2048-dim features plus run_fold()'s per-fold
        # deepcopy make this the peak-memory dataset in the suite.
        'batch_size': 4,
    })
    return c


def _load_sidecar_gt(dataset):
    """(n_bags, N, L) bool instance labels for a side-car dataset."""
    from datasets.voc_miml import DEFAULT_OUT_DIR
    fname, key = SIDECAR_GT[dataset]
    z = np.load(os.path.join(DEFAULT_OUT_DIR, fname), allow_pickle=True)
    return z[key].astype(bool)

INSTANCE_METRIC_KEYS = [
    'InstP@1', 'InstP@3', 'InstMRR', 'InstAUC',
    'ChanceP@1', 'LabelAgnosticCeiling',
]


def _device(force=None):
    if force is not None:
        return torch.device(force)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def evaluate_attribution(model, test_loader, dataset, test_idx, device,
                         sidecar=None):
    """Instance-level metrics for one trained model on its test fold.

    Uses `eff_attn` when the model exposes it (the true HAP mixture
    distribution) and falls back to `attn_weights` otherwise. Using
    `attn_weights` for a HAP model would silently evaluate the pre-mixture
    softmax, which is invariant to alpha.
    """
    model.eval()

    attn_batches, inst_batches, bag_batches, mask_batches = [], [], [], []
    ptr = 0
    used_eff = False

    with torch.no_grad():
        for bags, labels, mask in test_loader:
            bs = bags.shape[0]
            out = model(bags.to(device), mask=mask.to(device))
            if 'eff_attn' in out:
                attn = out['eff_attn']
                used_eff = True
            else:
                attn = out['attn_weights']
            attn = attn.detach().cpu().numpy()

            idxs = test_idx[ptr:ptr + bs]
            ptr += bs

            N = bags.shape[1]
            L = labels.shape[1]
            inst = np.zeros((bs, N, L), dtype=bool)
            for j, gi in enumerate(idxs):
                if sidecar is not None:
                    il = sidecar[gi]
                else:
                    il = dataset.get_instance_labels(gi).numpy() > 0
                inst[j, :il.shape[0]] = il

            attn_batches.append(attn)
            inst_batches.append(inst)
            bag_batches.append(labels.numpy() > 0)
            mask_batches.append(mask.numpy() > 0)

    if ptr != len(test_idx):
        raise RuntimeError(
            f'test_loader yielded {ptr} bags but test_idx has {len(test_idx)}; '
            'ordering assumption violated')

    # Batches may have different padded widths — pad all to the global max.
    N = max(a.shape[-1] for a in attn_batches)

    def pad_attn(a):
        out = np.full((a.shape[0], a.shape[1], N), -np.inf, dtype=np.float64)
        out[:, :, :a.shape[2]] = a
        return out

    def pad_inst(a):
        out = np.zeros((a.shape[0], N, a.shape[2]), dtype=bool)
        out[:, :a.shape[1], :] = a
        return out

    def pad_mask(a):
        out = np.ones((a.shape[0], N), dtype=bool)     # True = padded
        out[:, :a.shape[1]] = a
        return out

    attn = np.concatenate([pad_attn(a) for a in attn_batches], axis=0)
    inst = np.concatenate([pad_inst(a) for a in inst_batches], axis=0)
    bag = np.concatenate(bag_batches, axis=0)
    mask = np.concatenate([pad_mask(a) for a in mask_batches], axis=0)

    metrics = all_instance_metrics(attn, inst, bag, mask=mask, topk=(1, 3))
    metrics['_used_eff_attn'] = used_eff
    return metrics


def run_cell(method, dataset, folds, outdir, rerun, force_device, no_ensemble):
    out_path = os.path.join(outdir, f'{method}__{dataset}.json')
    if os.path.exists(out_path) and not rerun:
        print(f'  cached: {out_path}', flush=True)
        return out_path

    if dataset not in INSTANCE_DATASETS:
        raise ValueError(
            f'{dataset!r} has no per-instance labels; choose from '
            f'{INSTANCE_DATASETS}')

    is_baseline = method in MODEL_FACTORIES
    if not is_baseline and method not in ROW_OVERRIDES:
        raise ValueError(
            f'Unknown method {method!r}; choose a HAP row {list(ROW_OVERRIDES)} '
            f'or a baseline {list(MODEL_FACTORIES)}')

    sidecar = _load_sidecar_gt(dataset) if dataset in SIDECAR_GT else None
    if dataset in SIDECAR_GT:
        config = _voc_seg_config(dataset)
    else:
        config = deepcopy(CONFIGS[dataset])
    hap_init_beta, hap_freeze = None, False
    if is_baseline:
        config.update(BASELINE_OVERRIDES)
    else:
        config.update(ROW_OVERRIDES[method])
        hap_init_beta = config.pop('_hap_init_beta', None)
        hap_freeze = config.pop('_hap_freeze_beta', False)

    if no_ensemble:
        config.pop('ensemble_seeds', None)
    num_folds = folds if folds is not None else config.get('default_folds', 5)

    device = _device(force_device)
    print(f'\n>>> [instance-level] {method} on {dataset} (device={device}, '
          f'folds={num_folds})', flush=True)

    records = []

    def collector(fold_idx, seed, model, test_loader, ds, test_idx, dev):
        m = evaluate_attribution(model, test_loader, ds, test_idx, dev,
                                 sidecar=sidecar)
        m['fold'] = int(fold_idx)
        m['seed'] = None if seed is None else int(seed)
        records.append(m)
        print(f'    [attr] fold {fold_idx + 1} seed {seed}: '
              f"P@1={m['InstP@1']:.4f} AUC={m['InstAUC']:.4f} "
              f"(chance {m['ChanceP@1']:.4f})", flush=True)

    original_create_model = _etcv.create_model
    _etcv.ATTENTION_COLLECTOR = collector
    if is_baseline:
        factory = MODEL_FACTORIES[method]
        _etcv.create_model = lambda cfg, dev='cpu': factory(cfg, dev)
    elif hap_init_beta is not None:
        def _patched(cfg, dev='cpu'):
            m = original_create_model(cfg, dev)
            _patch_hap_beta(m, hap_init_beta, hap_freeze)
            return m
        _etcv.create_model = _patched

    t0 = time.time()
    try:
        results, all_metrics = run_cv(config, device=device,
                                      num_folds=num_folds)
    finally:
        _etcv.ATTENTION_COLLECTOR = None
        _etcv.create_model = original_create_model

    # Aggregate instance metrics over every trained model.
    agg = {}
    for k in INSTANCE_METRIC_KEYS:
        vals = [r[k] for r in records if k in r and not np.isnan(r[k])]
        if vals:
            agg[k] = [float(np.mean(vals)), float(np.std(vals))]

    payload = {
        'dataset': dataset,
        'method': method,
        'kind': 'baseline' if is_baseline else 'hap_row',
        'num_folds': num_folds,
        'ensemble_seeds': config.get('ensemble_seeds'),
        'used_eff_attn': bool(records and records[0].get('_used_eff_attn')),
        'bag_metrics_mean_std': results,
        'instance_metrics_mean_std': agg,
        'instance_per_model': records,
        'config': {k: v for k, v in config.items()
                   if isinstance(v, (int, float, str, bool, list, type(None)))},
        'wall_clock_seconds': time.time() - t0,
    }

    os.makedirs(outdir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'  saved: {out_path}  ({(time.time() - t0) / 60:.1f} min)',
          flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--methods', required=True)
    ap.add_argument('--datasets', default=','.join(INSTANCE_DATASETS))
    ap.add_argument('--folds', type=int, default=None)
    ap.add_argument('--no-ensemble', action='store_true')
    ap.add_argument('--outdir', default='results/instance_level')
    ap.add_argument('--rerun', action='store_true')
    ap.add_argument('--device', default=None,
                    choices=['cpu', 'mps', 'cuda'])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    methods = [m for m in args.methods.split(',') if m]
    datasets = [d for d in args.datasets.split(',') if d]

    t0 = time.time()
    for method in methods:
        for dataset in datasets:
            try:
                run_cell(method, dataset, args.folds, args.outdir,
                         args.rerun, args.device, args.no_ensemble)
            except Exception as exc:      # one bad cell must not kill the sweep
                err = os.path.join(args.outdir, f'{method}__{dataset}.error.txt')
                with open(err, 'w') as f:
                    f.write(f'{type(exc).__name__}: {exc}\n')
                print(f'  ERROR ({method}, {dataset}): {exc!r}', flush=True)
    print(f'\n=== instance-level sweep done in '
          f'{(time.time() - t0) / 60:.1f} min ===')


if __name__ == '__main__':
    main()

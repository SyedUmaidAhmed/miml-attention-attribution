"""Instance-level attribution sweep.

For each (method, dataset) cell this trains exactly the same models as the
regular sweep — same config, same fold splits, same seeds — and additionally
measures, for every trained model, how well its per-label attention localizes
the true witness instances. Bag-level and instance-level numbers therefore come
from the *same* models on the *same* splits, which is what makes the
"bag accuracy does not predict instance grounding" comparison valid.

Only datasets carrying per-instance ground truth (the DD-format files with a
`YR` matrix) can be used: MSCV2, letter_frost, letter_carroll, hja_birdsong
(HJA Bird Song, built by datasets/build_hja_birdsong.py), plus the VOC-Patch
subsets via side-car files.

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
from miml_clam.metrics.miml_metrics import (                         # noqa: E402
    optimize_thresholds, AllFiveWithThresholds,
)

# HAP top-k ablation rows (DMKD revision; reviewer 1 pt 6, reviewer 3 pt 4).
# Same as hap_only except k. hap_soft_only / hap_hard_only already exist in
# ROW_OVERRIDES (alpha frozen at ~1 / ~0).
ROW_OVERRIDES.setdefault('hap_k1', dict(ROW_OVERRIDES['hap_only'], hap_topk=1))
ROW_OVERRIDES.setdefault('hap_k5', dict(ROW_OVERRIDES['hap_only'], hap_topk=5))

# Label-agnostic baselines with a linear head on the pooled vector, for which
# y_l = sum_j a_j u_l^T h_j + c_l holds exactly (reviewer 1 pt 2).
ADDITIVE_METHODS = ('attnmil', 'gated_attn')

# Datasets carrying per-instance ground truth. The three DD-format files have it
# natively in their `YR` matrix; voc2012seg carries it in a side-car .npz built
# from VOC's pixel-exact segmentation masks.
INSTANCE_DATASETS = ['mscv2', 'letter_frost', 'letter_carroll',
                     'hja_birdsong', 'voc2012seg', 'voc2012segml',
                     # DMKD revision: no-crop features (4x4) and the 8x8 grid
                     'voc2012seg_nc', 'voc2012segml_nc', 'voc2012segml_g8',
                     # the submitted (centre-cropped) Seg-ML features under the
                     # revision protocol, to measure the crop effect (R1 pt 4)
                     'voc2012segml_crop',
                     # COCO-Patch-ML 8x8; built by
                     # datasets/build_coco_patch.py into COCO_PATCH_DIR
                     'coco2017val_g8']

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
    'voc2012seg_nc': ('voc2012seg_nc_patch_gt.npz', 'patch_seg'),
    'voc2012segml_nc': ('voc2012segml_nc_patch_gt.npz', 'patch_seg'),
    'voc2012segml_g8': ('voc2012segml_g8_patch_gt.npz', 'patch_seg'),
    'voc2012segml_crop': ('voc2012segml_crop_patch_gt.npz', 'patch_seg'),
    'coco2017val_g8': ('coco2017val_g8_patch_gt.npz', 'patch_seg'),
}

# Side-car datasets that live outside the VOC data directory.
# $COCO_PATCH_DIR, else ~/data/coco_patch, else the repository's data/coco_patch
COCO_PATCH_DIR = os.environ.get('COCO_PATCH_DIR') or next(
    (d for d in (os.path.expanduser('~/data/coco_patch'), 'data/coco_patch')
     if os.path.isdir(d)), os.path.expanduser('~/data/coco_patch'))
SIDECAR_DIR = {'coco2017val_g8': COCO_PATCH_DIR}

# DMKD revision VOC protocol: one batch size for every method (v1 used 4, with
# AB-MIL on VOC-Patch-Seg at 8). Applies only to the revision datasets.
R1_VOC_BATCH = 8


def _voc_seg_config(name='voc2012seg'):
    """Config for a pixel-exact VOC subset, mirroring _voc_config()."""
    from run_voc_experiment import _voc_config
    from datasets.voc_miml import DEFAULT_OUT_DIR
    ddir = SIDECAR_DIR.get(name, DEFAULT_OUT_DIR)
    c = dict(_voc_config())
    c.update({
        'data_path': os.path.join(ddir, f'{name}.mat'),
        'cv_mat_path': os.path.join(ddir, f'{name}_10CV.mat'),
        'dataset': name,
        # Smaller batches: the 2048-dim features plus run_fold()'s per-fold
        # deepcopy make this the peak-memory dataset in the suite.
        'batch_size': R1_VOC_BATCH if name.endswith(('_nc', '_g8', '_crop')) else 4,
    })
    return c


def _load_sidecar_gt(dataset):
    """(n_bags, N, L) bool instance labels for a side-car dataset."""
    from datasets.voc_miml import DEFAULT_OUT_DIR
    fname, key = SIDECAR_GT[dataset]
    ddir = SIDECAR_DIR.get(dataset, DEFAULT_OUT_DIR)
    z = np.load(os.path.join(ddir, fname), allow_pickle=True)
    return z[key].astype(bool)

INSTANCE_METRIC_KEYS = [
    'InstP@1', 'InstP@3', 'InstMRR', 'InstAUC', 'InstAUCPooled',
    'ChanceP@1', 'LabelAgnosticCeiling',
    # DMKD revision additions (absent from v1 cells)
    'AddP@1', 'AddAUC', 'AddAUCPooled', 'AddIdentityMaxErr',
    'DelTopSelf', 'DelTopOther', 'DelRndSelf', 'DelRndOther',
    'DelTopSelective', 'DelRndSelective', 'DelPairs',
    'BagAveragePrecision', 'BagHammingLoss', 'BagOneError', 'BagCoverage',
    'BagRankingLoss', 'BagMacroF1', 'BagMicroF1',
]


def additive_contributions(model, bags, mask):
    """(B, L, N) per-label instance contributions a_j * u_l^T h_j.

    Exact decomposition of the logit for AB-MIL / gated AB-MIL in eval mode:
    logits[:, l] = contrib[:, l, :].sum(-1) + c_l. Returns (contrib, logits_rebuilt).
    """
    h = model.instance_embedding(bags)                       # (B, N, D)
    out = model(bags, mask=mask)
    a = out['attn_weights'][:, 0, :]                         # (B, N), shared
    W = torch.cat([c.weight for c in model.classifiers], 0)  # (L, D)
    bvec = torch.cat([c.bias for c in model.classifiers], 0) # (L,)
    proj = torch.einsum('bnd,ld->bln', h, W)                 # (B, L, N)
    contrib = a.unsqueeze(1) * proj
    rebuilt = contrib.sum(-1) + bvec.unsqueeze(0)
    return contrib, rebuilt, out['logits']


def deletion_test(model, bags, mask, attn, labels, rng, chunk=256):
    """Drop the top-attended instance for each present label (and, as a
    control, a random other real instance) and re-run the model.

    Reviewer 1 pt 7 / reviewer 3 pt 1. Returns arrays for post-hoc analysis:
    pair_b, pair_l, top_j, rnd_j, base (B,L) raw logits, top (K,L), rnd (K,L).
    Bags with fewer than two real instances are skipped (deleting would leave
    an empty bag).
    """
    real = ~mask.bool().cpu().numpy()
    lab = labels.cpu().numpy() > 0
    pb, pl, tj, rj = [], [], [], []
    for b in range(bags.shape[0]):
        idx = np.nonzero(real[b])[0]
        if len(idx) < 2:
            continue
        for l in np.nonzero(lab[b])[0]:
            sc = attn[b, l, idx]
            top = int(idx[int(np.argmax(sc))])
            others = idx[idx != top]
            pb.append(b); pl.append(int(l)); tj.append(top)
            rj.append(int(rng.choice(others)))
    with torch.no_grad():
        base = model(bags, mask=mask)['logits'].float().cpu().numpy()
    K = len(pb)
    L = base.shape[1]
    top_out = np.zeros((K, L), np.float32)
    rnd_out = np.zeros((K, L), np.float32)
    for which, js, dest in (('top', tj, top_out), ('rnd', rj, rnd_out)):
        for s0 in range(0, K, chunk):
            sl = slice(s0, min(K, s0 + chunk))
            bi = torch.as_tensor(pb[sl], device=bags.device)
            m2 = mask.bool()[bi].clone()
            m2[torch.arange(len(bi), device=bags.device),
               torch.as_tensor(js[sl], device=bags.device)] = True
            with torch.no_grad():
                dest[sl] = model(bags[bi], mask=m2)['logits'].float().cpu().numpy()
    return (np.asarray(pb, np.int32), np.asarray(pl, np.int32),
            np.asarray(tj, np.int32), np.asarray(rj, np.int32),
            base, top_out, rnd_out)


def _deletion_summary(pb, pl, base, top, rnd):
    """Mean logit drops: self = label l, other = mean over the other labels."""
    if len(pb) == 0:
        return {}
    K, L = top.shape
    b_rows = base[pb]                                        # (K, L)
    out = {}
    for name, arr in (('Top', top), ('Rnd', rnd)):
        drop = b_rows - arr                                  # (K, L) positive = fell
        self_d = drop[np.arange(K), pl]
        other = (drop.sum(1) - self_d) / max(L - 1, 1)
        out[f'Del{name}Self'] = float(self_d.mean())
        out[f'Del{name}Other'] = float(other.mean())
        out[f'Del{name}Selective'] = float((self_d > other).mean())
    out['DelPairs'] = int(K)
    return out


def _device(force=None):
    if force is not None:
        return torch.device(force)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def evaluate_attribution(model, test_loader, dataset, test_idx, device,
                         sidecar=None, method=None, extras=None, rng=None):
    """Instance-level metrics for one trained model on its test fold.

    Uses `eff_attn` when the model exposes it (the true HAP mixture
    distribution) and falls back to `attn_weights` otherwise. Using
    `attn_weights` for a HAP model would silently evaluate the pre-mixture
    softmax, which is invariant to alpha.
    """
    model.eval()

    attn_batches, inst_batches, bag_batches, mask_batches = [], [], [], []
    add_batches, del_batches = [], []
    add_err = 0.0
    ptr = 0
    used_eff = False
    if rng is None:
        rng = np.random.RandomState(0)

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

            if extras is not None:
                bd, md = bags.to(device), mask.to(device)
                if method in ADDITIVE_METHODS:
                    contrib, rebuilt, logits = additive_contributions(
                        model, bd, md)
                    add_err = max(add_err, float(
                        (rebuilt - logits).abs().max().item()))
                    add_batches.append(contrib.detach().cpu().numpy())
                pb, pl, tj, rj, base, top, rnd = deletion_test(
                    model, bd, md, attn, labels, rng)
                del_batches.append((pb + ptr, pl, tj, rj, base, top, rnd))

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

    if extras is not None:
        arrays = {'attn': attn.astype(np.float32), 'inst': inst,
                  'bag': bag, 'mask': mask,
                  'test_idx': np.asarray(test_idx, np.int64)}
        if add_batches:
            add = np.concatenate([pad_attn(a) for a in add_batches], axis=0)
            am = all_instance_metrics(add, inst, bag, mask=mask, topk=(1,))
            metrics['AddP@1'] = am['InstP@1']
            metrics['AddAUC'] = am['InstAUC']
            metrics['AddAUCPooled'] = am['InstAUCPooled']
            metrics['AddIdentityMaxErr'] = add_err
            arrays['additive'] = add.astype(np.float32)
        if del_batches:
            cat = lambda i: np.concatenate([d[i] for d in del_batches], 0)
            pb, pl, tj, rj = cat(0), cat(1), cat(2), cat(3)
            base, top, rnd = cat(4), cat(5), cat(6)
            metrics.update(_deletion_summary(pb, pl, base, top, rnd))
            arrays.update(del_b=pb, del_l=pl, del_top_j=tj, del_rnd_j=rj,
                          base_logits=base, del_top_logits=top,
                          del_rnd_logits=rnd)
        hap = getattr(getattr(model, 'label_attention', None), 'hap', None)
        if hap is not None and hasattr(hap, 'beta'):
            alpha = torch.sigmoid(hap.beta.detach().float()).cpu().numpy()
            arrays['hap_alpha'] = alpha
            metrics['HapAlpha'] = [float(x) for x in alpha]
        extras.update(arrays)
    return metrics


def run_cell(method, dataset, folds, outdir, rerun, force_device, no_ensemble,
             save_arrays=True, seeds=None, data_suffix=''):
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
    elif seeds is not None:
        config['ensemble_seeds'] = list(seeds)
    num_folds = folds if folds is not None else config.get('default_folds', 5)

    device = _device(force_device)
    print(f'\n>>> [instance-level] {method} on {dataset} (device={device}, '
          f'folds={num_folds})', flush=True)

    records = []
    pending = {}
    arr_dir = os.path.join(outdir, 'arrays')
    if save_arrays:
        os.makedirs(arr_dir, exist_ok=True)

    def collector(fold_idx, seed, model, test_loader, ds, test_idx, dev):
        extras = {} if save_arrays else None
        # local RNG: never touches the global streams that seed training
        rng = np.random.RandomState(1000 * int(fold_idx) + (seed or 0))
        m = evaluate_attribution(model, test_loader, ds, test_idx, dev,
                                 sidecar=sidecar, method=method,
                                 extras=extras, rng=rng)
        m['fold'] = int(fold_idx)
        m['seed'] = None if seed is None else int(seed)
        records.append(m)
        pending[(int(fold_idx), seed)] = (m, extras)
        msg = (f'    [attr] fold {fold_idx + 1} seed {seed}: '
               f"P@1={m['InstP@1']:.4f} AUC={m['InstAUC']:.4f} "
               f"pooled={m['InstAUCPooled']:.4f} "
               f"(chance {m['ChanceP@1']:.4f})")
        if 'DelTopSelf' in m:
            msg += (f" del self/other={m['DelTopSelf']:.3f}/"
                    f"{m['DelTopOther']:.3f} rnd={m['DelRndSelf']:.3f}")
        if 'AddP@1' in m:
            msg += f" add P@1={m['AddP@1']:.4f} (id err {m['AddIdentityMaxErr']:.1e})"
        print(msg, flush=True)

    def pred_collector(fold_idx, seed, tl, tp, vl, vp, source):
        m, extras = pending.pop((int(fold_idx), seed))
        thr = optimize_thresholds(vl, vp, metric='hamming')
        bm = AllFiveWithThresholds(tl, tp, thr)
        for k, v in bm.items():
            m[f'Bag{k}'] = float(v)
        m['BagHead'] = source
        print(f"    [bag]  fold {fold_idx + 1} seed {seed}: "
              f"AP={m['BagAveragePrecision']:.4f} ({source})", flush=True)
        if extras is not None:
            extras.update(test_labels=np.asarray(tl), test_probs=np.asarray(tp),
                          thresholds=np.asarray(thr))
            np.savez_compressed(
                os.path.join(arr_dir, f'{method}__{dataset}{data_suffix}'
                             f'__f{int(fold_idx)}_s{seed}.npz'), **extras)

    original_create_model = _etcv.create_model
    _etcv.ATTENTION_COLLECTOR = collector
    _etcv.PREDICTION_COLLECTOR = pred_collector
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
        _etcv.PREDICTION_COLLECTOR = None
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
        'device': str(device),
        'torch_version': torch.__version__,
        'protocol_rev': 'dmkd_r1',
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
    ap.add_argument('--no-arrays', action='store_true',
                    help='do not save per-model attention/deletion arrays')
    ap.add_argument('--seeds', default=None,
                    help='comma-separated ensemble seeds overriding the config')
    ap.add_argument('--threads', type=int, default=None,
                    help='torch.set_num_threads (for parallel CPU workers)')
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    seeds = ([int(x) for x in args.seeds.split(',')] if args.seeds else None)

    os.makedirs(args.outdir, exist_ok=True)
    methods = [m for m in args.methods.split(',') if m]
    datasets = [d for d in args.datasets.split(',') if d]

    t0 = time.time()
    for method in methods:
        for dataset in datasets:
            try:
                run_cell(method, dataset, args.folds, args.outdir,
                         args.rerun, args.device, args.no_ensemble,
                         save_arrays=not args.no_arrays, seeds=seeds)
            except Exception as exc:      # one bad cell must not kill the sweep
                err = os.path.join(args.outdir, f'{method}__{dataset}.error.txt')
                with open(err, 'w') as f:
                    f.write(f'{type(exc).__name__}: {exc}\n')
                print(f'  ERROR ({method}, {dataset}): {exc!r}', flush=True)
    print(f'\n=== instance-level sweep done in '
          f'{(time.time() - t0) / 60:.1f} min ===')


if __name__ == '__main__':
    main()

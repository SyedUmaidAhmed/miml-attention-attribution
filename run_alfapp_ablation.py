"""Headline MIML-ALFA++ ablation runner.

This script is the planned Week-2 driver. It runs three headline
configurations (BCE baseline, current ALFA-Full, MIML-ALFA++ Full) on the
five core datasets that have published baselines (scene, reuters, mscv2,
letter_frost, letter_carroll). Each run uses the dataset's default fold
count and the default 7-seed ensemble, so the resulting JSON entries are
directly comparable to the cached ALFA-Full numbers already published in
results/enhanced_all_combined.json.

Usage:
    python run_alfapp_ablation.py [--rows alfapp_full,alfa_full,bce]
                                  [--datasets scene,reuters,...]
                                  [--folds N]
                                  [--no-ensemble]

Wall-clock estimate (Apple M3, 7-seed ensemble, full CV):
    - scene/reuters: ~2 hr each
    - mscv2:         ~30 min
    - letter_frost / letter_carroll: ~10 min each

So the full 3 rows × 5 datasets ≈ 18 GPU-hours.

The runner is *idempotent* per (dataset, row): if the result JSON for that
combination already exists in results/alfapp_ablation/, it is skipped
unless --rerun is specified.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np
import torch

# Reuse the canonical config registry.
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_all_enhanced import CONFIGS  # noqa: E402
from miml_clam.training.enhanced_train_cv import run_cv  # noqa: E402
from miml_clam.training import enhanced_train_cv as _etc  # noqa: E402


ROW_OVERRIDES = {
    # Plain BCE, no Mixup, single model — anchors the bottom of the table.
    'bce': {
        'use_asymmetric_loss': False,
        'use_two_way_loss': False,
        'use_mixup': False,
        'use_psemix': False,
        'use_contrastive': False,
        'use_anchor': False,
    },
    # Current published ALFA-Full configuration.
    'alfa_full': {
        'use_asymmetric_loss': True,
        'use_two_way_loss': False,
        'use_mixup': True,
        'use_psemix': False,
        'use_contrastive': False,
        'use_anchor': False,
    },
    # MIML-ALFA++ headline configuration.
    'alfapp_full': {
        'use_asymmetric_loss': True,    # used as the base of the Two-Way loss
        'use_two_way_loss': True,
        'use_mixup': True,
        'use_psemix': True,
        'use_contrastive': True,
        'use_anchor': False,            # LAA documented as negative finding
    },
    # Single-component ablations (each adds exactly one ALFA++ piece on top
    # of current ALFA-Full).
    'asl2_only': {
        'use_asymmetric_loss': True,
        'use_two_way_loss': True,       # ASL2 replaces ASL
        'use_mixup': True,              # original random-partner Mixup
        'use_psemix': False,
        'use_contrastive': False,
        'use_anchor': False,
    },
    'psemix_only': {
        'use_asymmetric_loss': True,    # original ASL
        'use_two_way_loss': False,
        'use_mixup': True,
        'use_psemix': True,             # PseMix replaces random Mixup
        'use_contrastive': False,
        'use_anchor': False,
    },
    'supcon_only': {
        'use_asymmetric_loss': True,    # original ASL
        'use_two_way_loss': False,
        'use_mixup': True,              # original Mixup
        'use_psemix': False,
        'use_contrastive': True,        # SupCon auxiliary added
        'use_anchor': False,
    },
    # NEW NOVELTY: Hybrid Attention Pooling alone (no other ALFA++ tricks).
    'hap_only': {
        'use_asymmetric_loss': False,   # match BCE baseline harness
        'use_two_way_loss': False,
        'use_mixup': False,
        'use_psemix': False,
        'use_contrastive': False,
        'use_anchor': False,
        'use_hap': True,                # the only change vs bce
    },
    # HAP combined with original ALFA-Full enhancements:
    'alfa_hap': {
        'use_asymmetric_loss': True,
        'use_two_way_loss': False,
        'use_mixup': True,
        'use_psemix': False,
        'use_contrastive': False,
        'use_anchor': False,
        'use_hap': True,
    },
    # ===== HAP component ablation (ICDM 2026) ====================================
    # Same backbone as 'hap_only' but the per-label mixture parameter beta_l is
    # initialized and (optionally) frozen to expose the four limit cases of the
    # mixture: pure soft, pure hard, fixed-half (no learning), and learned.
    # _hap_init_beta and _hap_freeze_beta are *not* model constructor kwargs --
    # they are read by run_row() and applied post-construction by patching
    # model.label_attention.hap.beta.data + .requires_grad.
    'hap_soft_only': {
        'use_asymmetric_loss': False, 'use_two_way_loss': False,
        'use_mixup': False, 'use_psemix': False,
        'use_contrastive': False, 'use_anchor': False,
        'use_hap': True,
        '_hap_init_beta': 10.0,        # sigmoid(10) ~ 0.99995 ~ pure soft
        '_hap_freeze_beta': True,
    },
    'hap_hard_only': {
        'use_asymmetric_loss': False, 'use_two_way_loss': False,
        'use_mixup': False, 'use_psemix': False,
        'use_contrastive': False, 'use_anchor': False,
        'use_hap': True,
        '_hap_init_beta': -10.0,       # sigmoid(-10) ~ 4e-5 ~ pure hard
        '_hap_freeze_beta': True,
    },
    'hap_fixed_half': {
        'use_asymmetric_loss': False, 'use_two_way_loss': False,
        'use_mixup': False, 'use_psemix': False,
        'use_contrastive': False, 'use_anchor': False,
        'use_hap': True,
        '_hap_init_beta': 0.0,         # sigmoid(0) = 0.5 = unbiased mixture
        '_hap_freeze_beta': True,
    },
    'hap_learned': {                   # alias of 'hap_only' for symmetry
        'use_asymmetric_loss': False, 'use_two_way_loss': False,
        'use_mixup': False, 'use_psemix': False,
        'use_contrastive': False, 'use_anchor': False,
        'use_hap': True,
        '_hap_init_beta': 0.0,
        '_hap_freeze_beta': False,     # default — left learnable
    },
    # HAP-Temperature: per-label sigmoid mixture (HAP) PLUS per-label learnable
    # temperature tau_l on the soft branch. Adds another L parameters; tau_l=1
    # at init (log_tau=0) so this row exactly recovers hap_learned at the start
    # of training. ICDM 2026 extension.
    'hap_temperature': {
        'use_asymmetric_loss': False, 'use_two_way_loss': False,
        'use_mixup': False, 'use_psemix': False,
        'use_contrastive': False, 'use_anchor': False,
        'use_hap': True,
        'hap_use_temperature': True,
        '_hap_init_beta': 0.0,
        '_hap_freeze_beta': False,
    },
}


def _patch_hap_beta(model, init_beta: float, freeze: bool):
    """Apply post-construction settings to the HAP mixture parameter.

    The HAP module is at model.label_attention.hap (see
    miml_clam/models/label_conditional_attn.py and hybrid_attention_pool.py).
    We set beta_l := init_beta for every l and toggle requires_grad.
    """
    hap = getattr(getattr(model, 'label_attention', None), 'hap', None)
    if hap is None:
        raise RuntimeError(
            'HAP component-ablation row requested but model has no '
            'label_attention.hap (use_hap=True must be set in the row config).'
        )
    with torch.no_grad():
        hap.beta.data.fill_(float(init_beta))
    hap.beta.requires_grad = (not bool(freeze))


CORE_DATASETS = ['scene', 'reuters', 'mscv2', 'letter_frost', 'letter_carroll']


def _device(force: str | None = None):
    if force is not None:
        return torch.device(force)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def run_row(dataset: str, row: str, folds: int | None, no_ensemble: bool,
            outdir: str, rerun: bool, force_device: str | None = None) -> str:
    """Run a single (dataset, row) cell of the ablation matrix.

    Returns the path of the JSON result file.
    """
    out_path = os.path.join(outdir, f'{row}__{dataset}.json')
    if os.path.exists(out_path) and not rerun:
        print(f'  cached: {out_path}', flush=True)
        return out_path

    config = deepcopy(CONFIGS[dataset])
    config.update(ROW_OVERRIDES[row])
    # Pull HAP-component sentinels out of the config so they don't reach the
    # model constructor; they're applied post-construction via _patch_hap_beta.
    hap_init_beta = config.pop('_hap_init_beta', None)
    hap_freeze = config.pop('_hap_freeze_beta', False)
    # Env-var overrides for sensitivity sweeps (gpu_run_pack/50_sensitivity.sh).
    # Override the top-k size for hard pooling (only meaningful when use_hap=True).
    env_topk = os.environ.get('HAP_TOPK')
    if env_topk is not None:
        config['hap_topk'] = int(env_topk)
    # Override the beta-init for the learned-alpha mode of HAP.
    env_binit = os.environ.get('HAP_BETA_INIT')
    if env_binit is not None and config.get('use_hap'):
        hap_init_beta = float(env_binit)
        hap_freeze = False  # init only; still learnable
    if no_ensemble:
        config.pop('ensemble_seeds', None)
    if folds is not None:
        num_folds = folds
    else:
        num_folds = config.get('default_folds', 5)

    device = _device(force_device)
    print(f'\n>>> {row} on {dataset} (device={device})  '
          f'(folds={num_folds}, ensemble={"on" if "ensemble_seeds" in config else "off"})',
          flush=True)
    # If this row needs HAP-beta patching, wrap create_model so every model
    # produced inside run_cv (including each ensemble seed) gets the patch.
    if hap_init_beta is not None:
        original_create_model = _etc.create_model

        def _patched_create_model(_cfg, _device='cpu'):
            m = original_create_model(_cfg, _device)
            _patch_hap_beta(m, hap_init_beta, hap_freeze)
            return m

        _etc.create_model = _patched_create_model
    else:
        original_create_model = None
    t0 = time.time()
    try:
        results, all_metrics = run_cv(config, device=device, num_folds=num_folds)
    finally:
        if original_create_model is not None:
            _etc.create_model = original_create_model
    dt = time.time() - t0

    payload = {
        'dataset': dataset,
        'row': row,
        'num_folds': num_folds,
        'ensemble_seeds': config.get('ensemble_seeds'),
        'config': {k: v for k, v in config.items()
                   if isinstance(v, (int, float, bool, str, list)) and 'path' not in k},
        'metrics_mean_std': {k: list(v) for k, v in results.items()},
        'per_fold': [{k: float(v) for k, v in m.items()} for m in all_metrics],
        'wall_clock_seconds': dt,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'  saved: {out_path}  ({dt/60:.1f} min)', flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='MIML-ALFA++ headline ablation runner.')
    parser.add_argument('--rows', type=str, default='bce,alfa_full,alfapp_full')
    parser.add_argument('--datasets', type=str, default=','.join(CORE_DATASETS))
    parser.add_argument('--folds', type=int, default=None)
    parser.add_argument('--no-ensemble', action='store_true')
    parser.add_argument('--outdir', type=str, default='results/alfapp_ablation')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--device', type=str, default=None,
                        choices=[None, 'cpu', 'mps', 'cuda'],
                        help='Force a specific device (default: auto MPS > CUDA > CPU)')
    args = parser.parse_args()

    rows = [r.strip() for r in args.rows.split(',') if r.strip()]
    for r in rows:
        if r not in ROW_OVERRIDES:
            raise ValueError(f'Unknown row {r!r}; choose from {list(ROW_OVERRIDES)}')
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    for d in datasets:
        if d not in CONFIGS:
            raise ValueError(f'Unknown dataset {d!r}; choose from {list(CONFIGS)}')

    outdir = os.path.join(_HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    print(f'Output directory: {outdir}')
    print(f'Rows:     {rows}')
    print(f'Datasets: {datasets}')

    grand_t0 = time.time()
    for row in rows:
        for dataset in datasets:
            try:
                run_row(dataset, row, args.folds, args.no_ensemble, outdir,
                        args.rerun, force_device=args.device)
            except Exception as exc:
                # Don't let one fold crash kill the whole sweep.
                err_path = os.path.join(outdir, f'{row}__{dataset}.error.txt')
                with open(err_path, 'w') as f:
                    f.write(f'{type(exc).__name__}: {exc}\n')
                print(f'  ERROR ({row}, {dataset}): {exc!r}', flush=True)
    print(f'\n=== ablation sweep done in {(time.time()-grand_t0)/60:.1f} min ===')


if __name__ == '__main__':
    main()

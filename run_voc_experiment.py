"""ICDM 2026 — Pascal VOC 2012 MIML experiment runner.

Adds the only modern image-recognition benchmark to the experimental matrix.
Reuses the same `run_cv` + `ROW_OVERRIDES` infrastructure as
`run_alfapp_ablation.py`, but injects a VOC-specific dataset config and
defaults appropriate to the larger 2048-dim ResNet-50 features.

Pipeline (handled by `datasets/voc_miml.py`):
    image -> 4x4 patch grid -> frozen ResNet-50 (avgpool, 2048-dim)
    bag : (16, 2048) float32; label : 20-dim {-1,+1}

Usage:
    # Build the VOC2012 .mat once (slow; ResNet-50 over ~11.5k images):
    python -m datasets.voc_miml

    # Run HAP + 4 baselines (writes to results/voc/<method>.json):
    python run_voc_experiment.py
    python run_voc_experiment.py --rows hap_only,alfa_full
    python run_voc_experiment.py --folds 5
    python run_voc_experiment.py --device mps
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from run_all_enhanced import _base_config, DATA_DIR  # noqa: E402
from run_alfapp_ablation import (  # noqa: E402
    ROW_OVERRIDES, _patch_hap_beta, _device,
)
from miml_clam.training.enhanced_train_cv import run_cv  # noqa: E402
from miml_clam.training import enhanced_train_cv as _etc  # noqa: E402


# ---------------------------------------------------------------------------
# VOC2012 config — input_dim=2048, embed_dim=128, batch_size=8,
# 3-seed ensemble (compute-budget exception, similar to corel5k).
# ---------------------------------------------------------------------------

def _voc_config() -> dict:
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'voc2012.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'voc2012_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels',
        'dataset': 'voc2012', 'dataset_format': 'standard',
        'fixed_bag_size': True,           # always 16 patches per image
        'default_folds': 10,
        # Larger feature dim -> wider embedding, smaller batch.
        'embed_dim': 128, 'ff_dim': 256, 'attn_dim': 64,
        'batch_size': 8,
        # Compute-budget exception: 3-seed instead of 7-seed ensemble.
        'ensemble_seeds': [42, 123, 456],
    })
    return c


CONFIGS = {'voc2012': _voc_config()}

# Default rows: HAP-only plus 4 comparators (BCE baseline, ALFA-Full,
# ALFA-Full+HAP, full ALFA++ stack).
DEFAULT_ROWS = ['hap_only', 'bce', 'alfa_full', 'alfa_hap', 'alfapp_full']


def _check_data_files(config: dict) -> None:
    """Fail fast with a clear message if the VOC .mat files aren't built yet."""
    missing = []
    for key in ('data_path', 'cv_mat_path'):
        path = config.get(key)
        if path and not os.path.exists(path):
            missing.append(path)
    if missing:
        msg = ('VOC2012 .mat files not found:\n  '
               + '\n  '.join(missing)
               + "\n\nBuild them first with:\n  "
                 "python -m datasets.voc_miml\n"
                 "(or set $VOC2012_ROOT to point at an existing VOC2012 cache).")
        raise FileNotFoundError(msg)


def run_row(row: str, folds: int | None, no_ensemble: bool,
            outdir: str, rerun: bool, force_device: str | None = None) -> str:
    """Run a single ablation row on voc2012 — mirrors run_alfapp_ablation.run_row.

    Returns the path of the JSON result file.
    """
    out_path = os.path.join(outdir, f'{row}.json')
    if os.path.exists(out_path) and not rerun:
        print(f'  cached: {out_path}', flush=True)
        return out_path

    config = deepcopy(CONFIGS['voc2012'])
    config.update(ROW_OVERRIDES[row])
    hap_init_beta = config.pop('_hap_init_beta', None)
    hap_freeze = config.pop('_hap_freeze_beta', False)
    if no_ensemble:
        config.pop('ensemble_seeds', None)
    num_folds = folds if folds is not None else config.get('default_folds', 10)

    _check_data_files(config)

    device = _device(force_device)
    print(f'\n>>> {row} on voc2012 (device={device})  '
          f'(folds={num_folds}, ensemble='
          f'{"on" if "ensemble_seeds" in config else "off"})', flush=True)

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
        'dataset': 'voc2012',
        'row': row,
        'num_folds': num_folds,
        'ensemble_seeds': config.get('ensemble_seeds'),
        'config': {k: v for k, v in config.items()
                   if isinstance(v, (int, float, bool, str, list))
                   and 'path' not in k},
        'metrics_mean_std': {k: list(v) for k, v in results.items()},
        'per_fold': [{k: float(v) for k, v in m.items()} for m in all_metrics],
        'wall_clock_seconds': dt,
    }
    os.makedirs(outdir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'  saved: {out_path}  ({dt/60:.1f} min)', flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='ICDM 2026 — VOC2012 runner.')
    parser.add_argument('--rows', type=str, default=','.join(DEFAULT_ROWS),
                        help=('Comma-separated method rows from '
                              'run_alfapp_ablation.ROW_OVERRIDES '
                              f'(default: {",".join(DEFAULT_ROWS)})'))
    parser.add_argument('--folds', type=int, default=None,
                        help='Override number of CV folds (default 10).')
    parser.add_argument('--no-ensemble', action='store_true',
                        help='Disable the 3-seed ensemble (single model).')
    parser.add_argument('--outdir', type=str,
                        default=os.path.join(_HERE, 'results', 'voc'),
                        help='Output directory for per-row JSON results.')
    parser.add_argument('--rerun', action='store_true',
                        help='Force re-run cells even if their JSON exists.')
    parser.add_argument('--device', type=str, default=None,
                        choices=[None, 'cpu', 'mps', 'cuda'],
                        help='Force a specific device (default: auto).')
    args = parser.parse_args(argv)

    rows = [r.strip() for r in args.rows.split(',') if r.strip()]
    for r in rows:
        if r not in ROW_OVERRIDES:
            raise ValueError(f'Unknown row {r!r}; choose from '
                             f'{list(ROW_OVERRIDES)}')

    os.makedirs(args.outdir, exist_ok=True)
    print(f'Output directory: {args.outdir}')
    print(f'Rows:     {rows}')
    print(f'Dataset:  voc2012')

    np.random.seed(42)
    torch.manual_seed(42)

    grand_t0 = time.time()
    for row in rows:
        try:
            run_row(row, args.folds, args.no_ensemble, args.outdir,
                    args.rerun, force_device=args.device)
        except Exception as exc:
            err_path = os.path.join(args.outdir, f'{row}.error.txt')
            os.makedirs(args.outdir, exist_ok=True)
            with open(err_path, 'w') as f:
                f.write(f'{type(exc).__name__}: {exc}\n')
            print(f'  ERROR ({row}): {exc!r}', flush=True)
    print(f'\n=== voc2012 sweep done in {(time.time()-grand_t0)/60:.1f} min ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())

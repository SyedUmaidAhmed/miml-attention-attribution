"""Sweep driver for the four external MIL baselines (AttnMIL, GatedAttention,
DSMIL, TransMIL-lite).

Mirrors the structure and conventions of `run_alfapp_ablation.py` so the
results sit alongside the in-house ablation cells and can be aggregated by
the same downstream tooling.

Key design points:
  * Reuses the canonical dataset registry from `run_all_enhanced.CONFIGS`
    so fold protocols / data paths / dataset_format flags are IDENTICAL to
    `run_alfapp_ablation.py`. In particular, `cv_mat_path` is preserved,
    which means the same `10CV.mat` (or DD-format) fold indices are used.
  * Swaps the model factory in `enhanced_train_cv` for the duration of the
    run via monkey-patching. This is intentionally non-invasive — no file
    under `miml_clam/` is modified.
  * Each (model, dataset) cell writes its own JSON to
    `results/baselines/<model>__<dataset>.json` and is idempotent.

Usage:
    python baselines/run_baselines.py \
        --models attnmil,gated_attn,dsmil,transmil_lite \
        --datasets scene,reuters,mscv2,letter_frost,letter_carroll

Smoke test (1 fold of attnmil on scene, no ensemble):
    python baselines/run_baselines.py --smoke
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

# Make the parent (project root) importable so we can grab CONFIGS and the
# enhanced_train_cv module just like run_alfapp_ablation.py does.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run_all_enhanced import CONFIGS as _BASE_CONFIGS                   # noqa: E402

# Inject VOC2012 and NUS-WIDE configs so external baselines can run on the
# modern image benchmarks via the same fold-index protocol as HAP.
CONFIGS = dict(_BASE_CONFIGS)
try:
    from run_voc_experiment import _voc_config                            # noqa: E402
    CONFIGS['voc2012'] = _voc_config()
except Exception as _e:                                                   # noqa: BLE001
    pass
try:
    from run_nuswide_experiment import _nuswide_config                    # noqa: E402
    CONFIGS['nuswide'] = _nuswide_config()
except Exception as _e:                                                   # noqa: BLE001
    pass
from miml_clam.training import enhanced_train_cv as _etcv                # noqa: E402
from miml_clam.training.enhanced_train_cv import run_cv                  # noqa: E402

# Local baseline classes
from baselines.attnmil import AttnMIL                                    # noqa: E402
from baselines.gated_attn import GatedAttention                          # noqa: E402
from baselines.dsmil import DSMIL                                        # noqa: E402
from baselines.transmil_lite import TransMILLite                         # noqa: E402
from baselines.ml_decoder import MLDecoder                                # noqa: E402
from baselines.q2l import Q2L                                            # noqa: E402
from baselines.dsmil_orig import DSMILOriginal                          # noqa: E402


# ----------------------------------------------------------------------
# Model factories
# ----------------------------------------------------------------------

def _make_attnmil(cfg, device):
    return AttnMIL(
        input_dim=cfg['input_dim'],
        num_labels=cfg['num_labels'],
        embed_dim=cfg['embed_dim'],
        dropout=cfg['dropout'],
        embedding_layers=cfg.get('embedding_layers', 2),
        attn_dim=cfg.get('attn_dim', cfg['embed_dim'] // 2),
    ).to(device)


def _make_gated_attn(cfg, device):
    return GatedAttention(
        input_dim=cfg['input_dim'],
        num_labels=cfg['num_labels'],
        embed_dim=cfg['embed_dim'],
        dropout=cfg['dropout'],
        embedding_layers=cfg.get('embedding_layers', 2),
        attn_dim=cfg.get('attn_dim', cfg['embed_dim'] // 2),
    ).to(device)


def _make_dsmil(cfg, device):
    return DSMIL(
        input_dim=cfg['input_dim'],
        num_labels=cfg['num_labels'],
        embed_dim=cfg['embed_dim'],
        dropout=cfg['dropout'],
        embedding_layers=cfg.get('embedding_layers', 2),
        attn_dim=cfg.get('attn_dim', cfg['embed_dim'] // 2),
    ).to(device)


def _make_dsmil_orig(cfg, device):
    return DSMILOriginal(
        input_dim=cfg['input_dim'],
        num_labels=cfg['num_labels'],
        embed_dim=cfg['embed_dim'],
        dropout=cfg['dropout'],
        embedding_layers=cfg.get('embedding_layers', 2),
    ).to(device)


def _make_transmil_lite(cfg, device):
    return TransMILLite(
        input_dim=cfg['input_dim'],
        num_labels=cfg['num_labels'],
        embed_dim=cfg['embed_dim'],
        dropout=cfg['dropout'],
        embedding_layers=cfg.get('embedding_layers', 2),
        nhead=cfg.get('nhead', 4),
        ff_dim=cfg.get('ff_dim', cfg['embed_dim'] * 2),
        transformer_layers=cfg.get('transformer_layers', 1),
    ).to(device)


def _make_ml_decoder(cfg, device):
    return MLDecoder(
        input_dim=cfg['input_dim'],
        num_labels=cfg['num_labels'],
        embed_dim=cfg['embed_dim'],
        dropout=cfg['dropout'],
        embedding_layers=cfg.get('embedding_layers', 2),
        nhead=cfg.get('nhead', 4),
        ff_dim=cfg.get('ff_dim', cfg['embed_dim'] * 2),
    ).to(device)


def _make_q2l(cfg, device):
    return Q2L(
        input_dim=cfg['input_dim'],
        num_labels=cfg['num_labels'],
        embed_dim=cfg['embed_dim'],
        dropout=cfg['dropout'],
        embedding_layers=cfg.get('embedding_layers', 2),
        nhead=cfg.get('nhead', 4),
        ff_dim=cfg.get('ff_dim', cfg['embed_dim'] * 2),
        num_decoder_layers=cfg.get('q2l_decoder_layers', 2),
    ).to(device)


MODEL_FACTORIES = {
    'attnmil':       _make_attnmil,
    'gated_attn':    _make_gated_attn,
    'dsmil':         _make_dsmil,
    'dsmil_orig':    _make_dsmil_orig,
    'transmil_lite': _make_transmil_lite,
    'ml_decoder':    _make_ml_decoder,
    'q2l':           _make_q2l,
}


# ----------------------------------------------------------------------
# Config overrides — disable everything that assumes the MIML_CLAM internals
# (correlation head, instance clustering, contrastive/anchor heads). The
# trainer still supports ASL / Mixup / SWA / threshold-opt, which we keep
# OFF for these external baselines so we are reporting the published
# baseline behaviour rather than a baseline + our training tricks.
# ----------------------------------------------------------------------
BASELINE_OVERRIDES = {
    'use_correlation':         False,
    'use_instance_clustering': False,
    'use_contrastive':         False,
    'use_anchor':              False,
    'use_hap':                 False,
    'use_asymmetric_loss':     False,
    'use_two_way_loss':        False,
    'use_mixup':               False,
    'use_psemix':              False,
}


CORE_DATASETS = ['scene', 'reuters', 'mscv2', 'letter_frost', 'letter_carroll']


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _device(force: str | None = None):
    if force is not None:
        return torch.device(force)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _patched_create_model(factory):
    """Build a `create_model(config, device)` replacement using `factory`."""
    def _inner(config, device='cpu'):
        return factory(config, device)
    return _inner


def run_cell(model_name: str, dataset: str, folds: int | None, no_ensemble: bool,
             outdir: str, rerun: bool, force_device: str | None = None) -> str:
    """Run a single (model, dataset) cell and write the JSON result."""
    out_path = os.path.join(outdir, f'{model_name}__{dataset}.json')
    if os.path.exists(out_path) and not rerun:
        print(f'  cached: {out_path}', flush=True)
        return out_path

    if model_name not in MODEL_FACTORIES:
        raise ValueError(
            f'Unknown model {model_name!r}; choose from {list(MODEL_FACTORIES)}')
    if dataset not in CONFIGS:
        raise ValueError(
            f'Unknown dataset {dataset!r}; choose from {list(CONFIGS)}')

    config = deepcopy(CONFIGS[dataset])
    config.update(BASELINE_OVERRIDES)
    if no_ensemble:
        config.pop('ensemble_seeds', None)
    num_folds = folds if folds is not None else config.get('default_folds', 5)

    device = _device(force_device)

    # Monkey-patch the model factory used inside enhanced_train_cv. We restore
    # it in a finally block so an exception in one cell does not leak the
    # patched factory into the next iteration of the sweep.
    factory = MODEL_FACTORIES[model_name]
    original_create_model = _etcv.create_model
    _etcv.create_model = _patched_create_model(factory)

    try:
        print(f'\n>>> {model_name} on {dataset} (device={device})  '
              f'(folds={num_folds}, ensemble='
              f'{"on" if "ensemble_seeds" in config else "off"})',
              flush=True)
        t0 = time.time()
        results, all_metrics = run_cv(config, device=device, num_folds=num_folds)
        dt = time.time() - t0
    finally:
        _etcv.create_model = original_create_model

    payload = {
        'dataset': dataset,
        'model':   model_name,
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
    parser = argparse.ArgumentParser(
        description='External MIL-baseline sweep (AttnMIL / GatedAttention / '
                    'DSMIL / TransMIL-lite).')
    parser.add_argument('--models',   type=str,
                        default=','.join(MODEL_FACTORIES.keys()))
    parser.add_argument('--datasets', type=str,
                        default=','.join(CORE_DATASETS))
    parser.add_argument('--folds',    type=int, default=None)
    parser.add_argument('--no-ensemble', action='store_true')
    parser.add_argument('--outdir',   type=str, default='results/baselines')
    parser.add_argument('--rerun',    action='store_true')
    parser.add_argument('--device',   type=str, default=None,
                        choices=[None, 'cpu', 'mps', 'cuda'],
                        help='Force a specific device (default: auto).')
    parser.add_argument('--smoke', action='store_true',
                        help='1 fold of attnmil on scene, no ensemble — '
                             'quick wiring check.')
    args = parser.parse_args()

    if args.smoke:
        # Smoke test path — caller can run this for a quick pipeline check.
        models   = ['attnmil']
        datasets = ['scene']
        folds    = 1
        no_ensemble = True
        outdir   = os.path.join(_ROOT, 'results', 'baselines_smoke')
        rerun    = True
    else:
        models   = [m.strip() for m in args.models.split(',')   if m.strip()]
        datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
        folds    = args.folds
        no_ensemble = args.no_ensemble
        outdir   = os.path.join(_ROOT, args.outdir)
        rerun    = args.rerun

    for m in models:
        if m not in MODEL_FACTORIES:
            raise ValueError(
                f'Unknown model {m!r}; choose from {list(MODEL_FACTORIES)}')
    for d in datasets:
        if d not in CONFIGS:
            raise ValueError(
                f'Unknown dataset {d!r}; choose from {list(CONFIGS)}')

    os.makedirs(outdir, exist_ok=True)
    print(f'Output directory: {outdir}')
    print(f'Models:   {models}')
    print(f'Datasets: {datasets}')

    grand_t0 = time.time()
    for model in models:
        for dataset in datasets:
            try:
                run_cell(model, dataset, folds, no_ensemble, outdir,
                         rerun, force_device=args.device)
            except Exception as exc:
                err_path = os.path.join(outdir, f'{model}__{dataset}.error.txt')
                with open(err_path, 'w') as f:
                    f.write(f'{type(exc).__name__}: {exc}\n')
                print(f'  ERROR ({model}, {dataset}): {exc!r}', flush=True)
    print(f'\n=== baseline sweep done in {(time.time()-grand_t0)/60:.1f} min ===')


if __name__ == '__main__':
    main()

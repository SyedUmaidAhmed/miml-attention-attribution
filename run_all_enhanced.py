#!/usr/bin/env python3
"""MIML-CLAM++ Enhanced: Run on ALL 5 MIML datasets.

Enhancements over base MIML-CLAM++:
  1. Asymmetric Loss (ASL) replaces BCE
  2. Bag Mixup augmentation
  3. 7-seed ensemble

Usage:
    python run_all_enhanced.py scene              # Scene (cached result)
    python run_all_enhanced.py reuters             # Reuters (cached result)
    python run_all_enhanced.py mscv2               # MSCV2 (5-fold, 7-seed)
    python run_all_enhanced.py letter_frost         # Letter Frost (5-fold, 7-seed)
    python run_all_enhanced.py letter_carroll       # Letter Carroll (5-fold, 7-seed)
    python run_all_enhanced.py dd                   # All 3 DD datasets
    python run_all_enhanced.py all                  # All 5 datasets
    python run_all_enhanced.py mscv2 --folds 2      # Quick sanity test
    python run_all_enhanced.py mscv2 --rerun         # Force rerun (ignore cache)
    python run_all_enhanced.py mscv2 --no-ensemble   # Single model
    python run_all_enhanced.py mscv2 --no-mixup      # Disable mixup
    python run_all_enhanced.py mscv2 --no-asl        # Use BCE instead of ASL
"""

import torch
import numpy as np
import sys
import os
import json
import argparse
from datetime import datetime

# Resolve local imports (miml_clam package lives alongside this file)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from miml_clam.training.enhanced_train_cv import run_cv

# Path to datasets (sibling VIT_MIL directory)
# Dataset directory resolution (2026-09-05, for the public code release):
#   1. $MIML_DATA_DIR if set;
#   2. the historical sibling checkout ../VIT_MIL/Code/MIML-SELFVIT if present;
#   3. <repo>/data/MIML-SELFVIT (what the release repository ships).
_LEGACY_DATA_DIR = os.path.join(_HERE, '..', 'VIT_MIL', 'Code', 'MIML-SELFVIT')
DATA_DIR = (os.environ.get('MIML_DATA_DIR')
            or (_LEGACY_DATA_DIR if os.path.isdir(_LEGACY_DATA_DIR)
                else os.path.join(_HERE, 'data', 'MIML-SELFVIT')))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


# ============================================================
# Cached results (already obtained, skip training by default)
# ============================================================

CACHED_RESULTS = {
    'scene': {
        'HammingLoss': [0.1649, 0.0145],
        'OneError': [0.2855, 0.0348],
        'Coverage': [0.892, 0.0456],
        'RankingLoss': [0.1545, 0.0239],
        'AveragePrecision': [0.8112, 0.0212],
    },
    'reuters': {
        'HammingLoss': [0.0372, 0.0063],
        'OneError': [0.0665, 0.0138],
        'Coverage': [0.2965, 0.0510],
        'RankingLoss': [0.0213, 0.0058],
        'AveragePrecision': [0.9584, 0.0085],
    },
}


# ============================================================
# Enhanced Dataset Configurations (all 5 datasets)
# ============================================================

def _base_config():
    """Shared architecture and training hyperparameters."""
    return {
        # Architecture
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        # Training
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'dropout': 0.15,
        'use_threshold_opt': True,
        # Enhancements (current ALFA-Full defaults)
        'use_asymmetric_loss': True,
        'asl_gamma_neg': 2, 'asl_gamma_pos': 0, 'asl_clip': 0.02,
        'use_mixup': True, 'mixup_alpha': 0.2,
        'ensemble_seeds': [42, 123, 456, 789, 1024, 2048, 3072],
        # MIML-ALFA++ extensions (all default OFF — opt-in via CLI flags)
        'use_two_way_loss': False,         # ASL2 (Two-Way ML loss)
        'twl_margin': 0.05, 'twl_lam_rel': 0.5,
        'use_psemix': False,               # similarity-driven pseudo-bag Mixup
        'psemix_K': 4, 'psemix_share_label_bias': 0.7,
        'use_contrastive': False,          # Jaccard-weighted SupCon auxiliary
        'contrastive_dim': 128, 'contrastive_tau': 0.1,
        # lam_con dropped from 0.15 -> 0.05 after smoke Tier 3 showed
        # contrastive component dominating the total loss when ASL2 main
        # loss is small (Reuters AP regressed 0.952 -> 0.941 at lam=0.15).
        'lam_con': 0.05, 'lam_con_warmup': 20,
        # Label Anchor Alignment (LAA) — novelty term:
        'use_anchor': False,               # opt-in via --use-laa
        'anchor_dim': 32,
        'lam_anchor': 0.3, 'lam_anchor_warmup': 10,
        # Hybrid Attention Pooling (HAP) — soft+hard attention mix per label:
        'use_hap': False,                  # opt-in via --use-hap
        'hap_topk': 3,
    }


def _make_configs():
    """Build all 5 enhanced dataset configs."""
    configs = {}

    # Scene: standard format, 10-fold, fixed bag size
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'scene.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, '10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels',
        'dataset': 'scene', 'dataset_format': 'standard',
        'fixed_bag_size': True,
        'default_folds': 10,
    })
    configs['scene'] = c

    # Reuters: standard format, 10-fold, variable bag size
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'reuters_MIML.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, '10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels',
        'dataset': 'reuters', 'dataset_format': 'standard',
        'fixed_bag_size': False,
        'default_folds': 10,
    })
    configs['reuters'] = c

    # MSCV2: DD format, 5-fold, variable bag size
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'MSCV2.mat'),
        'cv_mat_path': None,
        'dataset': 'mscv2', 'dataset_format': 'dd',
        'fixed_bag_size': False,
        'default_folds': 5,
    })
    configs['mscv2'] = c

    # Letter Frost: DD format, 5-fold, variable bag size
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'letter_frost.mat'),
        'cv_mat_path': None,
        'dataset': 'letter_frost', 'dataset_format': 'dd',
        'fixed_bag_size': False,
        'default_folds': 5,
    })
    configs['letter_frost'] = c

    # Letter Carroll: DD format, 5-fold, variable bag size
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'letter_carroll.mat'),
        'cv_mat_path': None,
        'dataset': 'letter_carroll', 'dataset_format': 'dd',
        'fixed_bag_size': False,
        'default_folds': 5,
    })
    configs['letter_carroll'] = c

    # --- New MIML datasets ---

    # Birdsong: standard format, 10-fold, variable bag size, 19 labels
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'birdsong.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'birdsong_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels',
        'dataset': 'birdsong', 'dataset_format': 'standard',
        'fixed_bag_size': False,
        'default_folds': 10,
    })
    configs['birdsong'] = c

    # Protein Haloarcula: 304 bags, 216-dim, 21 labels
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'protein_haloarcula.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'protein_haloarcula_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels',
        'dataset': 'protein_haloarcula', 'dataset_format': 'standard',
        'fixed_bag_size': False,
        'default_folds': 10,
    })
    configs['protein_haloarcula'] = c

    # Protein Pyrococcus: 425 bags, 216-dim, 52 labels
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'protein_pyrococcus.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'protein_pyrococcus_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels',
        'dataset': 'protein_pyrococcus', 'dataset_format': 'standard',
        'fixed_bag_size': False,
        'default_folds': 10,
    })
    configs['protein_pyrococcus'] = c

    # Protein Geobacter: 379 bags, 216-dim, 28 labels
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'protein_geobacter.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'protein_geobacter_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels',
        'dataset': 'protein_geobacter', 'dataset_format': 'standard',
        'fixed_bag_size': False,
        'default_folds': 10,
    })
    configs['protein_geobacter'] = c

    # Protein Azotobacter: 407 bags, 216-dim, 46 labels
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'protein_azotobacter.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'protein_azotobacter_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels',
        'dataset': 'protein_azotobacter', 'dataset_format': 'standard',
        'fixed_bag_size': False,
        'default_folds': 10,
    })
    configs['protein_azotobacter'] = c

    # Yeast: standard format (k-NN bagged), 10-fold, fixed bag size (5 instances), 14 labels
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'yeast.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'yeast_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels',
        'dataset': 'yeast', 'dataset_format': 'standard',
        'fixed_bag_size': True,
        'default_folds': 10,
    })
    configs['yeast'] = c

    # Emotions: standard format (k-NN bagged), 10-fold, fixed bag size, 6 labels
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'emotions.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'emotions_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels',
        'dataset': 'emotions', 'dataset_format': 'standard',
        'fixed_bag_size': True,
        'default_folds': 10,
    })
    configs['emotions'] = c

    # Medical: standard format (k-NN bagged), 10-fold, fixed bag size, 45 labels
    # batch_size lowered for OOM guard given high label count
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'medical.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'medical_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels',
        'dataset': 'medical', 'dataset_format': 'standard',
        'fixed_bag_size': True,
        'default_folds': 10,
        'batch_size': 8,
    })
    configs['medical'] = c

    # Corel5k: standard format, 10-fold, fixed bag size, 374 labels.
    # Largest label space in the suite; OOM guards: B=4, fewer epochs.
    c = _base_config()
    c.update({
        'data_path': os.path.join(DATA_DIR, 'corel5k.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'corel5k_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels',
        'dataset': 'corel5k', 'dataset_format': 'standard',
        'fixed_bag_size': True,
        'default_folds': 10,
        'batch_size': 4,
        'phase1_epochs': 50,
        'phase2_max_epochs': 80,
        # corel5k restricted to 3-seed ensemble (compute-budget footnote).
        'ensemble_seeds': [42, 123, 456],
    })
    configs['corel5k'] = c

    return configs


CONFIGS = _make_configs()

# Published baselines (Scene/Reuters only)
BASELINES = {
    'scene': {
        'HammingLoss': (0.165, 'lower', 'MIML-kNN'),
        'OneError': (0.310, 'lower', 'MIML-kNN'),
        'Coverage': (0.896, 'lower', 'MIML-kNN'),
        'RankingLoss': (0.159, 'lower', 'MIML-kNN'),
        'AveragePrecision': (0.801, 'higher', 'MIML-kNN'),
    },
    'reuters': {
        'HammingLoss': (0.033, 'lower', 'MimlSvm'),
        'OneError': (0.055, 'lower', 'MimlSvmmi'),
        'Coverage': (0.284, 'lower', 'MimlSvmmi'),
        'RankingLoss': (0.020, 'lower', 'MimlSvmmi'),
        'AveragePrecision': (0.965, 'higher', 'MimlSvmmi'),
    },
}


def compare_to_baselines(results, dataset_name):
    """Compare results to published baselines."""
    if dataset_name not in BASELINES:
        print(f"\n  No published baselines for {dataset_name}.")
        return 0

    baselines = BASELINES[dataset_name]
    beats = 0
    print(f"\n  Comparison to Published Baselines:")
    print(f"  {'Metric':<20} {'Ours':>10} {'Baseline':>10} {'Method':<15} {'Status':<10}")
    print(f"  {'-'*65}")

    for metric, (baseline, direction, method) in baselines.items():
        if metric in results:
            ours = results[metric][0]
            if direction == 'lower':
                won = ours < baseline
            else:
                won = ours > baseline
            status = "BEAT" if won else "miss"
            if won:
                beats += 1
            print(f"  {metric:<20} {ours:>10.4f} {baseline:>10.3f} {method:<15} {status:<10}")

    print(f"\n  Result: {beats}/5 beats")
    return beats


def main():
    all_datasets = list(CONFIGS.keys())
    dd_datasets = ['mscv2', 'letter_frost', 'letter_carroll']
    new_datasets = ['birdsong', 'protein_haloarcula', 'protein_pyrococcus',
                    'protein_geobacter', 'protein_azotobacter',
                    'yeast', 'emotions', 'medical', 'corel5k']

    parser = argparse.ArgumentParser(description='MIML-CLAM++ Enhanced — All Datasets')
    parser.add_argument('dataset', type=str, nargs='?', default='all',
                        choices=all_datasets + ['dd', 'new', 'all'],
                        help='Dataset to run (default: all)')
    parser.add_argument('--folds', type=int, default=None,
                        help='Override number of CV folds')
    parser.add_argument('--rerun', action='store_true',
                        help='Force rerun even for cached datasets')
    parser.add_argument('--no-ensemble', action='store_true',
                        help='Disable ensemble (single model)')
    parser.add_argument('--no-mixup', action='store_true',
                        help='Disable bag mixup')
    parser.add_argument('--no-asl', action='store_true',
                        help='Disable asymmetric loss (use BCE)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory')
    # --- MIML-ALFA++ opt-in flags (default OFF) ---
    parser.add_argument('--use-asl2', action='store_true',
                        help='Use Two-Way ML loss instead of ASL (MIML-ALFA++)')
    parser.add_argument('--use-psemix', action='store_true',
                        help='Use similarity-driven pseudo-bag Mixup (MIML-ALFA++)')
    parser.add_argument('--use-contrastive', action='store_true',
                        help='Add Jaccard-weighted SupCon auxiliary loss (MIML-ALFA++)')
    parser.add_argument('--use-laa', action='store_true',
                        help='Add Label-Anchor Alignment auxiliary loss (MIML-ALFA++ novelty)')
    parser.add_argument('--use-hap', action='store_true',
                        help='Replace soft attention pooling with Hybrid Attention Pooling (MIML-ALFA++ novelty)')
    args = parser.parse_args()

    # Resolve dataset list
    if args.dataset == 'all':
        datasets_to_run = all_datasets
    elif args.dataset == 'dd':
        datasets_to_run = dd_datasets
    elif args.dataset == 'new':
        datasets_to_run = new_datasets
    else:
        datasets_to_run = [args.dataset]

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Check if any enhancement is disabled or new MIML-ALFA++ flag is set
    # (either invalidates the cache).
    any_override = (
        args.no_ensemble or args.no_mixup or args.no_asl or args.folds is not None
        or args.use_asl2 or args.use_psemix or args.use_contrastive or args.use_laa
        or args.use_hap
    )

    all_results = {}
    for dataset_name in datasets_to_run:
        config = CONFIGS[dataset_name].copy()

        # Override data directory if specified
        if args.data_dir:
            for key in ['data_path', 'cv_mat_path']:
                if config.get(key):
                    filename = os.path.basename(config[key])
                    config[key] = os.path.join(args.data_dir, filename)

        # Override folds
        num_folds = args.folds or config.get('default_folds', 10)

        # Apply disabling flags
        if args.no_ensemble:
            config.pop('ensemble_seeds', None)
        if args.no_mixup:
            config['use_mixup'] = False
        if args.no_asl:
            config['use_asymmetric_loss'] = False
        # MIML-ALFA++ opt-ins
        if args.use_asl2:
            config['use_two_way_loss'] = True
            # Two-Way takes precedence over ASL in trainer; keep ASL flag too
            # so the cache key still reflects the asymmetric-loss family.
        if args.use_psemix:
            config['use_psemix'] = True
            config['use_mixup'] = True   # ensure mixup-pipeline path runs
        if args.use_contrastive:
            config['use_contrastive'] = True
        if args.use_laa:
            config['use_anchor'] = True
        if args.use_hap:
            config['use_hap'] = True

        # Check cache
        use_cache = (not args.rerun and not any_override
                     and dataset_name in CACHED_RESULTS)

        enhancements = []
        if config.get('use_asymmetric_loss'):
            enhancements.append('ASL')
        if config.get('use_mixup'):
            enhancements.append('Mixup')
        seeds = config.get('ensemble_seeds', [])
        if seeds:
            enhancements.append(f'{len(seeds)}-seed')

        print(f"\n{'#'*70}")
        print(f"# MIML-CLAM++ Enhanced: {dataset_name} ({num_folds}-fold CV)")
        print(f"# Enhancements: {', '.join(enhancements) if enhancements else 'none'}")
        if use_cache:
            print(f"# Using CACHED results (use --rerun to force rerun)")
        print(f"{'#'*70}")

        if use_cache:
            results = {k: tuple(v) for k, v in CACHED_RESULTS[dataset_name].items()}
        else:
            torch.manual_seed(42)
            np.random.seed(42)
            results, per_fold = run_cv(config, device, num_folds=num_folds)

        all_results[dataset_name] = results

        # Print results
        print(f"\n  Results for {dataset_name}:")
        for metric in ['HammingLoss', 'OneError', 'Coverage', 'RankingLoss', 'AveragePrecision']:
            if metric in results:
                mean, std = results[metric][0], results[metric][1]
                print(f"    {metric}: {mean:.4f} +/- {std:.4f}")

        compare_to_baselines(results, dataset_name)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f'enhanced_all_{ts}.json')
    with open(output_path, 'w') as f:
        json.dump({
            name: {k: [float(v[0]), float(v[1])] for k, v in res.items()}
            for name, res in all_results.items()
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print("MIML-CLAM++ ENHANCED — ALL DATASETS SUMMARY")
    print(f"{'='*70}")
    metrics = ['HammingLoss', 'OneError', 'Coverage', 'RankingLoss', 'AveragePrecision']
    header = f"  {'Dataset':<16}" + "".join(f"{'  ' + m:>20}" for m in metrics)
    print(header)
    print(f"  {'-'*len(header)}")
    for name, res in all_results.items():
        row = f"  {name:<16}"
        for m in metrics:
            if m in res:
                row += f"{res[m][0]:>20.4f}"
            else:
                row += f"{'N/A':>20}"
        print(row)


if __name__ == '__main__':
    main()

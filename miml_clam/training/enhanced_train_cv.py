"""Enhanced Cross-Validation training loop for MIML-CLAM++.

Uses EnhancedTrainer with ASL + Mixup support.
Otherwise identical to train_cv.py.
"""

import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from ..data.miml_dataset_cv import MIMLDatasetCV, collate_fn, get_cv_splits
from ..data.miml_dataset_dd import MIMLDatasetDD, collate_fn_dd
from ..models.miml_clam import MIML_CLAM
from ..metrics.miml_metrics import (
    AllFive, AllFiveWithThresholds, optimize_thresholds, AveragePrecision as AP_fn
)
from .enhanced_trainer import EnhancedTrainer


# Optional hook for the instance-level attribution study. When set to a
# callable it is invoked after each model finishes training, with
#   (fold_idx, seed, model, test_loader, dataset, test_idx, device)
# and may inspect the trained model however it likes. It is None by default and
# is never consulted during normal sweeps, so published cells are unaffected.
# Set it via `enhanced_train_cv.ATTENTION_COLLECTOR = fn` and restore to None.
ATTENTION_COLLECTOR = None


def create_model(config, device='cpu'):
    """Create MIML-CLAM++ model from config dict."""
    model = MIML_CLAM(
        input_dim=config['input_dim'],
        num_labels=config['num_labels'],
        embed_dim=config['embed_dim'],
        nhead=config['nhead'],
        ff_dim=config.get('ff_dim', config['embed_dim'] * 2),
        transformer_layers=config.get('transformer_layers', 1),
        attn_dim=config.get('attn_dim', config['embed_dim'] // 2),
        dropout=config['dropout'],
        k_sample=config['k_sample'],
        use_transformer=config.get('use_transformer', True),
        use_correlation=config.get('use_correlation', True),
        use_instance_clustering=config.get('use_instance_clustering', False),
        use_gated_attention=config.get('use_gated_attention', True),
        embedding_layers=config.get('embedding_layers', 2),
        use_contrastive_head=config.get('use_contrastive', False),
        contrastive_dim=config.get('contrastive_dim', 128),
        use_anchor_head=config.get('use_anchor', False),
        anchor_dim=config.get('anchor_dim', 32),
        use_hap=config.get('use_hap', False),
        hap_topk=config.get('hap_topk', 3),
        hap_use_temperature=config.get('hap_use_temperature', False),
    )
    return model.to(device)


def _init_model(model, config, dataset, actual_train_idx, device):
    """Initialize correlation head from data statistics."""
    if config.get('use_correlation', True):
        co_matrix = dataset.get_label_cooccurrence(actual_train_idx)
        model.label_head.init_correlation_from_data(co_matrix.to(device))


def run_fold(fold_idx, dataset_orig, train_idx, test_idx, config, device='cpu'):
    """Run a single fold of cross-validation."""
    print(f"\n{'='*60}", flush=True)
    print(f"Fold {fold_idx + 1}", flush=True)
    print(f"{'='*60}", flush=True)

    dataset = copy.deepcopy(dataset_orig)
    dataset.normalize_features(train_idx)

    # Split train into train/val (90/10)
    np.random.seed(fold_idx)
    train_arr = np.array(train_idx)
    np.random.shuffle(train_arr)
    val_size = max(len(train_arr) // 10, 1)
    val_idx = train_arr[:val_size].tolist()
    actual_train_idx = train_arr[val_size:].tolist()

    collate = collate_fn_dd if config.get('dataset_format', 'standard') == 'dd' else collate_fn

    sampler = dataset.get_balanced_sampler(actual_train_idx)
    train_loader = DataLoader(
        Subset(dataset, actual_train_idx), batch_size=config['batch_size'],
        sampler=sampler, collate_fn=collate, num_workers=0, drop_last=False)
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate, num_workers=0)

    # Multi-seed ensemble
    ensemble_seeds = config.get('ensemble_seeds', None)
    if ensemble_seeds and len(ensemble_seeds) > 1:
        return _run_fold_ensemble(
            fold_idx, dataset, actual_train_idx, val_idx, test_idx,
            train_loader, val_loader, test_loader, config, device, ensemble_seeds)

    # Single model training.
    # Seed this path explicitly. The ensemble path seeds per-seed below, but
    # without this the single-model path draws model init and the
    # WeightedRandomSampler from the unseeded global RNG, so two identical runs
    # differ by as much as 2.6 AP points. (Verified 2026-08-16; no published
    # result cell uses this path.) The np.random.seed(fold_idx) above governs the
    # train/val split and deliberately still runs first, so splits are unchanged.
    single_seed = int(config.get('single_model_seed', 42)) + fold_idx
    torch.manual_seed(single_seed)
    np.random.seed(single_seed)

    model = create_model(config, device)
    _init_model(model, config, dataset, actual_train_idx, device)

    trainer = EnhancedTrainer(model, config, device)

    loss_name = "ASL" if config.get('use_asymmetric_loss') else "BCE"
    mixup_str = " + Mixup" if config.get('use_mixup') else ""
    print(f"Phase 1: Independent label training ({loss_name}{mixup_str})...", flush=True)
    trainer.train_phase1(
        train_loader, val_loader,
        num_epochs=config.get('phase1_epochs', 80),
        lr=config.get('phase1_lr', 1e-3))

    print(f"Phase 2: Joint training ({loss_name}{mixup_str} + SWA)...", flush=True)
    trainer.train_phase2(
        train_loader, val_loader,
        max_epochs=config.get('phase2_max_epochs', 120),
        lr=config.get('phase2_lr', 3e-4),
        patience=config.get('early_stop_patience', 25))

    if ATTENTION_COLLECTOR is not None:
        ATTENTION_COLLECTOR(fold_idx, None, model, test_loader,
                            dataset, test_idx, device)

    use_threshold_opt = config.get('use_threshold_opt', False)

    if use_threshold_opt:
        val_labels, val_raw, val_coupled = trainer.get_predictions(val_loader)
        test_labels, test_raw, test_coupled = trainer.get_predictions(test_loader)

        # Pick raw or coupled based on val AP
        val_ap_raw = AP_fn(val_labels, val_raw)
        val_ap_coupled = AP_fn(val_labels, val_coupled)

        if val_ap_coupled >= val_ap_raw:
            val_probs, test_probs = val_coupled, test_coupled
            source = "coupled"
        else:
            val_probs, test_probs = val_raw, test_raw
            source = "raw"

        thresholds = optimize_thresholds(val_labels, val_probs, metric='hamming')
        print(f"  Optimized thresholds ({source}): {thresholds.round(3)}", flush=True)
        test_metrics = AllFiveWithThresholds(test_labels, test_probs, thresholds)
    else:
        test_metrics = trainer.evaluate(test_loader)

    print(f"\nFold {fold_idx + 1} Results:", flush=True)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}", flush=True)
    return test_metrics


def _run_fold_ensemble(fold_idx, dataset, actual_train_idx, val_idx, test_idx,
                       train_loader, val_loader, test_loader, config, device, seeds):
    """Run multiple seeds and average predictions for ensemble."""
    print(f"  Running {len(seeds)}-seed ensemble...", flush=True)

    all_test_probs = []
    all_val_probs = []
    test_labels = None
    val_labels = None

    loss_name = "ASL" if config.get('use_asymmetric_loss') else "BCE"
    mixup_str = " + Mixup" if config.get('use_mixup') else ""

    for seed_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = create_model(config, device)
        _init_model(model, config, dataset, actual_train_idx, device)
        trainer = EnhancedTrainer(model, config, device)

        print(f"  Seed {seed} ({seed_idx+1}/{len(seeds)}) [{loss_name}{mixup_str}]:", flush=True)
        trainer.train_phase1(
            train_loader, val_loader,
            num_epochs=config.get('phase1_epochs', 80),
            lr=config.get('phase1_lr', 1e-3))
        trainer.train_phase2(
            train_loader, val_loader,
            max_epochs=config.get('phase2_max_epochs', 120),
            lr=config.get('phase2_lr', 3e-4),
            patience=config.get('early_stop_patience', 25))

        if ATTENTION_COLLECTOR is not None:
            ATTENTION_COLLECTOR(fold_idx, seed, model, test_loader,
                                dataset, test_idx, device)

        vl, vr, vc = trainer.get_predictions(val_loader)
        tl, tr, tc = trainer.get_predictions(test_loader)

        if AP_fn(vl, vc) >= AP_fn(vl, vr):
            all_val_probs.append(vc)
            all_test_probs.append(tc)
        else:
            all_val_probs.append(vr)
            all_test_probs.append(tr)

        val_labels = vl
        test_labels = tl

    # Average probabilities across seeds
    val_probs = np.mean(all_val_probs, axis=0)
    test_probs = np.mean(all_test_probs, axis=0)

    use_threshold_opt = config.get('use_threshold_opt', False)
    if use_threshold_opt:
        thresholds = optimize_thresholds(val_labels, val_probs, metric='hamming')
        print(f"  Ensemble thresholds: {thresholds.round(3)}", flush=True)
        test_metrics = AllFiveWithThresholds(test_labels, test_probs, thresholds)
    else:
        test_metrics = AllFive(test_labels, test_probs)

    print(f"\nFold {fold_idx + 1} Ensemble Results:", flush=True)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}", flush=True)
    return test_metrics


def run_cv(config, device='cpu', num_folds=None):
    """Run cross-validation."""
    dataset_format = config.get('dataset_format', 'standard')

    if dataset_format == 'dd':
        dataset = MIMLDatasetDD(mat_path=config['data_path'])
    else:
        dataset = MIMLDatasetCV(
            mat_path=config['data_path'],
            bag_key=config.get('bag_key', 'bags'),
            label_key=config.get('label_key', 'labels'),
        )

    config['input_dim'] = dataset.input_dim
    config['num_labels'] = dataset.num_labels

    print(f"Dataset: {len(dataset)} samples, {dataset.num_labels} labels, "
          f"{dataset.input_dim}-dim features", flush=True)

    # CV splits
    if 'cv_mat_path' in config and config['cv_mat_path']:
        splits = get_cv_splits(config['cv_mat_path'], len(dataset), 10)
    else:
        k = num_folds or 5
        indices = np.arange(len(dataset))
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = len(dataset) // k
        splits = []
        for i in range(k):
            test_idx = indices[i * fold_size:(i + 1) * fold_size].tolist()
            train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]]).tolist()
            splits.append((train_idx, test_idx))

    if num_folds is not None:
        splits = splits[:num_folds]

    all_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_metrics = run_fold(fold_idx, dataset, train_idx, test_idx, config, device)
        all_metrics.append(fold_metrics)

    # Aggregate results
    n = len(splits)
    print(f"\n{'='*60}", flush=True)
    print(f"{n}-Fold CV Results (mean +/- std)", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}
    for name in all_metrics[0].keys():
        values = [m[name] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        results[name] = (mean, std)
        print(f"  {name}: {mean:.4f} +/- {std:.4f}", flush=True)

    return results, all_metrics

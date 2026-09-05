"""Cross-Validation training loop for MIML-CLAM++.

Supports:
  - Standard format datasets (scene.mat, reuters_MIML.mat) with 10CV.mat splits
  - DD format datasets (MSCV2, letter_frost, letter_carroll) with random K-fold
  - Per-label threshold optimization
  - Multi-seed ensemble (average probabilities across seeds)
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
from .trainer import CurriculumTrainer


def create_model(config, device='cpu', model_class=None):
    """Create model from config dict.

    Args:
        config: Hyperparameter dict.
        device: Target device.
        model_class: Model class to instantiate. None defaults to MIML_CLAM.
    """
    if model_class is None:
        model_class = MIML_CLAM
    model = model_class(
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
    )
    return model.to(device)


def _init_model(model, config, dataset, actual_train_idx, device):
    """Initialize correlation head from data statistics (if model supports it)."""
    if config.get('use_correlation', True) and hasattr(model, 'label_head'):
        co_matrix = dataset.get_label_cooccurrence(actual_train_idx)
        model.label_head.init_correlation_from_data(co_matrix.to(device))


def run_fold(fold_idx, dataset_orig, train_idx, test_idx, config, device='cpu',
             model_class=None):
    """Run a single fold of cross-validation.

    Supports threshold optimization and multi-seed ensemble.
    """
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
            train_loader, val_loader, test_loader, config, device, ensemble_seeds,
            model_class=model_class)

    # Single model training
    model = create_model(config, device, model_class=model_class)
    _init_model(model, config, dataset, actual_train_idx, device)

    trainer = CurriculumTrainer(model, config, device)

    print("Phase 1: Independent label training...", flush=True)
    trainer.train_phase1(
        train_loader, val_loader,
        num_epochs=config.get('phase1_epochs', 80),
        lr=config.get('phase1_lr', 1e-3))

    print("Phase 2: Joint training...", flush=True)
    trainer.train_phase2(
        train_loader, val_loader,
        max_epochs=config.get('phase2_max_epochs', 120),
        lr=config.get('phase2_lr', 3e-4),
        patience=config.get('early_stop_patience', 25))

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
                       train_loader, val_loader, test_loader, config, device, seeds,
                       model_class=None):
    """Run multiple seeds and average predictions for ensemble."""
    print(f"  Running {len(seeds)}-seed ensemble...", flush=True)

    all_test_probs = []
    all_val_probs = []
    test_labels = None
    val_labels = None

    for seed_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = create_model(config, device, model_class=model_class)
        _init_model(model, config, dataset, actual_train_idx, device)
        trainer = CurriculumTrainer(model, config, device)

        print(f"  Seed {seed} ({seed_idx+1}/{len(seeds)}):", flush=True)
        trainer.train_phase1(
            train_loader, val_loader,
            num_epochs=config.get('phase1_epochs', 80),
            lr=config.get('phase1_lr', 1e-3))
        trainer.train_phase2(
            train_loader, val_loader,
            max_epochs=config.get('phase2_max_epochs', 120),
            lr=config.get('phase2_lr', 3e-4),
            patience=config.get('early_stop_patience', 25))

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


def run_cv(config, device='cpu', num_folds=None, model_class=None):
    """Run cross-validation.

    Supports both standard (.mat with 10CV.mat) and DD format datasets.
    """
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
        fold_metrics = run_fold(fold_idx, dataset, train_idx, test_idx, config, device,
                               model_class=model_class)
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

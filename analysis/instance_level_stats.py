"""Dataset-level statistics for the instance-level attribution study.

Reports, per dataset with per-instance ground truth:
  - bag / instance / label counts and mean labels per bag
  - OR-consistency (bag labels == OR of instance labels) — the assumption that
    makes instance-level localization well posed
  - witness disjointness across co-occurring label pairs
  - chance precision@1 (what uniform attention gets)
  - the exact, tight precision@1 ceiling for any label-agnostic method

Run:
    python3 analysis/instance_level_stats.py
    python3 analysis/instance_level_stats.py --json results/instance_level/dataset_stats.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miml_clam.data.miml_dataset_dd import MIMLDatasetDD          # noqa: E402
from miml_clam.metrics.instance_metrics import (                   # noqa: E402
    check_or_consistency,
    chance_precision_at_1,
    label_agnostic_ceiling,
    witness_disjointness,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'data', 'MIML-SELFVIT')

# Datasets whose .mat files carry the per-instance label matrix YR.
INSTANCE_LABELLED = ['MSCV2', 'letter_frost', 'letter_carroll']


def load_padded(name):
    """Load a DD dataset into padded (B, N, *) arrays plus a padding mask."""
    ds = MIMLDatasetDD(os.path.join(DATA_DIR, f'{name}.mat'))
    if not ds.has_instance_labels:
        raise ValueError(f'{name} has no instance labels')

    B = len(ds)
    L = ds.num_labels
    N = max(ds.num_instances)

    inst = np.zeros((B, N, L), dtype=bool)
    mask = np.ones((B, N), dtype=bool)          # True = padded
    bag = np.zeros((B, L), dtype=bool)

    for i in range(B):
        n = ds.num_instances[i]
        inst[i, :n] = ds.get_instance_labels(i).numpy() > 0
        mask[i, :n] = False
        bag[i] = ds.labels[i].numpy() > 0

    return ds, inst, bag, mask


def stats_for(name):
    ds, inst, bag, mask = load_padded(name)
    valid = ~mask
    n_inst = int(valid.sum())
    labels_per_bag = bag.sum(axis=1)
    inst_per_bag = valid.sum(axis=1)

    consistency, uncovered = check_or_consistency(inst, bag, mask)

    return {
        'dataset': name,
        'bags': int(len(ds)),
        'instances': n_inst,
        'labels': int(ds.num_labels),
        'features': int(ds.input_dim),
        'mean_instances_per_bag': float(inst_per_bag.mean()),
        'mean_labels_per_bag': float(labels_per_bag.mean()),
        'pct_multilabel_bags': float(100.0 * (labels_per_bag >= 2).mean()),
        'or_consistency': consistency,
        'bag_labels_without_witness': uncovered,
        'witness_disjointness': witness_disjointness(inst, bag, mask),
        'chance_p_at_1': chance_precision_at_1(inst, bag, mask),
        'label_agnostic_ceiling': label_agnostic_ceiling(inst, bag, mask),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None, help='write results to this path')
    ap.add_argument('--datasets', default=','.join(INSTANCE_LABELLED))
    args = ap.parse_args()

    rows = [stats_for(n) for n in args.datasets.split(',') if n]

    hdr = (f"{'dataset':16s} {'bags':>6s} {'inst':>7s} {'L':>4s} "
           f"{'lbl/bag':>8s} {'OR-cons':>8s} {'disjoint':>9s} "
           f"{'chance':>8s} {'CEILING':>8s}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['dataset']:16s} {r['bags']:6d} {r['instances']:7d} "
              f"{r['labels']:4d} {r['mean_labels_per_bag']:8.2f} "
              f"{r['or_consistency']:8.3f} "
              f"{100 * r['witness_disjointness']:8.1f}% "
              f"{100 * r['chance_p_at_1']:7.1f}% "
              f"{100 * r['label_agnostic_ceiling']:7.1f}%")
    print()
    print('CEILING = exact, tight upper bound on instance precision@1 for any')
    print('          label-agnostic method (one attention shared across labels).')

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()

"""Dataset loader for DD-format MIML .mat files (MSCV2, letter_frost, letter_carroll).

DD format:
  X: (total_instances, features) — all instance features
  XtB: (total_instances, 1) — bag assignment
  B: (num_bags, 1) — bag IDs
  YB: (num_bags, labels) — bag-level labels
"""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from scipy.io import loadmat
import numpy as np
import os


class MIMLDatasetDD(Dataset):
    """MIML dataset from DD-format .mat files."""

    def __init__(self, mat_path):
        super().__init__()
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Data file not found: {mat_path}")

        self.bags = []
        self.labels = []
        self.num_instances = []
        self.inst_labels = []   # (N_i, L) per bag; empty if the file has no YR

        self._load_dd_format(mat_path)
        self.num_labels = self.labels[0].shape[0]
        self.input_dim = self.bags[0].shape[1]

    @property
    def has_instance_labels(self):
        """True when per-instance ground truth (YR) was found in the .mat."""
        return len(self.inst_labels) == len(self.bags) and len(self.bags) > 0

    def get_instance_labels(self, idx):
        """(N_i, L) binary instance-label matrix for bag `idx`."""
        if not self.has_instance_labels:
            raise ValueError(
                'This dataset has no per-instance labels (no YR in the .mat).')
        return self.inst_labels[idx]

    def _load_dd_format(self, mat_path):
        mat = loadmat(mat_path)

        if 'DD' in mat:
            dd = mat['DD'][0, 0]
            src = dd
        else:
            src = mat

        X = src['X'].astype(np.float32)
        XtB = src['XtB'].flatten()
        YB = src['YB'].astype(np.float32)
        B = src['B'].flatten()

        # Per-instance labels. Present in the DD-format files as `YR` but
        # historically never read (the loader used only the bag labels YB).
        # They are the ground truth for instance-level attribution evaluation.
        YR = None
        try:
            names = src.dtype.names
        except AttributeError:
            names = None
        if names is not None and 'YR' in names:
            YR = np.asarray(src['YR']).astype(np.float32)
        elif isinstance(src, dict) and 'YR' in src:
            YR = np.asarray(src['YR']).astype(np.float32)

        for i, bag_id in enumerate(B):
            inst_mask = (XtB == bag_id)
            bag_features = X[inst_mask]

            if bag_features.shape[0] == 0:
                continue

            bag_tensor = torch.tensor(bag_features, dtype=torch.float32)
            label = YB[i]
            label = (label > 0).astype(np.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)

            self.bags.append(bag_tensor)
            self.labels.append(label_tensor)
            self.num_instances.append(bag_tensor.shape[0])

            if YR is not None:
                self.inst_labels.append(
                    torch.tensor((YR[inst_mask] > 0).astype(np.float32))
                )

    def normalize_features(self, train_indices):
        """Standardize features using training set statistics."""
        all_instances = []
        for i in train_indices:
            all_instances.append(self.bags[i])
        all_instances = torch.cat(all_instances, dim=0)
        self.feat_mean = all_instances.mean(dim=0)
        self.feat_std = all_instances.std(dim=0).clamp(min=1e-6)

        for i in range(len(self.bags)):
            self.bags[i] = (self.bags[i] - self.feat_mean) / self.feat_std

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx]

    def get_label_cooccurrence(self, indices=None):
        """Compute label co-occurrence matrix."""
        if indices is None:
            indices = range(len(self))
        labels = torch.stack([self.labels[i] for i in indices])
        co = labels.T @ labels
        counts = labels.sum(dim=0).clamp(min=1)
        co = co / counts.unsqueeze(1)
        co.fill_diagonal_(0)
        return co

    def get_balanced_sampler(self, indices):
        """Create WeightedRandomSampler for balanced label combinations."""
        labels = torch.stack([self.labels[i] for i in indices])
        combo_strings = [''.join(str(int(x)) for x in row) for row in labels.numpy()]
        from collections import Counter
        combo_counts = Counter(combo_strings)
        weights = [1.0 / combo_counts[s] for s in combo_strings]
        weights = torch.tensor(weights, dtype=torch.float64)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def collate_fn_dd(batch):
    """Collate bags of variable length with padding. mask: 1=padded, 0=real."""
    bags, labels = zip(*batch)
    max_len = max(bag.shape[0] for bag in bags)
    dim = bags[0].shape[1]

    padded_bags = torch.zeros((len(bags), max_len, dim))
    mask = torch.ones((len(bags), max_len))

    for i, bag in enumerate(bags):
        n = bag.shape[0]
        padded_bags[i, :n] = bag
        mask[i, :n] = 0

    labels = torch.stack(labels)
    return padded_bags, labels, mask

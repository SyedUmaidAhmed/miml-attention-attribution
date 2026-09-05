"""Bag-level Mixup augmentation for MIML.

Provides:
  - bag_mixup: original interpolation/concatenation Mixup (random partner).
  - psemix:    similarity-driven pseudo-bag pairing for variable-size bags
               (Liu et al., PseMix, IEEE TMI 2024). Restricted to the current
               minibatch to prevent cross-fold leakage. For fixed-size bags
               psemix falls back to standard interpolation Mixup.
"""

import torch
import numpy as np


def bag_mixup(bags, labels, mask, alpha=0.2, fixed_size=True):
    """Apply bag-level mixup augmentation.

    For fixed-size bags (Scene): interpolate instance features, soft labels.
    For variable-length bags (Reuters): concatenate instances, soft labels.

    Args:
        bags: (B, N, D)
        labels: (B, L)
        mask: (B, N) — True where padded
        alpha: Beta distribution parameter
        fixed_size: If True, all bags have same # instances (interpolate mode)

    Returns:
        mixed_bags, mixed_labels, mixed_mask
    """
    B = bags.size(0)
    if B < 2:
        return bags, labels, mask

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    lam = max(lam, 1 - lam)
    indices = torch.randperm(B, device=bags.device)

    if fixed_size:
        mixed_bags = lam * bags + (1 - lam) * bags[indices]
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        mixed_mask = mask
    else:
        mixed_bags_list = []
        mixed_mask_list = []
        mixed_labels = lam * labels + (1 - lam) * labels[indices]

        for i in range(B):
            j = indices[i].item()
            valid_i = (~mask[i].bool()).sum().item()
            valid_j = (~mask[j].bool()).sum().item()

            bag_i = bags[i, :valid_i]
            bag_j = bags[j, :valid_j]

            combined = torch.cat([bag_i, bag_j], dim=0)
            mixed_bags_list.append(combined)
            mixed_mask_list.append(torch.zeros(combined.shape[0], device=bags.device))

        max_len = max(b.shape[0] for b in mixed_bags_list)
        D = bags.shape[2]
        mixed_bags = torch.zeros(B, max_len, D, device=bags.device)
        mixed_mask = torch.ones(B, max_len, device=bags.device)

        for i, (b, m) in enumerate(zip(mixed_bags_list, mixed_mask_list)):
            n = b.shape[0]
            mixed_bags[i, :n] = b
            mixed_mask[i, :n] = 0

    return mixed_bags, mixed_labels, mixed_mask


def _psemix_select_partners(bags, labels, mask, K=4, share_label_bias=0.7):
    """Choose a partner index for each bag in the minibatch.

    With probability `share_label_bias`, prefer partners that share at least
    one label and lie among the K most-similar bags in mean-feature space.
    Otherwise pick uniformly among the K most-similar bags. Pool is the
    current minibatch only — no cross-batch / cross-fold leakage.
    """
    B = bags.size(0)
    device = bags.device
    if B < 2:
        return torch.arange(B, device=device)

    valid = (~mask.bool()).float().unsqueeze(-1)              # (B, N, 1)
    counts = valid.sum(dim=1).clamp(min=1.0)                  # (B, 1)
    bag_means = (bags * valid).sum(dim=1) / counts            # (B, D)
    bm = torch.nn.functional.normalize(bag_means, dim=1)
    sim = bm @ bm.t()                                          # (B, B)
    sim.fill_diagonal_(-float('inf'))

    K_eff = min(K, B - 1)
    topk = sim.topk(K_eff, dim=1).indices                     # (B, K_eff)

    y_hard = (labels > 0.5).float()
    overlap = (y_hard @ y_hard.t()) > 0                       # (B, B), bool
    overlap.fill_diagonal_(False)

    partner = torch.empty(B, dtype=torch.long, device=device)
    for i in range(B):
        cand = topk[i]                                         # (K_eff,)
        if torch.rand((), device=device).item() < share_label_bias:
            mask_share = overlap[i, cand]
            if mask_share.any():
                pool = cand[mask_share]
            else:
                pool = cand
        else:
            pool = cand
        pick = torch.randint(0, pool.numel(), (1,), device=device).item()
        partner[i] = pool[pick]

    return partner


def psemix(bags, labels, mask, alpha=0.2, fixed_size=False,
           K=4, share_label_bias=0.7):
    """Similarity-driven pseudo-bag Mixup.

    For variable-size bags, picks a similarity-aware partner (PseMix), then
    concatenates instances and mixes labels. For fixed-size bags, falls back
    to the existing element-wise interpolation in `bag_mixup`.
    """
    B = bags.size(0)
    if B < 2:
        return bags, labels, mask

    if fixed_size:
        # PseMix is a partner-selection idea; element-wise interpolation
        # already yields a valid mixed bag for fixed-size data.
        return bag_mixup(bags, labels, mask, alpha=alpha, fixed_size=True)

    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    lam = max(lam, 1.0 - lam)

    partner = _psemix_select_partners(
        bags, labels, mask, K=K, share_label_bias=share_label_bias
    )

    mixed_labels = lam * labels + (1.0 - lam) * labels[partner]

    mixed_bags_list = []
    for i in range(B):
        j = int(partner[i].item())
        valid_i = int((~mask[i].bool()).sum().item())
        valid_j = int((~mask[j].bool()).sum().item())
        bag_i = bags[i, :valid_i]
        bag_j = bags[j, :valid_j]
        mixed_bags_list.append(torch.cat([bag_i, bag_j], dim=0))

    max_len = max(b.shape[0] for b in mixed_bags_list)
    D = bags.shape[2]
    mixed_bags = torch.zeros(B, max_len, D, device=bags.device, dtype=bags.dtype)
    mixed_mask = torch.ones(B, max_len, device=bags.device, dtype=mask.dtype)
    for i, b in enumerate(mixed_bags_list):
        n = b.shape[0]
        mixed_bags[i, :n] = b
        mixed_mask[i, :n] = 0

    return mixed_bags, mixed_labels, mixed_mask

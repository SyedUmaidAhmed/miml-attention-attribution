"""Jaccard-weighted Multi-Label Supervised Contrastive Loss (MulSupCon).

Zhang & Wu, "Multi-Label Supervised Contrastive Learning", AAAI 2024.
For a batch of L2-normalised embeddings v_i and multi-hot labels y_i, pairs
(i, j) are weighted by Jaccard overlap w_{ij} = |y_i ∩ y_j| / |y_i ∪ y_j|;
the loss pulls together pairs with w_{ij} > 0 and pushes apart all others.

Returns 0 when the batch contains no positive pair (e.g. small-batch
regimes, or when label sets are pairwise disjoint).
"""

import torch
import torch.nn as nn


class JaccardSupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, d) — L2-normalised contrastive embeddings.
            labels: (B, L) — soft or hard multi-hot labels in [0, 1].

        Returns:
            scalar loss (float32). Zero if no in-batch positive pair.
        """
        # Force float32 for the contrastive sub-graph (Apple MPS NaN guard).
        v = embeddings.float()
        y = (labels > 0.5).float()

        B = v.shape[0]
        if B < 2:
            return torch.zeros((), device=v.device, dtype=v.dtype)

        # Jaccard weights w_{ij} for i != j.
        intersect = y @ y.t()                        # (B, B)
        sums = y.sum(dim=1, keepdim=True)            # (B, 1)
        union = sums + sums.t() - intersect          # (B, B)
        w = torch.where(union > 0, intersect / union, torch.zeros_like(union))
        eye = torch.eye(B, device=v.device, dtype=w.dtype)
        w = w * (1.0 - eye)                          # zero diagonal

        if w.sum() < 1e-6:
            return torch.zeros((), device=v.device, dtype=v.dtype)

        # Cosine similarities (v already L2-normalised, but normalise again
        # defensively for numerical stability).
        v = nn.functional.normalize(v, dim=1)
        sim = v @ v.t() / self.temperature           # (B, B)

        # Numerical stabilisation: subtract row-wise max.
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # log-prob_{ij} = sim_{ij} - log Σ_{k != i} exp(sim_{ik})
        mask_self = 1.0 - eye
        exp_sim = torch.exp(sim) * mask_self
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-12))
        log_prob = sim - log_denom                   # (B, B)

        # Weighted sum over positive pairs.
        per_anchor_w = w.sum(dim=1).clamp(min=1e-12)
        weighted = (w * log_prob).sum(dim=1) / per_anchor_w
        # Loss = -mean over anchors that have at least one positive pair.
        anchor_has_pos = (w.sum(dim=1) > 0).float()
        if anchor_has_pos.sum() < 1:
            return torch.zeros((), device=v.device, dtype=v.dtype)
        loss = -(weighted * anchor_has_pos).sum() / anchor_has_pos.sum()
        return loss

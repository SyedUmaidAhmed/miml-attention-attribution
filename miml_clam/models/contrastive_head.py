"""Contrastive projection head for MIML-ALFA++.

A 2-layer MLP that takes the per-label bag representation
bag_repr ∈ ℝ^{B×L×D}, mean-pools over the label axis, projects to a
contrastive embedding dimension, and L2-normalises.
"""

import torch
import torch.nn as nn


class ContrastiveHead(nn.Module):
    def __init__(self, embed_dim, proj_dim=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, bag_repr):
        """
        Args:
            bag_repr: (B, L, D) — per-label bag representations from
                LabelConditionalGatedAttention.

        Returns:
            (B, proj_dim) L2-normalised embeddings.
        """
        x = bag_repr.mean(dim=1)              # (B, D)
        v = self.net(x)
        return nn.functional.normalize(v, dim=1)

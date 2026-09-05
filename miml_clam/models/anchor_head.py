"""Label Anchor Alignment (LAA) head.

Novel auxiliary classifier that gives the per-label query embeddings a
*dual role*: in addition to driving label-focused attention they serve as
semantic anchors in the instance-feature space.

For each (bag i, label l) the existing label-focused attention produces a
bag representation r_{i,l} = sum_j a_{l,j} h_{i,j}. The LAA head projects
both r_{i,l} and the label embedding e_l into a shared anchor subspace,
L2-normalises, and computes a cosine-similarity-based logit:

    s_{i,l} = tau * cos(W_a r_{i,l}, W_a e_l) + b

A standard BCE-with-logits loss is then applied over s_{i,l} versus the
multi-label targets y_{i,l}. The result is an alternative classifier that
is grounded in the *same* embedding e_l that already drives attention,
which couples the attention pathway and the classifier pathway through
the shared anchor.

This is distinct from:

  - CLAM-style instance-clustering losses, which require pseudo-instance
    labels via top-k positive / bottom-k negative sampling.
  - Bag-level supervised contrastive losses (MulSupCon and friends),
    which contrast bags against bags but do not give labels their own
    semantic anchors in feature space.
  - Query2Label-style cross-attention heads, which use label queries
    only as attention seeds, never as classification anchors.
"""

import torch
import torch.nn as nn


class LabelAnchorHead(nn.Module):
    """Cosine-similarity classifier head that aligns per-label bag
    representations to the same label embedding that drives attention.

    Args:
        embed_dim:   instance / bag-rep feature dimension D.
        num_labels:  number of labels L (used only for logging).
        anchor_dim:  dimension of the shared anchor subspace.
        init_tau:    initial value of the learnable temperature.
        init_bias:   initial value of the learnable bias term.
        dropout:     dropout applied before the projection.
    """

    def __init__(self, embed_dim, num_labels, anchor_dim=32,
                 init_tau=5.0, init_bias=0.0, dropout=0.0):
        super().__init__()
        self.num_labels = int(num_labels)
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, anchor_dim, bias=False),
        )
        self.log_tau = nn.Parameter(torch.tensor(float(init_tau)).log())
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, bag_repr, label_anchors):
        """
        Args:
            bag_repr:      (B, L, D) — per-label bag representation.
            label_anchors: (L, D)   — label embeddings (shared with the
                           attention module).

        Returns:
            anchor_logits: (B, L) — alignment-classifier logits suitable
                           for BCEWithLogitsLoss.
        """
        u = self.proj(bag_repr)                              # (B, L, A)
        v = self.proj(label_anchors)                         # (L, A)
        u = nn.functional.normalize(u, dim=-1)
        v = nn.functional.normalize(v, dim=-1)
        # cos sim per (b, l): take diagonal in the L axis.
        cos_sim = (u * v.unsqueeze(0)).sum(dim=-1)           # (B, L)
        logits = self.log_tau.exp() * cos_sim + self.bias
        return logits

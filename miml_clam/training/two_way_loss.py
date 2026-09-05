"""Two-Way Multi-Label Loss (ASL2).

Kobayashi, "Two-Way Multi-Label Loss", CVPR 2023. Extends ASL with a
relative-margin term that pushes the smallest positive logit above the
largest negative logit by a margin m. Returns vanilla-ASL behaviour when
lam_rel = 0.

Drop-in for AsymmetricLoss: same forward(logits, targets) -> scalar.
"""

import torch
import torch.nn as nn

from .asymmetric_loss import AsymmetricLoss


class TwoWayMLLoss(nn.Module):
    """ASL plus a per-bag relative-margin penalty between min-positive
    and max-negative sigmoid scores.

    Args:
        gamma_neg, gamma_pos, clip: forwarded to the underlying ASL.
        margin (m): target gap between min-positive and max-negative.
        lam_rel: weight of the relative-margin term.
    """

    def __init__(self, gamma_neg=2, gamma_pos=0, clip=0.02,
                 margin=0.05, lam_rel=0.5):
        super().__init__()
        self.asl = AsymmetricLoss(gamma_neg=gamma_neg,
                                  gamma_pos=gamma_pos,
                                  clip=clip)
        self.margin = float(margin)
        self.lam_rel = float(lam_rel)

    def forward(self, logits, targets):
        loss = self.asl(logits, targets)
        if self.lam_rel <= 0:
            return loss

        # Relative-margin term: per bag, encourage
        # min_{l in P} sigmoid(z_l) - max_{l in N} sigmoid(z_l) >= m.
        probs = torch.sigmoid(logits)
        # Treat soft labels (e.g., from label smoothing / mixup) as positive
        # iff > 0.5 for margin computation. Bags with no positive or no
        # negative labels contribute zero margin loss.
        is_pos = targets > 0.5
        is_neg = ~is_pos

        very_small = torch.tensor(1.0, device=logits.device, dtype=probs.dtype)
        very_large = torch.tensor(0.0, device=logits.device, dtype=probs.dtype)

        pos_scores = torch.where(is_pos, probs, very_small)  # mask non-pos to 1
        neg_scores = torch.where(is_neg, probs, very_large)  # mask non-neg to 0

        min_pos = pos_scores.min(dim=1).values  # (B,)
        max_neg = neg_scores.max(dim=1).values  # (B,)

        # Only count bags that have at least one positive and one negative.
        valid = is_pos.any(dim=1) & is_neg.any(dim=1)
        if not valid.any():
            return loss

        gap = min_pos - max_neg
        margin_loss = torch.relu(self.margin - gap).clamp(max=1.0)
        margin_loss = (margin_loss * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

        return loss + self.lam_rel * margin_loss

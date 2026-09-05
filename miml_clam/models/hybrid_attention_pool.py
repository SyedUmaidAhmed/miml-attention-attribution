"""Hybrid Attention Pooling (HAP) — a per-label learnable mixture of soft
softmax-weighted attention and hard top-k attention pooling.

Motivation: in MIML, different labels have different witness structures.
Some labels are supported by a single key instance ("witness-based" — top-k
hard attention is appropriate). Others are supported by diffuse evidence
across many instances ("diffuse" — soft softmax pooling is appropriate).
Standard label-focused gated attention always uses soft pooling.

HAP adds exactly L parameters (one mixture coefficient per label) and zero
gradient-signal terms beyond the existing classifier loss, making it
orthogonal to ASL/Mixup/SupCon-style regularizers (which empirically
over-regularise small variable-bag MIML data — see Section 6.4).

For label l, the bag representation is computed as
    r_{i,l} = alpha_l * sum_j a_soft_{l,j} h_j + (1 - alpha_l) * sum_j a_hard_{l,j} h_j
where alpha_l = sigmoid(beta_l) is the learnable mixture coefficient,
a_soft is the usual softmax over attention scores, and a_hard is a uniform
1/k mask over the top-k attention scores.
"""

import torch
import torch.nn as nn


class HybridAttentionPool(nn.Module):
    """Per-label soft-vs-hard attention mixer.

    Args:
        num_labels: number of labels L.
        topk:       number of instances per label that participate in the
                    hard branch (default 3).
        init_beta:  initial value of the per-label mixture logit. The
                    sigmoid of this is the soft-mix weight; default 0.0
                    => 50/50 mix at initialisation.
    """

    def __init__(self, num_labels, topk=3, init_beta=0.0,
                 use_temperature=False, init_log_tau=0.0):
        super().__init__()
        self.num_labels = int(num_labels)
        self.topk = int(topk)
        self.beta = nn.Parameter(torch.full((num_labels,), float(init_beta)))
        # HAP-Temperature extension (ICDM 2026): optional per-label learnable
        # temperature tau_l = exp(log_tau_l) on the soft branch. init log_tau=0
        # -> tau=1, recovering plain softmax exactly. Adds L parameters.
        self.use_temperature = bool(use_temperature)
        if self.use_temperature:
            self.log_tau = nn.Parameter(
                torch.full((num_labels,), float(init_log_tau)))

    def forward(self, attn_scores, h, mask=None):
        """
        Args:
            attn_scores: (B, L, N) — pre-softmax attention scores already
                with masked positions set to -inf.
            h:           (B, N, D) — instance representations.
            mask:        (B, N) — True where padded.

        Returns:
            bag_repr:    (B, L, D) — hybrid-pooled bag representation.
            soft_attn:   (B, L, N) — softmax weights (preserved for any
                downstream consumer that expected them).
            eff_attn:    (B, L, N) — the *effective* pooling distribution
                alpha_l * a_soft + (1 - alpha_l) * a_hard. By bilinearity of
                bmm this is exactly the distribution that produces bag_repr,
                and being a convex combination of two distributions over
                instances it is itself a valid distribution. This — not
                soft_attn — is the object to use for instance-level
                attribution, because soft_attn does not depend on alpha at
                all, so every alpha variant would otherwise look identical.
        """
        B, L, N = attn_scores.shape
        D = h.size(-1)

        # Soft attention via (optionally per-label temperature-scaled) softmax.
        if self.use_temperature:
            tau = torch.exp(self.log_tau).view(1, L, 1).clamp(min=0.1, max=10.0)
            # attn_scores has padded positions set to -inf. Dividing -inf by a
            # *learnable* tau makes the backward term (-attn_scores / tau^2)
            # evaluate to -inf at those positions, and -inf * 0 (the upstream
            # softmax grad there) = NaN, which poisons the whole network.
            # Replace -inf with a large finite negative before the division so
            # padded positions still receive ~zero softmax mass but contribute a
            # finite (zero) gradient to log_tau.
            safe_scores = attn_scores.masked_fill(
                torch.isneginf(attn_scores), -1e9)
            soft_attn = torch.softmax(safe_scores / tau, dim=-1)
        else:
            soft_attn = torch.softmax(attn_scores, dim=-1)        # (B, L, N)
        soft_repr = torch.bmm(soft_attn, h)                       # (B, L, D)

        # Hard attention: top-k positions per (bag, label) get equal mass.
        k_eff = min(self.topk, N)
        topk_idx = attn_scores.topk(k_eff, dim=-1).indices        # (B, L, k_eff)
        hard_attn = torch.zeros_like(attn_scores)                 # (B, L, N)
        hard_attn.scatter_(-1, topk_idx, 1.0 / float(k_eff))

        # If a top-k slot landed on padding, redistribute. (top-k from
        # -inf scores can still pick a padded position only if all
        # positions are padded; we guard against that with renormalisation.)
        if mask is not None:
            mask_b = mask.bool().unsqueeze(1)                     # (B, 1, N)
            hard_attn = hard_attn.masked_fill(mask_b, 0.0)
            hard_sum = hard_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            hard_attn = hard_attn / hard_sum

        hard_repr = torch.bmm(hard_attn, h)                       # (B, L, D)

        # Per-label sigmoid mixture coefficient.
        alpha = torch.sigmoid(self.beta).view(1, L, 1)            # (1, L, 1)
        bag_repr = alpha * soft_repr + (1.0 - alpha) * hard_repr  # (B, L, D)

        # Effective pooling distribution. Since bmm is linear in its first
        # argument, alpha*bmm(soft,h) + (1-alpha)*bmm(hard,h) == bmm(eff, h),
        # so eff_attn is exactly the attention that generated bag_repr.
        eff_attn = alpha * soft_attn + (1.0 - alpha) * hard_attn  # (B, L, N)

        return bag_repr, soft_attn, eff_attn

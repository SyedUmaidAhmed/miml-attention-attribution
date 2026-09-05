"""DSMIL baseline (Li et al., CVPR 2021).

"Dual-Stream Multiple Instance Learning Network for Whole Slide Image
Classification with Self-Supervised Contrastive Learning."

Architecture (multi-label adaptation):
  Per-label, two streams:
    - Stream 1 (instance / critic): each instance scored by a per-label
      linear classifier; the *max-scoring* instance is the critic.
    - Stream 2 (bag): query-key attention between every instance and the
      critic embedding gives an attention distribution; weighted sum gives
      the bag representation, which is scored by a per-label classifier.
  Final score per label = 0.5 * (s1_max + s2_bag).

For multi-label we use one critic per label (matches the per-label style of
the existing label-conditional attention but without learned label
embeddings). Reuses `InstanceEmbedding`. NO transformer (the original paper
uses raw SimCLR features).

Output contract matches `MIML_CLAM`:
    {'logits': (B,L), 'coupled_logits': (B,L), 'attn_weights': (B,L,N)}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from miml_clam.models.instance_embedding import InstanceEmbedding


class DSMIL(nn.Module):
    """Dual-stream MIL with per-label critics for multi-label classification."""

    # Tell EnhancedTrainer not to attempt any instance-clustering branches.
    use_instance_clustering = False

    def __init__(
        self,
        input_dim,
        num_labels,
        embed_dim=64,
        dropout=0.15,
        embedding_layers=2,
        attn_dim=32,
        **kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim

        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            num_layers=embedding_layers,
        )

        # Stream 1: per-label instance classifier (one logit per instance per label)
        # Implemented as a single Linear(D, L) for efficiency: row l scores label l.
        self.instance_classifier = nn.Linear(embed_dim, num_labels)

        # Stream 2: query / value projections for critic-vs-instance attention.
        # The critic embedding is the instance with the largest stream-1 score
        # for that label; queries come from the critic, keys/values from
        # every instance.
        self.q_proj = nn.Linear(embed_dim, attn_dim)
        self.k_proj = nn.Linear(embed_dim, attn_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Per-label bag classifier on stream-2 representation
        self.bag_classifier = nn.Linear(embed_dim, num_labels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        """Forward following Li et al. 2021 dual-stream pooling."""
        B, N, _ = bags.shape
        L = self.num_labels

        h = self.instance_embedding(bags)              # (B, N, D)

        # ---- Stream 1: per-instance scores -----------------------------
        inst_logits = self.instance_classifier(h)       # (B, N, L)

        if mask is not None:
            # Mask padded instances out of stream-1 max
            mfill = mask.bool().unsqueeze(-1)
            inst_logits_masked = inst_logits.masked_fill(mfill, float('-inf'))
        else:
            inst_logits_masked = inst_logits

        # max over instances per label  -> (B, L)
        s1, critic_idx = inst_logits_masked.max(dim=1)  # critic_idx: (B, L)

        # Gather the critic embeddings: (B, L, D)
        crit = torch.gather(
            h, dim=1,
            index=critic_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )  # (B, L, D)

        # ---- Stream 2: attention with critic queries -------------------
        q = self.q_proj(crit)                           # (B, L, attn_dim)
        k = self.k_proj(h)                              # (B, N, attn_dim)
        v = self.v_proj(h)                              # (B, N, D)

        scores = torch.bmm(q, k.transpose(1, 2)) / (self.attn_dim ** 0.5)  # (B, L, N)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).bool(), float('-inf'))
        attn = torch.softmax(scores, dim=-1)            # (B, L, N)
        attn = self.dropout(attn)

        bag_repr = torch.bmm(attn, v)                   # (B, L, D)

        # Per-label bag scoring: row l of bag_classifier on bag_repr[:, l, :]
        # bag_classifier.weight: (L, D), bias: (L,)
        s2 = (bag_repr * self.bag_classifier.weight.unsqueeze(0)).sum(dim=-1) \
             + self.bag_classifier.bias.unsqueeze(0)    # (B, L)

        # Final logits: average of the two streams (paper §3.2)
        logits = 0.5 * (s1 + s2)

        # Re-normalise attn_weights so each label's row sums to 1 with zeros
        # at padded positions (dropout above can alter the sum).
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).bool(), 0.0)
        denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn = attn / denom

        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': attn,
        }

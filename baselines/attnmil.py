"""AttnMIL baseline (Ilse et al., ICML 2018).

"Attention-based Deep Multiple Instance Learning."
Single-head (non-gated) attention pooling, then per-label linear classifiers
on the SAME bag representation.

Multi-label adaptation:
  1. One global attention distribution (single-head, NOT label-conditional).
  2. The same pooled bag representation is shared across all L labels.
  3. L independent linear heads on top.

NO transformer, NO gated mechanism (use the gated variant in `gated_attn.py`).

Output contract matches `MIML_CLAM`:
    {'logits': (B,L), 'coupled_logits': (B,L), 'attn_weights': (B,L,N)}
"""

import torch
import torch.nn as nn

from miml_clam.models.instance_embedding import InstanceEmbedding


class AttnMIL(nn.Module):
    """ICML 2018 attention-MIL with multi-label heads.

    Attention:  a_n = softmax(w^T tanh(V h_n)) over instances.
    Pooling:    z   = sum_n a_n * h_n
    Heads:      one nn.Linear(embed_dim, 1) per label (multi-label adaptation).
    """

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

        # Reused instance encoder (no transformer)
        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            num_layers=embedding_layers,
        )

        # Single-head attention (eq. 8 in Ilse et al. 2018)
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_w = nn.Linear(attn_dim, 1)

        # L independent linear classifiers on the shared bag representation.
        # The paper uses a single linear head; we extend to L for multi-label.
        self.classifiers = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_labels)
        ])

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        """
        bags: (B, N, input_dim) padded
        mask: (B, N) True where padded
        Returns dict with logits, coupled_logits, attn_weights.
        """
        h = self.instance_embedding(bags)               # (B, N, D)

        # Attention scores
        a = self.attention_V(h)                          # (B, N, attn_dim)
        scores = self.attention_w(a).squeeze(-1)         # (B, N)

        if mask is not None:
            scores = scores.masked_fill(mask.bool(), float('-inf'))
        attn = torch.softmax(scores, dim=-1)             # (B, N)

        # Single bag representation, shared across labels
        bag_repr = torch.bmm(attn.unsqueeze(1), h).squeeze(1)  # (B, D)

        # Per-label classification on the shared representation
        logits = torch.cat([clf(bag_repr) for clf in self.classifiers], dim=1)  # (B, L)

        # Expand attention to (B, L, N) by broadcasting (every label sees the
        # same distribution — matches the ABMILML pattern in baselines.py).
        attn_expanded = attn.unsqueeze(1).expand(-1, self.num_labels, -1).contiguous()

        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': attn_expanded,
        }

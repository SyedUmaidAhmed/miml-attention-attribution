"""Gated-Attention MIL baseline (Ilse et al., ICML 2018, gated variant).

"Attention-based Deep Multiple Instance Learning" eq. 9: gated attention with
two parallel branches whose outputs are multiplied element-wise.

    a_n  proportional to  w^T (tanh(V h_n) odot sigmoid(U h_n))

Multi-label adaptation:
  1. ONE global attention distribution (not label-conditional).
  2. Shared bag representation across labels.
  3. L independent linear classifiers on top.

NO transformer (the original paper uses raw conv features for histology;
we mirror the no-transformer setup so the baseline is the published GA-MIL
rather than a stronger transformer-augmented variant).

Output contract matches `MIML_CLAM`:
    {'logits': (B,L), 'coupled_logits': (B,L), 'attn_weights': (B,L,N)}
"""

import torch
import torch.nn as nn

from miml_clam.models.instance_embedding import InstanceEmbedding


class GatedAttention(nn.Module):
    """Gated attention pooling + L independent linear heads."""

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

        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            num_layers=embedding_layers,
        )

        # Gated attention (eq. 9 in Ilse et al. 2018)
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_w = nn.Linear(attn_dim, 1)

        # L independent linear classifiers
        self.classifiers = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_labels)
        ])

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        """Same interface as MIML_CLAM.forward."""
        h = self.instance_embedding(bags)             # (B, N, D)

        # Gated attention
        v = self.attention_V(h)                        # (B, N, attn_dim)
        u = self.attention_U(h)                        # (B, N, attn_dim)
        scores = self.attention_w(v * u).squeeze(-1)   # (B, N)

        if mask is not None:
            scores = scores.masked_fill(mask.bool(), float('-inf'))
        attn = torch.softmax(scores, dim=-1)           # (B, N)

        # Shared bag representation
        bag_repr = torch.bmm(attn.unsqueeze(1), h).squeeze(1)  # (B, D)

        logits = torch.cat([clf(bag_repr) for clf in self.classifiers], dim=1)  # (B, L)

        attn_expanded = attn.unsqueeze(1).expand(-1, self.num_labels, -1).contiguous()

        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': attn_expanded,
        }

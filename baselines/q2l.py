"""Query2Label / Q2L (Liu et al., 2021): transformer-decoder multi-label
classifier with learnable label queries.

Original paper: "Query2Label: A Simple Transformer Way to Multi-Label
Classification". The architecture stacks self-attention over queries (so labels
can interact) followed by cross-attention from queries to features, then a
position-wise FFN, repeated for D layers (D=2 in the original).

Adapted for MIML: instances replace spatial tokens; one decoder block (D=2
internally) processes the L learnable queries against the bag instances;
per-label logit = linear projection of the corresponding query output.

Differences from the original (documented for reviewer transparency):
  * No positional encoding on instances (set inputs are permutation-invariant).
  * Two decoder layers (self+cross+FFN) — matches Q2L's reported configuration.
"""

import torch
import torch.nn as nn

from miml_clam.models.instance_embedding import InstanceEmbedding


class _Q2LDecoderLayer(nn.Module):
    def __init__(self, embed_dim, nhead, ff_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead,
                                               dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead,
                                                dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, q, kv, kv_mask=None):
        a, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.norm1(q + a)
        b, attn = self.cross_attn(q, kv, kv, key_padding_mask=kv_mask,
                                  need_weights=True, average_attn_weights=True)
        q = self.norm2(q + b)
        q = self.norm3(q + self.ffn(q))
        return q, attn


class Q2L(nn.Module):
    use_instance_clustering = False

    def __init__(self, input_dim, num_labels,
                 embed_dim=64, nhead=4, ff_dim=128, dropout=0.15,
                 embedding_layers=2, num_decoder_layers=2, **kwargs):
        super().__init__()
        self.num_labels = int(num_labels)

        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim, embed_dim=embed_dim,
            dropout=dropout, num_layers=embedding_layers,
        )

        self.queries = nn.Parameter(torch.randn(num_labels, embed_dim) * 0.02)
        self.layers = nn.ModuleList([
            _Q2LDecoderLayer(embed_dim, nhead, ff_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_labels, bias=True)

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        B, N, _ = bags.shape
        h = self.instance_embedding(bags)              # (B, N, D)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)
        kpm = mask.bool() if mask is not None else None

        last_attn = None
        for layer in self.layers:
            q, last_attn = layer(q, h, kv_mask=kpm)

        W = self.classifier.weight
        b = self.classifier.bias
        logits = (q * W.unsqueeze(0)).sum(-1) + b.unsqueeze(0)  # (B, L)

        if last_attn is None:
            last_attn = torch.zeros(B, self.num_labels, N, device=bags.device)
        if mask is not None:
            last_attn = last_attn.masked_fill(mask.unsqueeze(1).bool(), 0.0)
            last_attn = last_attn / last_attn.sum(-1, keepdim=True).clamp(min=1e-8)

        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': last_attn,
        }

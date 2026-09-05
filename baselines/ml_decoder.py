"""ML-Decoder (Ridnik et al., WACV 2023): scalable transformer-decoder
classification head for multi-label classification.

Original paper: "ML-Decoder: Scalable and Versatile Classification Head".
Adapted for the MIML setting: instance embeddings replace spatial tokens; per-
label learnable queries cross-attend to instances; per-query classifier emits
the per-label logit.

Differences vs original (documented for reviewer transparency):
  * Original uses spatial CNN feature map; we use bag instances (already
    permutation-invariant), so no positional encoding is added.
  * The shared FFN ("group decoding") is preserved: a single FFN consumed by
    every query, with a per-label linear head on top.
  * Single decoder layer (matches the original baseline configuration);
    multi-layer variants are reported in the original paper's appendix to give
    marginal gains on multi-label benchmarks.

Constructor matches the rest of `baselines/`:
    forward(bags, mask, bag_labels, compute_instance_loss) -> dict
        with keys {'logits', 'coupled_logits', 'attn_weights'}.
"""

import torch
import torch.nn as nn

from miml_clam.models.instance_embedding import InstanceEmbedding


class MLDecoder(nn.Module):
    use_instance_clustering = False

    def __init__(self, input_dim, num_labels,
                 embed_dim=64, nhead=4, ff_dim=128, dropout=0.15,
                 embedding_layers=2, **kwargs):
        super().__init__()
        self.num_labels = int(num_labels)
        self.embed_dim = int(embed_dim)

        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim, embed_dim=embed_dim,
            dropout=dropout, num_layers=embedding_layers,
        )

        # Per-label learnable query.
        self.queries = nn.Parameter(torch.randn(num_labels, embed_dim) * 0.02)

        # Cross-attention: queries -> instances. Use PyTorch MHA for clarity.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Group-shared FFN (the "ML-Decoder trick" that makes it scale to many labels).
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        # Per-label linear classifier (one row per label).
        self.classifier = nn.Linear(embed_dim, num_labels, bias=True)

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        B, N, _ = bags.shape
        h = self.instance_embedding(bags)              # (B, N, D)

        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)

        # mask: True where padded; MultiheadAttention's key_padding_mask is True for "ignore".
        kpm = mask.bool() if mask is not None else None
        attended, attn = self.cross_attn(query=q, key=h, value=h,
                                         key_padding_mask=kpm,
                                         need_weights=True, average_attn_weights=True)
        attended = self.norm1(q + attended)
        attended = self.norm2(attended + self.ffn(attended))   # (B, L, D)

        # Per-label logit: dot product with the corresponding row of the classifier
        # weight matrix. Equivalent to "per-query linear head".
        W = self.classifier.weight  # (L, D)
        b = self.classifier.bias    # (L,)
        logits = (attended * W.unsqueeze(0)).sum(-1) + b.unsqueeze(0)  # (B, L)

        # attn comes back as (B, L, N) when average_attn_weights=True.
        if attn is None:
            attn = torch.zeros(B, self.num_labels, N, device=bags.device)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).bool(), 0.0)
            attn = attn / attn.sum(-1, keepdim=True).clamp(min=1e-8)

        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': attn,
        }

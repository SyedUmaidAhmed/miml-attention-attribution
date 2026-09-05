"""TransMIL-lite baseline (port of Shao et al., NeurIPS 2021).

"TransMIL: Transformer based Correlated Multiple Instance Learning."

This is a simplified port for fair head-to-head with the other attention-MIL
methods in the paper. We keep the central TransMIL idea — prepend a [CLS]
token and let it aggregate information through a Transformer self-attention
block — but skip the PPEG (Pyramid Position Encoding Generator) module from
the original paper, which would otherwise advantage a transformer-based
baseline with a feature the other baselines lack (and which is specific to
the WSI grid structure of the original work).

Architecture:
  1. Instance embedding (reused).
  2. Prepend a learnable [CLS] token: instances are now (B, N+1, D).
  3. Single-layer pre-norm Transformer encoder (reused `ContextualEncoder`).
  4. Per-label linear classifiers on the post-attention CLS token.

Multi-label adaptation: L independent linear heads on the CLS token, mirroring
the way a [CLS]-token classifier is extended to multi-label in BERT-style
models.

Output contract matches `MIML_CLAM`:
    {'logits': (B,L), 'coupled_logits': (B,L), 'attn_weights': (B,L,N)}
The attention map returned is the [CLS]-row of the LAST encoder layer's
self-attention matrix (averaged over heads, as `nn.MultiheadAttention` reports
it), restricted to the original N instances, renormalised to sum to one over
real instances, and broadcast over labels (the [CLS] token shares a single
attention pattern across all labels, just like `ABMILML`).

History (2026-09-04): until this date the map returned here was NOT the
encoder's attention. It was a softmax over a bilinear similarity
`<W cls_out, inst_out_n>` with a separate `attn_proj` weight W that received
no gradient from any loss and therefore stayed at its random initialisation
-- i.e. a random projection of the post-encoder representations. Every
TransMIL attribution cell produced before this date measured that random
read-out; all five cells were re-run after the fix. The instance-level
results on the partitioned benchmarks are unaffected in principle (any
label-agnostic map scores exactly at chance there, Cor. 1 of the paper) but
the VOC-Patch cells are not.
"""

import torch
import torch.nn as nn

from miml_clam.models.instance_embedding import InstanceEmbedding
from miml_clam.models.contextual_encoder import ContextualEncoder


class TransMILLite(nn.Module):
    """TransMIL-style [CLS] aggregator + per-label linear heads.

    NOTE: this is the *lite* version — no PPEG positional encoder. That keeps
    it on equal footing with the other attention-MIL baselines and avoids
    an apples-to-oranges comparison driven by spatial priors that only the
    original WSI setting provides.
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
        nhead=4,
        ff_dim=128,
        transformer_layers=1,
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

        # Learnable [CLS] token (BERT-style)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.encoder = ContextualEncoder(
            embed_dim=embed_dim,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=transformer_layers,
        )

        # L independent linear classifiers on the post-attention CLS token
        self.classifiers = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_labels)
        ])


    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        """Forward pass with [CLS] aggregation."""
        B, N, _ = bags.shape
        h = self.instance_embedding(bags)                       # (B, N, D)

        # Prepend CLS token -> (B, N+1, D)
        cls = self.cls_token.expand(B, -1, -1)                  # (B, 1, D)
        h_seq = torch.cat([cls, h], dim=1)                      # (B, N+1, D)

        # Extend mask with a False for the CLS position (always attended)
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=mask.dtype, device=mask.device)
            mask_seq = torch.cat([cls_mask, mask], dim=1)        # (B, N+1)
        else:
            mask_seq = None

        # Self-attention (CLS aggregates information from all instances).
        # The layers are stepped manually so that the input of the last layer
        # is available: the attention map we report is that layer's actual
        # [CLS]-row attention, recomputed from the same pre-norm input with
        # need_weights=True. The layer maths is identical to calling
        # `self.encoder(...)`; nn.TransformerEncoder is a plain loop over its
        # layers with no final norm in this configuration.
        x = h_seq
        last_in = x
        for layer in self.encoder.encoder.layers:
            last_in = x
            x = layer(x, src_key_padding_mask=mask_seq)
        h_seq = x                                                # (B, N+1, D)

        cls_out = h_seq[:, 0, :]                                 # (B, D)

        # Per-label classification on CLS
        logits = torch.cat([clf(cls_out) for clf in self.classifiers], dim=1)  # (B, L)

        # Real attention map: [CLS] row of the last layer's self-attention.
        # norm_first=True, so the attention input is norm1(last_in).
        last = self.encoder.encoder.layers[-1]
        z = last.norm1(last_in)
        _, attn_full = last.self_attn(
            z, z, z, key_padding_mask=mask_seq,
            need_weights=True, average_attn_weights=True)        # (B, N+1, N+1)
        attn = attn_full[:, 0, 1:]                               # (B, N): CLS -> instances
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), 0.0)
        # Drop the CLS->CLS mass and renormalise over real instances so the
        # row is a distribution over the bag (attribution metrics are
        # rank-based, so this rescaling cannot change any reported number).
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attn_expanded = attn.unsqueeze(1).expand(-1, self.num_labels, -1).contiguous()

        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': attn_expanded,
        }

"""MIML-CLAM++: Multi-Instance Multi-Label CLAM.

Full model wiring: Instance Embedding -> Contextual Encoder ->
Label-Conditional Gated Attention -> Label Correlation Head + Instance Clustering.

Architecture:
  A. Instance Embedding: MLP with LayerNorm + GELU + residual
  B. Contextual Encoder: Pre-norm Transformer (instance-instance interactions)
  C. Label-Conditional Gated Attention: Different labels attend to different instances
  D. Label Correlation Head: Per-label classifiers + correlation coupling
  E. Instance Clustering Loss: CLAM-adapted pseudo instance-level supervision
"""

import torch
import torch.nn as nn

from .instance_embedding import InstanceEmbedding
from .contextual_encoder import ContextualEncoder
from .label_conditional_attn import LabelConditionalGatedAttention
from .label_correlation_head import LabelCorrelationHead
from .instance_clustering import InstanceClusteringLoss
from .contrastive_head import ContrastiveHead
from .anchor_head import LabelAnchorHead


class MIML_CLAM(nn.Module):
    """Multi-Instance Multi-Label CLAM++ model.

    Shape flow (Scene): [B,9,15] -> embed [B,9,64] -> transformer [B,9,64]
                        -> attention [B,6,9] -> bag_repr [B,6,64] -> logits [B,6]
    """

    def __init__(
        self,
        input_dim,
        num_labels,
        embed_dim=64,
        nhead=4,
        ff_dim=128,
        transformer_layers=1,
        attn_dim=32,
        dropout=0.15,
        k_sample=2,
        use_transformer=True,
        use_correlation=True,
        use_instance_clustering=False,
        use_gated_attention=True,
        embedding_layers=2,
        use_contrastive_head=False,
        contrastive_dim=128,
        use_anchor_head=False,
        anchor_dim=32,
        use_hap=False,
        hap_topk=3,
        hap_use_temperature=False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.use_transformer = use_transformer
        self.use_correlation = use_correlation
        self.use_instance_clustering = use_instance_clustering
        self.use_contrastive_head = use_contrastive_head
        self.use_anchor_head = use_anchor_head

        # Component A: Instance Embedding
        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            num_layers=embedding_layers,
        )

        # Component B: Contextual Encoder
        if use_transformer:
            self.contextual_encoder = ContextualEncoder(
                embed_dim=embed_dim,
                nhead=nhead,
                ff_dim=ff_dim,
                dropout=dropout,
                num_layers=transformer_layers,
            )

        # Component C: Label-Conditional Gated Attention (optionally
        # with Hybrid Attention Pooling on top — see Section X)
        self.label_attention = LabelConditionalGatedAttention(
            embed_dim=embed_dim,
            num_labels=num_labels,
            attn_dim=attn_dim,
            dropout=dropout,
            use_gated=use_gated_attention,
            use_hap=use_hap,
            hap_topk=hap_topk,
            hap_use_temperature=hap_use_temperature,
        )

        # Component D: Label Correlation Head
        self.label_head = LabelCorrelationHead(
            embed_dim=embed_dim,
            num_labels=num_labels,
            dropout=dropout,
        )

        # Component E: Instance Clustering Loss
        if use_instance_clustering:
            self.instance_clustering = InstanceClusteringLoss(
                embed_dim=embed_dim,
                num_labels=num_labels,
                k_sample=k_sample,
            )

        # Component F: Contrastive projection head (MIML-ALFA++ optional)
        if use_contrastive_head:
            self.contrastive_head = ContrastiveHead(
                embed_dim=embed_dim,
                proj_dim=contrastive_dim,
                dropout=dropout,
            )

        # Component G: Label Anchor Alignment head (MIML-ALFA++ novelty)
        if use_anchor_head:
            self.anchor_head = LabelAnchorHead(
                embed_dim=embed_dim,
                num_labels=num_labels,
                anchor_dim=anchor_dim,
                dropout=dropout,
            )

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        """
        Args:
            bags: (B, N, input_dim) — padded bags of instances
            mask: (B, N) — True where padded
            bag_labels: (B, L) — bag-level labels (for instance clustering)
            compute_instance_loss: bool — whether to compute instance clustering loss

        Returns dict with:
            logits: (B, L) — raw logits
            coupled_logits: (B, L) — correlation-coupled logits
            attn_weights: (B, L, N) — attention weights per label
            instance_loss: scalar (if compute_instance_loss)
            instance_acc: float (if compute_instance_loss)
        """
        # A: Instance Embedding
        h = self.instance_embedding(bags)  # (B, N, D)

        # B: Contextual Encoder
        if self.use_transformer:
            h = self.contextual_encoder(h, mask=mask)  # (B, N, D)

        # C: Label-Conditional Gated Attention
        # eff_attn is the distribution that actually produced bag_repr; it
        # differs from attn_weights only under HAP (where attn_weights is the
        # pre-mixture softmax). attn_weights keeps its original meaning so
        # training behaviour and previously published results are unchanged.
        bag_repr, attn_weights, eff_attn = self.label_attention(h, mask=mask)

        # D: Label Correlation Head
        logits, coupled_logits = self.label_head(
            bag_repr, use_correlation=self.use_correlation
        )

        result = {
            'logits': logits,
            'coupled_logits': coupled_logits,
            'attn_weights': attn_weights,
            'eff_attn': eff_attn,
        }

        # F: Contrastive embedding (optional, MIML-ALFA++)
        if self.use_contrastive_head:
            result['contrastive_emb'] = self.contrastive_head(bag_repr)

        # G: Label-Anchor alignment logits (optional, MIML-ALFA++ novelty)
        if self.use_anchor_head:
            result['anchor_logits'] = self.anchor_head(
                bag_repr, self.label_attention.label_embeddings
            )

        # E: Instance Clustering Loss
        if compute_instance_loss and self.use_instance_clustering and bag_labels is not None:
            inst_loss, inst_acc = self.instance_clustering(
                h, attn_weights.detach(), bag_labels, mask=mask
            )
            result['instance_loss'] = inst_loss
            result['instance_acc'] = inst_acc

        return result

    def get_attention_maps(self, bags, mask=None):
        """Get attention maps for visualization (no grad)."""
        with torch.no_grad():
            result = self.forward(bags, mask=mask)
        return result['attn_weights']

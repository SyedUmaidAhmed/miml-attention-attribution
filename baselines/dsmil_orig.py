"""DSMIL in its original form (Li, Li & Eliceiri, CVPR 2021).

Added for the DMKD revision (reviewer 1, point 6). `baselines/dsmil.py` is a
variant: it projects the critical instance with W_q and every instance with a
separate W_k, adds a value projection, and scores each label with its own row
of a linear head. The official implementation (github.com/binli123/dsmil-wsi,
`dsmil.py`, class `BClassifier`) instead:

  * passes *every* instance and the critical instances through ONE shared
    query network  q(.) = Tanh(Linear(ReLU(Linear(.))))   (nonlinear=True),
  * scores instance j for class c by  <q(h_j), q(h_{m_c})>  scaled by
    1/sqrt(dim q), softmax over instances,
  * pools the *unprojected* instances (passing_v=False -> V = identity),
  * classifies with Conv1d(C, C, kernel_size=D) over the (C, D) bag matrix,
    which mixes all classes' pooled vectors into every class score,
  * and the final score is the mean of the max-instance stream and the bag
    stream (the 0.5/0.5 combination used for training and testing).

Everything this file changes relative to `dsmil.py` is listed above; the
instance encoder (`InstanceEmbedding`) and the stream-1 instance classifier are
the same, so the two DSMIL forms differ only in the attention / bag stream.
Batched and masked for padded bags; mask is True at padded positions.

Output contract matches `MIML_CLAM`:
    {'logits': (B,L), 'coupled_logits': (B,L), 'attn_weights': (B,L,N)}
"""

import torch
import torch.nn as nn

from miml_clam.models.instance_embedding import InstanceEmbedding


class DSMILOriginal(nn.Module):
    """Dual-stream MIL with the official shared query network."""

    use_instance_clustering = False

    def __init__(self, input_dim, num_labels, embed_dim=64, dropout=0.15,
                 embedding_layers=2, q_dim=128, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.q_dim = q_dim

        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim, embed_dim=embed_dim, dropout=dropout,
            num_layers=embedding_layers)

        # Stream 1 (official IClassifier.fc): instance -> class logits.
        self.instance_classifier = nn.Linear(embed_dim, num_labels)

        # Shared query network (official BClassifier.q, nonlinear=True).
        self.q = nn.Sequential(
            nn.Linear(embed_dim, q_dim), nn.ReLU(),
            nn.Linear(q_dim, q_dim), nn.Tanh())

        # Official BClassifier.fcc: Conv1d(C, C, kernel_size=D).
        self.fcc = nn.Conv1d(num_labels, num_labels, kernel_size=embed_dim)

    def forward(self, bags, mask=None, bag_labels=None,
                compute_instance_loss=False):
        h = self.instance_embedding(bags)                   # (B, N, D)
        B, N, D = h.shape

        inst_logits = self.instance_classifier(h)            # (B, N, L)
        if mask is not None:
            inst_logits_m = inst_logits.masked_fill(
                mask.bool().unsqueeze(-1), float('-inf'))
        else:
            inst_logits_m = inst_logits
        s1, critic_idx = inst_logits_m.max(dim=1)            # (B, L)

        Q = self.q(h)                                        # (B, N, q)
        q_max = torch.gather(                                # (B, L, q)
            Q, 1, critic_idx.unsqueeze(-1).expand(-1, -1, self.q_dim))

        scores = torch.bmm(q_max, Q.transpose(1, 2)) / (self.q_dim ** 0.5)
        if mask is not None:                                 # (B, L, N)
            scores = scores.masked_fill(mask.bool().unsqueeze(1),
                                        float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        bag_mat = torch.bmm(attn, h)                         # (B, L, D)
        s2 = self.fcc(bag_mat).squeeze(-1)                   # (B, L)

        logits = 0.5 * (s1 + s2)
        return {'logits': logits, 'coupled_logits': logits,
                'attn_weights': attn}

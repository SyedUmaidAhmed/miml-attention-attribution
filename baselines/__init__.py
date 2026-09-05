"""External MIL baselines for ICDM 2026 head-to-head comparison.

Each baseline implements the same forward() contract as
`miml_clam.models.miml_clam.MIML_CLAM`:

    forward(bags, mask=None, bag_labels=None, compute_instance_loss=False)
        -> {'logits': (B,L), 'coupled_logits': (B,L), 'attn_weights': (B,L,N)}

so the existing `EnhancedTrainer` works unchanged.

Reference papers:
  - AttnMIL / GatedAttention : Ilse et al., ICML 2018
  - DSMIL                    : Li et al., CVPR 2021
  - TransMIL-lite            : Shao et al., NeurIPS 2021 (PPEG omitted)
  - ML-Decoder               : Ridnik et al., WACV 2023
  - Q2L (Query2Label)        : Liu et al., 2021
"""

from .attnmil import AttnMIL
from .gated_attn import GatedAttention
from .dsmil import DSMIL
from .transmil_lite import TransMILLite
from .ml_decoder import MLDecoder
from .q2l import Q2L

__all__ = ["AttnMIL", "GatedAttention", "DSMIL", "TransMILLite", "MLDecoder", "Q2L"]

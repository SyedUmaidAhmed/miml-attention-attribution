"""Enhanced Curriculum Trainer with Asymmetric Loss.

Modifications over base CurriculumTrainer:
  1. ASL replaces BCE for better handling of multi-label class imbalance
  2. Supports bag mixup in both phases
  3. Keeps all other training dynamics identical (SWA, early stopping, etc.)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np

from .augmentation import bag_mixup, psemix
from .asymmetric_loss import AsymmetricLoss
from .two_way_loss import TwoWayMLLoss
from .contrastive_loss import JaccardSupConLoss
from ..metrics.miml_metrics import AllFive


class EnhancedTrainer:
    """Two-phase curriculum trainer with ASL and bag mixup."""

    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device

        self.label_smooth = config.get('label_smoothing', 0.0)

        # Choose loss function (priority: ASL2 > ASL > BCE)
        if config.get('use_two_way_loss', False):
            self.loss_fn = TwoWayMLLoss(
                gamma_neg=config.get('asl_gamma_neg', 2),
                gamma_pos=config.get('asl_gamma_pos', 0),
                clip=config.get('asl_clip', 0.02),
                margin=config.get('twl_margin', 0.05),
                lam_rel=config.get('twl_lam_rel', 0.5),
            )
            self.bce_loss = self.loss_fn
        elif config.get('use_asymmetric_loss', False):
            self.loss_fn = AsymmetricLoss(
                gamma_neg=config.get('asl_gamma_neg', 4),
                gamma_pos=config.get('asl_gamma_pos', 1),
                clip=config.get('asl_clip', 0.05),
            )
            self.bce_loss = self.loss_fn  # Use ASL everywhere
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.bce_loss = nn.BCEWithLogitsLoss()

        # Optional auxiliary contrastive loss (MIML-ALFA++)
        self.use_contrastive = config.get('use_contrastive', False)
        if self.use_contrastive:
            self.contrastive_loss_fn = JaccardSupConLoss(
                temperature=config.get('contrastive_tau', 0.1),
            )
        self.lam_con_max = float(config.get('lam_con', 0.05))
        self.lam_con_warmup = int(config.get('lam_con_warmup', 20))

        # Optional Label-Anchor Alignment auxiliary loss (MIML-ALFA++ novelty)
        self.use_anchor = config.get('use_anchor', False)
        if self.use_anchor:
            self.anchor_loss_fn = nn.BCEWithLogitsLoss()
        self.lam_anchor = float(config.get('lam_anchor', 0.3))
        self.lam_anchor_warmup = int(config.get('lam_anchor_warmup', 10))

    def _smooth_labels(self, labels):
        if self.label_smooth > 0:
            return labels * (1 - self.label_smooth) + (1 - labels) * self.label_smooth
        return labels

    def _maybe_mixup(self, bags, labels, mask):
        """Apply mixup / PseMix if enabled."""
        use_psemix = self.config.get('use_psemix', False)
        if use_psemix:
            fixed = self.config.get('fixed_bag_size', True)
            alpha = self.config.get('mixup_alpha', 0.2)
            return psemix(
                bags, labels, mask, alpha=alpha, fixed_size=fixed,
                K=self.config.get('psemix_K', 4),
                share_label_bias=self.config.get('psemix_share_label_bias', 0.7),
            )
        if self.config.get('use_mixup', False):
            fixed = self.config.get('fixed_bag_size', True)
            alpha = self.config.get('mixup_alpha', 0.2)
            return bag_mixup(bags, labels, mask, alpha=alpha, fixed_size=fixed)
        return bags, labels, mask

    def train_phase1(self, train_loader, val_loader, num_epochs=80, lr=1e-3):
        """Phase 1: Independent label training on raw logits."""
        self.model.train()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=self.config.get('weight_decay', 5e-4)
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(num_epochs // 3, 10), T_mult=1
        )

        best_ap = 0.0
        best_state = None

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for bags, labels, mask in train_loader:
                bags, labels, mask = bags.to(self.device), labels.to(self.device), mask.to(self.device)

                # Mixup augmentation
                bags, labels, mask = self._maybe_mixup(bags, labels, mask)

                smooth_labels = self._smooth_labels(labels)
                optimizer.zero_grad()

                result = self.model(bags, mask=mask, bag_labels=labels)
                loss = self.bce_loss(result['logits'], smooth_labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config.get('gradient_clip', 1.0))
                optimizer.step()
                scheduler.step(epoch + num_batches / len(train_loader))

                epoch_loss += loss.item()
                num_batches += 1

            val_metrics = self.evaluate(val_loader, use_raw_logits=True)
            val_ap = val_metrics['AveragePrecision']
            if val_ap > best_ap:
                best_ap = val_ap
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  Phase1 Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/max(num_batches,1):.4f} | "
                      f"Val AP: {val_ap:.4f} | Best: {best_ap:.4f}", flush=True)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return best_ap

    def train_phase2(self, train_loader, val_loader, max_epochs=120, lr=3e-4,
                     patience=25, inst_warmup=5):
        """Phase 2: Joint training with SWA and early stopping."""
        self.model.train()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=self.config.get('weight_decay', 5e-4)
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        # SWA
        swa_model = AveragedModel(self.model)
        swa_start = self.config.get('swa_start', 30)
        swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.5, anneal_epochs=5)
        swa_active = False

        bag_weight = self.config.get('bag_weight', 0.9)
        inst_weight = self.config.get('inst_weight', 0.05)

        best_score = float('-inf')
        patience_counter = 0
        best_state = None

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            compute_inst = (epoch >= inst_warmup) and self.model.use_instance_clustering

            for bags, labels, mask in train_loader:
                bags, labels, mask = bags.to(self.device), labels.to(self.device), mask.to(self.device)
                optimizer.zero_grad()

                # Instance clustering on clean data
                inst_loss_val = None
                if compute_inst:
                    result_inst = self.model(bags, mask=mask, bag_labels=labels, compute_instance_loss=True)
                    if 'instance_loss' in result_inst:
                        inst_loss_val = result_inst['instance_loss']

                # Mixup augmentation
                mixed_bags, mixed_labels, mixed_mask = self._maybe_mixup(bags, labels, mask)

                result = self.model(mixed_bags, mask=mixed_mask, bag_labels=mixed_labels)

                # Main loss on coupled logits
                smooth_labels = self._smooth_labels(mixed_labels)
                total_loss = bag_weight * self.loss_fn(result['coupled_logits'], smooth_labels)

                if inst_loss_val is not None:
                    total_loss = total_loss + inst_weight * inst_loss_val

                # Auxiliary Label-Anchor Alignment loss (LAA) with warm-up.
                # Active in both phase-1 and phase-2; here we are inside phase 2.
                if self.use_anchor and 'anchor_logits' in result:
                    if self.lam_anchor_warmup > 0:
                        lam_a = self.lam_anchor * min(
                            1.0, (epoch + 1) / self.lam_anchor_warmup)
                    else:
                        lam_a = self.lam_anchor
                    if lam_a > 0:
                        anchor_loss = self.anchor_loss_fn(
                            result['anchor_logits'], smooth_labels
                        )
                        total_loss = total_loss + lam_a * anchor_loss

                # Auxiliary contrastive loss with linear warm-up over phase 2
                if self.use_contrastive and 'contrastive_emb' in result:
                    if self.lam_con_warmup > 0:
                        lam_con = self.lam_con_max * min(1.0, (epoch + 1) / self.lam_con_warmup)
                    else:
                        lam_con = self.lam_con_max
                    if lam_con > 0:
                        con_loss = self.contrastive_loss_fn(
                            result['contrastive_emb'], mixed_labels.detach()
                        )
                        total_loss = total_loss + lam_con * con_loss

                # NaN guard: skip batch if loss is non-finite (rare; protects MPS)
                if not torch.isfinite(total_loss):
                    optimizer.zero_grad()
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config.get('gradient_clip', 1.0))
                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

            # SWA
            if epoch >= swa_start:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
                swa_active = True
            else:
                scheduler.step()

            # Validation
            eval_model = swa_model.module if swa_active else self.model
            val_metrics = self._evaluate_model(eval_model, val_loader)
            val_ap = val_metrics['AveragePrecision']

            if (epoch + 1) % 5 == 0:
                swa_str = " [SWA]" if swa_active else ""
                print(f"  Phase2 Epoch {epoch+1}/{max_epochs} | Loss: {epoch_loss/max(num_batches,1):.4f} | "
                      f"Val AP: {val_ap:.4f}{swa_str}", flush=True)

            if val_ap > best_score:
                best_score = val_ap
                patience_counter = 0
                state_dict = swa_model.module.state_dict() if swa_active else self.model.state_dict()
                best_state = {k: v.cpu().clone() for k, v in state_dict.items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stop at epoch {epoch+1} (best AP: {best_score:.4f})", flush=True)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return best_score

    def _evaluate_model(self, model, data_loader, use_raw_logits=False):
        """Evaluate a model on data loader."""
        model.eval()
        all_labels, all_raw, all_coupled = [], [], []

        with torch.no_grad():
            for bags, labels, mask in data_loader:
                bags, mask = bags.to(self.device), mask.to(self.device)
                result = model(bags, mask=mask)
                all_raw.append(torch.sigmoid(result['logits']).cpu().numpy())
                all_coupled.append(torch.sigmoid(result['coupled_logits']).cpu().numpy())
                all_labels.append(labels.numpy())

        all_labels = np.concatenate(all_labels)
        all_raw = np.concatenate(all_raw)
        all_coupled = np.concatenate(all_coupled)

        metrics_raw = AllFive(all_labels, all_raw)
        metrics_coupled = AllFive(all_labels, all_coupled)

        if use_raw_logits:
            return metrics_raw
        return metrics_coupled if metrics_coupled['AveragePrecision'] >= metrics_raw['AveragePrecision'] else metrics_raw

    @torch.no_grad()
    def evaluate(self, data_loader, use_raw_logits=False):
        return self._evaluate_model(self.model, data_loader, use_raw_logits)

    @torch.no_grad()
    def get_predictions(self, data_loader):
        """Get raw predictions and labels (for threshold optimization)."""
        self.model.eval()
        all_labels, all_raw, all_coupled = [], [], []

        for bags, labels, mask in data_loader:
            bags, mask = bags.to(self.device), mask.to(self.device)
            result = self.model(bags, mask=mask)
            all_raw.append(torch.sigmoid(result['logits']).cpu().numpy())
            all_coupled.append(torch.sigmoid(result['coupled_logits']).cpu().numpy())
            all_labels.append(labels.numpy())

        return (np.concatenate(all_labels),
                np.concatenate(all_raw),
                np.concatenate(all_coupled))

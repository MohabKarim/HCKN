"""
HCKN v5 — Integrated System
============================
Combines all three critique solutions into a single, runnable pipeline:

1. CA3 Router         (Critique 1 — Task Identity)
2. Overlap-Aware Allocator + Neurogenesis  (Critique 2 — Neuron Overlap)
3. CLS Dual-Rate + Two-Phase + EWC         (Critique 3 — Transfer)

Usage
-----
>>> from hckn import HCKNv5, HCKNConfig
>>> cfg = HCKNConfig()
>>> system = HCKNv5(cfg)
>>> system.train_task(task_id=0, train_loader=..., val_loader=...)
>>> predictions = system.predict(inputs)  # Class-IL, no task ID needed
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from .config import HCKNConfig
from .model import HCKNModel
from .engram import SparseEngramEncoder
from .router import CA3Router
from .separation import OverlapAwareAllocator
from .cls_system import EWCRegularizer, TwoPhaseTrainer, TransferMetrics
from .metrics import AccuracyMatrix, memory_efficiency_report, print_results_table


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


class HCKNv5:
    """Integrated HCKN v5 continual learning system.

    Parameters
    ----------
    config:
        ``HCKNConfig`` instance controlling all hyper-parameters.

    Pipeline (per task)
    -------------------
    1. Phase 1 (Exploration): train with slow backbone LR + EWC, no inhibition.
    2. Phase 2 (Crystallisation): form overlap-aware engram, apply gating.
    3. Store CA3 prototype centroid.
    4. Record Fisher information for EWC.

    Inference (Class-IL, no task ID)
    ---------------------------------
    a. Pass input through backbone → features.
    b. CA3 Router: nearest prototype → predicted task ID.
    c. Apply engram gating for that task.
    d. Forward through task head → class prediction.
    """

    def __init__(self, config: HCKNConfig) -> None:
        self.cfg = config
        self.device = _resolve_device(config.device)

        torch.manual_seed(config.seed)

        # Core model
        self.model = HCKNModel(
            input_dim=config.input_dim,
            hidden_dims=list(config.hidden_dims),
            classes_per_task=config.classes_per_task,
        ).to(self.device)

        # Sub-systems
        self.encoder = SparseEngramEncoder(
            sparsity=config.engram_sparsity,
            inhibition_factor=config.inhibition_factor,
            bias_scale=config.bias_scale,
        )
        self.router = CA3Router(
            distance_metric=config.router_distance,
            temperature=config.router_temperature,
            uncertainty_threshold=config.router_uncertainty_threshold,
        )
        self.allocator = OverlapAwareAllocator(
            sparsity=config.engram_sparsity,
            penalty_weight=config.overlap_penalty_weight,
            neurogenesis_threshold=config.neurogenesis_threshold,
            neurogenesis_new_neurons=config.neurogenesis_new_neurons,
        )
        self.ewc = EWCRegularizer(ewc_lambda=config.ewc_lambda)

        # Tracking
        self.acc_matrix: Optional[AccuracyMatrix] = None
        self.transfer_metrics: Optional[TransferMetrics] = None
        self._num_tasks_seen = 0
        # Neurogenesis count per task
        self.neurogenesis_events: List[bool] = []

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict:
        """Train the model on *task_id*'s data and form its engram.

        Parameters
        ----------
        task_id:
            Zero-based task index.
        train_loader:
            DataLoader for the current task's training data.
        val_loader:
            Optional validation loader; used for prototype centroid
            computation (falls back to train_loader if None).

        Returns
        -------
        dict
            Training log with keys: ``losses``, ``neurogenesis``.
        """
        # Add a new classification head
        self.model.add_head()
        assert len(self.model.heads) == task_id + 1, (
            f"Head count mismatch: expected {task_id+1}, "
            f"got {len(self.model.heads)}"
        )

        # -------------------------------------------------------------- #
        # Phase 1 + 2: Two-Phase Training
        # -------------------------------------------------------------- #
        # NOTE: engram mask is not yet available for phase 2 at this point.
        # We do exploration first, then form the engram, then run a
        # short crystallisation pass.
        n_explore = max(1, int(self.cfg.num_epochs * self.cfg.phase_split))
        n_crystal = max(0, self.cfg.num_epochs - n_explore)

        # Exploration (no engram gating)
        explore_trainer = TwoPhaseTrainer(
            model=self.model,
            task_id=task_id,
            dataloader=train_loader,
            device=self.device,
            num_epochs=n_explore,
            phase_split=1.0,   # all exploration
            head_lr=self.cfg.head_lr,
            backbone_lr_ratio=self.cfg.backbone_lr_ratio,
            ewc_regularizer=self.ewc if task_id > 0 else None,
            engram_mask=None,
        )
        p1_losses = explore_trainer.exploration_phase(n_explore)

        # -------------------------------------------------------------- #
        # Form engram with overlap-aware allocation
        # -------------------------------------------------------------- #
        # Collect mean activations
        mean_act = self.encoder._collect_mean_activations(
            self.model.backbone, train_loader, self.device
        )
        # Allocate with overlap penalty (may trigger neurogenesis)
        engram_mask, neurogenesis_fired = self.allocator.allocate_engram(
            mean_act, task_id, model=self.model
        )
        self.neurogenesis_events.append(neurogenesis_fired)

        # Store the engram
        engram = self.encoder.form_engram(
            self.model, task_id, train_loader, self.device,
            pre_allocated_mask=engram_mask,
        )

        # -------------------------------------------------------------- #
        # Crystallisation (with engram gating)
        # -------------------------------------------------------------- #
        p2_losses: List[float] = []
        if n_crystal > 0:
            crystal_trainer = TwoPhaseTrainer(
                model=self.model,
                task_id=task_id,
                dataloader=train_loader,
                device=self.device,
                num_epochs=n_crystal,
                phase_split=0.0,
                head_lr=self.cfg.head_lr,
                backbone_lr_ratio=self.cfg.backbone_lr_ratio,
                ewc_regularizer=self.ewc,
                engram_mask=engram.mask,
                bias_shifts=engram.bias_shifts,
                inhibition_factor=self.cfg.inhibition_factor,
            )
            p2_losses = crystal_trainer.crystallisation_phase(n_crystal)

        # -------------------------------------------------------------- #
        # Store CA3 prototype centroid
        # -------------------------------------------------------------- #
        proto_loader = val_loader if val_loader is not None else train_loader
        features_list = self._collect_features(proto_loader)
        self.router.store_prototype(task_id, features_list)

        # -------------------------------------------------------------- #
        # Update Fisher information for EWC
        # -------------------------------------------------------------- #
        self.ewc.update_fisher(self.model, train_loader, self.device)

        self._num_tasks_seen += 1

        return {
            "task_id": task_id,
            "phase1_losses": p1_losses,
            "phase2_losses": p2_losses,
            "neurogenesis": neurogenesis_fired,
        }

    # ------------------------------------------------------------------ #
    # Inference (Class-IL)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """Class-IL inference — no task ID required.

        Parameters
        ----------
        inputs:
            Input batch ``[B, *]``.

        Returns
        -------
        Tuple[torch.Tensor, List[int]]
            ``(class_logits, predicted_task_ids)``
        """
        self.model.eval()
        inputs = inputs.to(self.device)

        # Step 1: backbone features
        features = self.model.forward_features(inputs)

        # Step 2: CA3 router — find nearest prototype
        pred_task_ids = self.router.route(features)

        # Step 3: per-sample engram gating + head prediction
        logits_list = []
        for i, task_id in enumerate(pred_task_ids):
            feat_i = features[i:i+1]
            gated = self.encoder.apply_engram_gating(feat_i, task_id)
            logit = self.model.heads[task_id](gated)
            logits_list.append(logit)

        logits = torch.cat(logits_list, dim=0)
        return logits, pred_task_ids

    @torch.no_grad()
    def predict_with_task_id(
        self, inputs: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        """Task-IL inference — uses known task ID (for comparison)."""
        self.model.eval()
        inputs = inputs.to(self.device)
        return self.model(
            inputs,
            task_id=task_id,
            engram_mask=self.encoder.engrams[task_id].mask,
            bias_shifts=self.encoder.engrams[task_id].bias_shifts,
            inhibition_factor=self.cfg.inhibition_factor,
        )

    # ------------------------------------------------------------------ #
    # Evaluation helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate_task(
        self,
        task_id: int,
        dataloader: DataLoader,
        use_router: bool = True,
    ) -> float:
        """Evaluate accuracy on a single task.

        Parameters
        ----------
        task_id:
            Ground-truth task index (used for label offset and task-IL eval).
        dataloader:
            DataLoader for the task's test/validation data.
        use_router:
            If True, use CA3 router (Class-IL).  If False, use known task ID.
        """
        self.model.eval()
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if use_router:
                logits, pred_task_ids = self.predict(inputs)
                # Update router accuracy tracking
                self.router.update_accuracy(
                    pred_task_ids, [task_id] * len(pred_task_ids)
                )
            else:
                logits = self.predict_with_task_id(inputs, task_id)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / max(total, 1)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        """Print a full results summary including all metrics."""
        overlap_stats = self.allocator.overlap_stats()
        mem_stats = memory_efficiency_report(
            self.model, self.encoder, self._num_tasks_seen
        )
        print_results_table(
            acc_matrix=self.acc_matrix or AccuracyMatrix(self._num_tasks_seen),
            router_accuracy=self.router.routing_accuracy() if self._num_tasks_seen > 0 else None,
            overlap_stats=overlap_stats,
            memory_stats=mem_stats,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _collect_features(self, dataloader: DataLoader) -> torch.Tensor:
        """Collect backbone features for all samples in *dataloader*."""
        self.model.eval()
        feats = []
        for inputs, _ in dataloader:
            inputs = inputs.to(self.device)
            feats.append(self.model.forward_features(inputs).cpu())
        return torch.cat(feats, dim=0)

"""
CLS Dual-System: Two-Phase Training + EWC  (Critique 3 — Transfer)
==================================================================
Implements McClelland et al.'s Complementary Learning Systems (CLS) theory
in a practical PyTorch training loop.

Neuroscience background
-----------------------
**CLS Theory** (McClelland, McNaughton & O'Reilly, Psychological Review 1995)
proposes two complementary memory systems:

1. **Hippocampus** — learns rapidly via pattern-separated episodic
   encoding.  Corresponds to our fast engram head.

2. **Neocortex** — learns slowly through repeated re-exposure to many
   examples.  Corresponds to our shared backbone with a low learning rate
   and EWC-lite regularisation.

**Two-phase training** mirrors offline hippocampal-to-neocortical
consolidation during slow-wave sleep (Diekelmann & Born, Nature Reviews
Neuroscience 2010):

- **Phase 1 (Exploration)**: No engram inhibition.  The backbone is free
  to learn from new data.  A light EWC penalty prevents catastrophic drift.
- **Phase 2 (Crystallisation)**: Backbone is frozen / heavily regularised.
  The engram is formed with inhibition gating, locking in the task-specific
  memory trace.

**EWC-Lite** (after Kirkpatrick et al., PNAS 2017): Diagonal Fisher
information approximation penalises changes to parameters that were
important for previous tasks.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# ====================================================================== #
# EWC Regulariser
# ====================================================================== #

class EWCRegularizer:
    """Elastic Weight Consolidation (diagonal Fisher approximation).

    Parameters
    ----------
    ewc_lambda:
        Penalty strength.  Higher → stronger consolidation.

    Neuroscience analogy
    --------------------
    Synaptic plasticity tagging — synapses that were most important for a
    previous memory are marked with a "tag" that resists further change
    (Frey & Morris, Nature 1997).
    """

    def __init__(self, ewc_lambda: float = 400.0) -> None:
        self.ewc_lambda = ewc_lambda
        # List of (param_name → fisher, param_name → optimal_theta) per task
        self._fishers: List[Dict[str, torch.Tensor]] = []
        self._optima: List[Dict[str, torch.Tensor]] = []

    def update_fisher(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_samples: int = 200,
    ) -> None:
        """Compute diagonal Fisher information for *model*'s parameters.

        Parameters
        ----------
        model:
            The trained model after completing a task.
        dataloader:
            DataLoader for the current task's data (used to compute
            expected log-likelihood gradients).
        num_samples:
            Number of samples to use (subsample for speed).
        """
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param.data)

        collected = 0
        for inputs, labels in dataloader:
            if collected >= num_samples:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)

            model.zero_grad()
            # Use backbone output → any head for gradient signal
            features = model.backbone(inputs)
            # Average over all heads for a task-agnostic Fisher estimate
            logits_list = [head(features) for head in model.heads]
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
            log_probs = torch.log_softmax(logits, dim=-1)
            # Use predicted label (online Fisher)
            pred = log_probs.max(dim=-1).indices
            loss = torch.nn.functional.nll_loss(log_probs, pred)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * batch_size

            collected += batch_size

        # Normalise
        for name in fisher:
            fisher[name] /= max(collected, 1)

        self._fishers.append({n: f.clone() for n, f in fisher.items()})
        self._optima.append(
            {n: p.data.clone() for n, p in model.named_parameters()}
        )

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty for the current model parameters.

        loss_ewc = λ/2 * Σ_i Σ_j F_ij * (θ_ij − θ*_ij)²

        where *i* indexes past tasks and *j* indexes parameters.
        """
        if not self._fishers:
            return torch.tensor(0.0, requires_grad=True)

        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)
        named_params = dict(model.named_parameters())

        for fisher, optima in zip(self._fishers, self._optima):
            for name, param in named_params.items():
                if name not in fisher:
                    continue
                f = fisher[name].to(device)
                theta_star = optima[name].to(device)
                loss = loss + (f * (param - theta_star).pow(2)).sum()

        return (self.ewc_lambda / 2.0) * loss


# ====================================================================== #
# CLS Dual-System parameter groups
# ====================================================================== #

class CLSDualSystem:
    """Manages separate learning rates for backbone vs. task head.

    Parameters
    ----------
    model:
        The ``HCKNModel`` instance.
    head_lr:
        Learning rate for the current task head (fast system).
    backbone_lr_ratio:
        Backbone LR = head_lr × backbone_lr_ratio (slow system).
    weight_decay:
        L2 regularisation applied to all parameters.

    Neuroscience analogy
    --------------------
    The neocortex integrates knowledge slowly over many experiences while
    the hippocampus encodes episodes rapidly.  The ratio of learning rates
    captures this asymmetry (McClelland et al., 1995).
    """

    def __init__(
        self,
        model: nn.Module,
        task_id: int,
        head_lr: float = 1e-3,
        backbone_lr_ratio: float = 0.1,
        weight_decay: float = 1e-4,
    ) -> None:
        self.model = model
        self.task_id = task_id
        self.head_lr = head_lr
        self.backbone_lr = head_lr * backbone_lr_ratio
        self.weight_decay = weight_decay
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> optim.Adam:
        head_params = list(self.model.heads[self.task_id].parameters())
        head_param_ids = {id(p) for p in head_params}
        backbone_params = [
            p for p in self.model.backbone.parameters()
            if id(p) not in head_param_ids
        ]
        return optim.Adam(
            [
                {"params": backbone_params, "lr": self.backbone_lr},
                {"params": head_params, "lr": self.head_lr},
            ],
            weight_decay=self.weight_decay,
        )

    @property
    def optim(self) -> optim.Adam:
        return self.optimizer


# ====================================================================== #
# Two-Phase Trainer
# ====================================================================== #

class TwoPhaseTrainer:
    """Train a task in two phases: Exploration then Crystallisation.

    Parameters
    ----------
    model:
        The ``HCKNModel``.
    task_id:
        Current task index.
    dataloader:
        Training data for this task.
    device:
        Compute device.
    num_epochs:
        Total number of epochs.
    phase_split:
        Fraction of epochs for Phase 1 (Exploration); rest → Phase 2.
    head_lr / backbone_lr_ratio / weight_decay:
        Passed to ``CLSDualSystem``.
    ewc_regularizer:
        Optional EWC regulariser; penalty is added during both phases.
    engram_mask / bias_shifts / inhibition_factor:
        Engram parameters applied during Phase 2 gating.

    Neuroscience analogy
    --------------------
    Mirrors the two-stage memory consolidation observed in rodents:
    initial encoding (exploration / awake) followed by offline replay
    and structural synaptic strengthening during slow-wave sleep
    (Diekelmann & Born, Nature Reviews Neuroscience 2010).
    """

    def __init__(
        self,
        model: nn.Module,
        task_id: int,
        dataloader: DataLoader,
        device: torch.device,
        num_epochs: int = 50,
        phase_split: float = 0.6,
        head_lr: float = 1e-3,
        backbone_lr_ratio: float = 0.1,
        weight_decay: float = 1e-4,
        ewc_regularizer: Optional[EWCRegularizer] = None,
        engram_mask: Optional[torch.Tensor] = None,
        bias_shifts: Optional[torch.Tensor] = None,
        inhibition_factor: float = 0.3,
    ) -> None:
        self.model = model
        self.task_id = task_id
        self.dataloader = dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.phase_split = phase_split
        self.ewc = ewc_regularizer
        self.engram_mask = engram_mask
        self.bias_shifts = bias_shifts
        self.inhibition_factor = inhibition_factor

        self.cls = CLSDualSystem(
            model, task_id,
            head_lr=head_lr,
            backbone_lr_ratio=backbone_lr_ratio,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self) -> Dict[str, List[float]]:
        """Run both training phases and return loss history.

        Returns
        -------
        Dict[str, List[float]]
            ``{'phase1_losses': [...], 'phase2_losses': [...]}``
        """
        n1 = max(1, int(self.num_epochs * self.phase_split))
        n2 = max(0, self.num_epochs - n1)

        phase1_losses = self.exploration_phase(n1)
        phase2_losses = self.crystallisation_phase(n2) if n2 > 0 else []

        return {"phase1_losses": phase1_losses, "phase2_losses": phase2_losses}

    def exploration_phase(self, num_epochs: int) -> List[float]:
        """Phase 1 — train without engram inhibition.

        The backbone is free to learn from new data.  EWC penalty prevents
        catastrophic drift from previously important parameters.
        """
        losses: List[float] = []
        self.model.train()
        opt = self.cls.optim

        for _ in range(num_epochs):
            epoch_loss = 0.0
            batches = 0
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                # No engram gating in exploration phase
                features = self.model.backbone(inputs)
                logits = self.model.heads[self.task_id](features)
                loss = self.criterion(logits, labels)
                if self.ewc is not None:
                    loss = loss + self.ewc.penalty(self.model)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                batches += 1
            losses.append(epoch_loss / max(batches, 1))
        return losses

    def crystallisation_phase(self, num_epochs: int) -> List[float]:
        """Phase 2 — apply engram gating; freeze backbone.

        The backbone parameters are frozen (or have a very small LR).
        Engram gating is applied to lock in the memory trace.
        """
        # Freeze backbone for crystallisation
        for param in self.model.backbone.parameters():
            param.requires_grad_(False)

        losses: List[float] = []
        self.model.train()
        # Only optimise the head in this phase
        opt = optim.Adam(
            self.model.heads[self.task_id].parameters(),
            lr=self.cls.head_lr,
        )

        for _ in range(num_epochs):
            epoch_loss = 0.0
            batches = 0
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                logits = self.model(
                    inputs,
                    task_id=self.task_id,
                    engram_mask=self.engram_mask,
                    bias_shifts=self.bias_shifts,
                    inhibition_factor=self.inhibition_factor,
                )
                loss = self.criterion(logits, labels)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                batches += 1
            losses.append(epoch_loss / max(batches, 1))

        # Unfreeze backbone for future tasks
        for param in self.model.backbone.parameters():
            param.requires_grad_(True)

        return losses


# ====================================================================== #
# Transfer Metrics
# ====================================================================== #

class TransferMetrics:
    """Compute ACC, BWT, and FWT for continual learning evaluation.

    The standard continual learning metrics (Lopez-Paz & Ranzato, NeurIPS
    2017):

    - **ACC**: Average accuracy across all tasks after all training.
    - **BWT**: Backward Transfer — measures forgetting.
      BWT = 1/T Σ_{i<T} (R_{T,i} − R_{i,i})
      Negative BWT indicates forgetting.
    - **FWT**: Forward Transfer — measures positive influence of prior
      tasks on new tasks.
      FWT = 1/(T-1) Σ_{i>0} (R_{i-1,i} − b_i)
      where b_i is the performance of a random initialisation on task i.
    """

    def __init__(self, num_tasks: int) -> None:
        self.num_tasks = num_tasks
        # R[i][j] = accuracy on task j evaluated right after training task i
        self.R: List[List[Optional[float]]] = [
            [None] * num_tasks for _ in range(num_tasks)
        ]
        # Baseline (random init) accuracy per task
        self.baseline: List[Optional[float]] = [None] * num_tasks

    def record(self, train_task: int, eval_task: int, accuracy: float) -> None:
        """Record R[train_task][eval_task] = accuracy."""
        self.R[train_task][eval_task] = accuracy

    def set_baseline(self, task_id: int, accuracy: float) -> None:
        """Set the random-init baseline for task *task_id*."""
        self.baseline[task_id] = accuracy

    def compute_acc(self) -> float:
        """Average accuracy across tasks after all training."""
        last = self.num_tasks - 1
        vals = [
            self.R[last][j]
            for j in range(self.num_tasks)
            if self.R[last][j] is not None
        ]
        return float(np.mean(vals)) if vals else 0.0

    def compute_bwt(self) -> float:
        """Backward Transfer (positive = improved recall, negative = forgot)."""
        vals = []
        for i in range(self.num_tasks - 1):
            r_Ti = self.R[self.num_tasks - 1][i]
            r_ii = self.R[i][i]
            if r_Ti is not None and r_ii is not None:
                vals.append(r_Ti - r_ii)
        return float(np.mean(vals)) if vals else 0.0

    def compute_fwt(self) -> float:
        """Forward Transfer (compared to random-init baseline)."""
        vals = []
        for i in range(1, self.num_tasks):
            r_prev_i = self.R[i - 1][i]
            b_i = self.baseline[i]
            if r_prev_i is not None and b_i is not None:
                vals.append(r_prev_i - b_i)
        return float(np.mean(vals)) if vals else 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "ACC": self.compute_acc(),
            "BWT": self.compute_bwt(),
            "FWT": self.compute_fwt(),
        }




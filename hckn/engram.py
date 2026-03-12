"""
Sparse Engram Encoding (SEE)
============================
Core engram logic: identify the most-activated neurons after training a
task, store a tiny per-task memory trace, and apply gating at inference.

Neuroscience background
-----------------------
When a memory is encoded, only a sparse ensemble of ~5-10 % of neurons
are activated and persistently marked as "engram cells" (Josselyn &
Tonegawa, Science 2020).  Re-activating these cells is sufficient to
retrieve the memory.  Non-engram cells are suppressed by fast-spiking
inhibitory interneurons (Stefanelli et al., Nature Neuroscience 2016).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EngramMemory:
    """Per-task engram stored in memory.

    Attributes
    ----------
    task_id: int
    mask: bool tensor of shape [hidden_dim] — True where neuron is in engram.
    bias_shifts: float tensor of shape [num_engram_neurons].
    head_weights: weight tensor for the task's output head.
    head_bias: bias tensor for the task's output head.
    """
    task_id: int
    mask: torch.Tensor          # bool [D]
    bias_shifts: torch.Tensor   # float [K]  K = mask.sum()
    head_weights: torch.Tensor  # float [classes_per_task, D]
    head_bias: torch.Tensor     # float [classes_per_task]


class SparseEngramEncoder:
    """Identify and store sparse engrams from trained models.

    Parameters
    ----------
    sparsity:
        Fraction of neurons to include in the engram (default 0.08 = 8 %).
    inhibition_factor:
        Suppression multiplier for non-engram neurons at inference.
    bias_scale:
        Scale applied to stored bias shifts.
    """

    def __init__(
        self,
        sparsity: float = 0.08,
        inhibition_factor: float = 0.3,
        bias_scale: float = 1.0,
    ) -> None:
        self.sparsity = sparsity
        self.inhibition_factor = inhibition_factor
        self.bias_scale = bias_scale
        self.engrams: Dict[int, EngramMemory] = {}

    # ------------------------------------------------------------------ #
    # Engram formation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def form_engram(
        self,
        model: nn.Module,
        task_id: int,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        pre_allocated_mask: Optional[torch.Tensor] = None,
    ) -> EngramMemory:
        """Identify engram neurons and store the memory trace.

        The engram mask is determined by the mean activation magnitude of
        each hidden neuron over the task's training data.  If an
        *pre_allocated_mask* is supplied (e.g., from the
        ``OverlapAwareAllocator``), it is used directly instead of the
        top-k selection.

        Parameters
        ----------
        model:
            Trained ``HCKNModel`` at the end of the task.
        task_id:
            Integer task index (0-based).
        dataloader:
            DataLoader yielding ``(inputs, labels)`` pairs from the task.
        device:
            Compute device.
        pre_allocated_mask:
            Optional pre-computed boolean mask; bypasses top-k selection.

        Returns
        -------
        EngramMemory
            The stored engram for this task.
        """
        model.eval()
        backbone = model.backbone

        # Collect mean activations for the last hidden layer
        mean_act = self._collect_mean_activations(backbone, dataloader, device)

        if pre_allocated_mask is not None:
            mask = pre_allocated_mask.to(device)
        else:
            k = max(1, int(self.sparsity * len(mean_act)))
            topk_idx = torch.topk(mean_act, k).indices
            mask = torch.zeros(len(mean_act), dtype=torch.bool, device=device)
            mask[topk_idx] = True

        # Bias shifts: mean activation of engram neurons (scaled)
        bias_shifts = (mean_act[mask] * self.bias_scale).clone()

        # Snapshot the head for this task
        head = model.heads[task_id]
        head_weights = head.weight.detach().clone()
        head_bias = head.bias.detach().clone()

        engram = EngramMemory(
            task_id=task_id,
            mask=mask,
            bias_shifts=bias_shifts,
            head_weights=head_weights,
            head_bias=head_bias,
        )
        self.engrams[task_id] = engram
        return engram

    # ------------------------------------------------------------------ #
    # Inference gating
    # ------------------------------------------------------------------ #

    def apply_engram_gating(
        self,
        features: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        """Apply inhibition + excitation for a given task's engram.

        Parameters
        ----------
        features:
            Backbone output ``[B, D]``.
        task_id:
            Which task's engram to activate.

        Returns
        -------
        torch.Tensor
            Gated feature tensor ``[B, D]``.

        Notes
        -----
        After neurogenesis the backbone output dimension ``D`` may exceed the
        dimension stored in the engram mask.  The mask is zero-padded with
        ``False`` values so that new neurons (which were not part of the
        original engram) are treated as non-engram and receive inhibition.
        If for any reason the mask is larger than ``D``, it is truncated.
        """
        if task_id not in self.engrams:
            raise KeyError(f"No engram stored for task {task_id}")
        engram = self.engrams[task_id]
        mask = engram.mask.to(features.device)
        feat_dim = features.shape[1]
        mask_dim = mask.shape[0]

        if feat_dim > mask_dim:
            # New neurons added by neurogenesis are not part of old engrams
            padded_mask = torch.zeros(feat_dim, dtype=torch.bool, device=features.device)
            padded_mask[:mask_dim] = mask
            mask = padded_mask
        elif mask_dim > feat_dim:
            mask = mask[:feat_dim]

        gated = features.clone()
        gated[:, ~mask] *= self.inhibition_factor

        bias = engram.bias_shifts
        if bias is not None:
            # Apply bias only to engram neurons within the current dimension
            engram_indices = mask.nonzero(as_tuple=True)[0]
            valid_bias_count = min(len(bias), len(engram_indices))
            if valid_bias_count > 0:
                gated[:, engram_indices[:valid_bias_count]] += (
                    bias[:valid_bias_count].to(features.device).unsqueeze(0)
                )
        return gated

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _collect_mean_activations(
        self,
        backbone: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> torch.Tensor:
        """Average absolute activation magnitude over the dataset."""
        total: Optional[torch.Tensor] = None
        count = 0
        backbone.eval()
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            acts = backbone(inputs)
            if total is None:
                total = acts.abs().sum(dim=0)
            else:
                total += acts.abs().sum(dim=0)
            count += inputs.size(0)
        assert total is not None
        return total / count

    def memory_size_bytes(self, task_id: int) -> int:
        """Approximate size in bytes of one engram (for reporting)."""
        engram = self.engrams[task_id]
        bits = (
            engram.mask.numel()                          # bool mask
            + engram.bias_shifts.numel() * 4             # float32
            + engram.head_weights.numel() * 4
            + engram.head_bias.numel() * 4
        )
        return int(bits)

    def total_params_stored(self, task_id: int) -> int:
        """Count of scalar parameters stored for one task."""
        engram = self.engrams[task_id]
        return int(
            engram.mask.numel()
            + engram.bias_shifts.numel()
            + engram.head_weights.numel()
            + engram.head_bias.numel()
        )

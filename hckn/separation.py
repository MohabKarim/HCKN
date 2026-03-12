"""
Overlap-Aware Engram Allocation + Neurogenesis  (Critique 2)
============================================================
Prevents neuron-level conflicts between engrams of different tasks by
penalising neurons that are already claimed and, when necessary, growing
new neurons (neurogenesis).

Neuroscience background
-----------------------
The **dentate gyrus (DG)** transforms similar cortical input patterns into
highly dissimilar, non-overlapping output patterns — a process called
**pattern separation** (Leutgeb et al., Science 2007).  Two mechanisms
cooperate:

1. **Decorrelation of existing neurons** — inhibitory interneurons reduce
   the firing probability of neurons already recruited into other memory
   traces.  We implement this as an *overlap penalty* that subtracts from
   the raw activation score of already-claimed neurons during engram
   allocation.

2. **Adult hippocampal neurogenesis** — when representational capacity is
   exhausted, the DG recruits brand-new granule cells that are
   preferentially incorporated into new, non-overlapping assemblies
   (Aimone et al., Neuron 2011).  We implement this as dynamic network
   expansion when the best possible allocation would still produce > 20 %
   Jaccard overlap with any existing engram.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_jaccard(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """Jaccard overlap between two boolean neuron masks.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Handles dimension mismatches after neurogenesis by padding the shorter
    mask with ``False`` (new neurons not present in the older mask are treated
    as not belonging to that engram).
    """
    a_len, b_len = len(mask_a), len(mask_b)
    if a_len != b_len:
        max_len = max(a_len, b_len)
        if a_len < max_len:
            padded = torch.zeros(max_len, dtype=torch.bool, device=mask_a.device)
            padded[:a_len] = mask_a
            mask_a = padded
        if b_len < max_len:
            padded = torch.zeros(max_len, dtype=torch.bool, device=mask_b.device)
            padded[:b_len] = mask_b
            mask_b = padded
    intersection = (mask_a & mask_b).sum().item()
    union = (mask_a | mask_b).sum().item()
    if union == 0:
        return 0.0
    return intersection / union


def compute_overlap_matrix(engrams: Dict[int, torch.Tensor]) -> np.ndarray:
    """Pairwise Jaccard overlap matrix for all stored engram masks.

    Parameters
    ----------
    engrams:
        Mapping from task_id to boolean mask tensor ``[D]``.

    Returns
    -------
    np.ndarray
        Square matrix of shape ``[T, T]`` where entry ``[i, j]`` is the
        Jaccard overlap between task *i* and task *j*.
    """
    task_ids = sorted(engrams.keys())
    T = len(task_ids)
    matrix = np.zeros((T, T), dtype=np.float32)
    for i, ti in enumerate(task_ids):
        for j, tj in enumerate(task_ids):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = compute_jaccard(engrams[ti], engrams[tj])
    return matrix


def should_trigger_neurogenesis(
    new_mask: torch.Tensor,
    existing_engrams: Dict[int, torch.Tensor],
    threshold: float = 0.20,
) -> bool:
    """Return True if the proposed mask would overlap too much with any existing engram.

    Parameters
    ----------
    new_mask:
        Boolean tensor ``[D]`` for the candidate new engram.
    existing_engrams:
        Mapping task_id → boolean mask for already-formed engrams.
    threshold:
        Jaccard overlap above which neurogenesis is triggered.
    """
    for mask in existing_engrams.values():
        if compute_jaccard(new_mask, mask) > threshold:
            return True
    return False


class OverlapAwareAllocator:
    """Allocate engram neurons while minimising overlap with prior tasks.

    Parameters
    ----------
    sparsity:
        Fraction of neurons to include in the engram.
    penalty_weight:
        How strongly to penalise neurons already claimed by prior engrams.
        ``adjusted[i] = raw[i] - penalty_weight * times_neuron_i_claimed``
    neurogenesis_threshold:
        Jaccard overlap above which neurogenesis is triggered.
    neurogenesis_new_neurons:
        How many neurons to add when neurogenesis fires.

    Neuroscience analogy
    --------------------
    The penalty mirrors fast-spiking inhibitory interneurons in the DG that
    suppress firing of already-recruited mossy cells, forcing new patterns
    to recruit a different (non-overlapping) subset of granule cells.
    """

    def __init__(
        self,
        sparsity: float = 0.08,
        penalty_weight: float = 1.0,
        neurogenesis_threshold: float = 0.20,
        neurogenesis_new_neurons: int = 64,
    ) -> None:
        self.sparsity = sparsity
        self.penalty_weight = penalty_weight
        self.neurogenesis_threshold = neurogenesis_threshold
        self.neurogenesis_new_neurons = neurogenesis_new_neurons

        # task_id → boolean mask  [D]
        self.engram_masks: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Claim counter (how many tasks claim each neuron)
    # ------------------------------------------------------------------ #

    def _claim_counts(self, dim: int, device: torch.device) -> torch.Tensor:
        """Count how many existing engrams claim each neuron index."""
        counts = torch.zeros(dim, dtype=torch.float32, device=device)
        for mask in self.engram_masks.values():
            # mask might have different length if network was expanded mid-run
            length = min(len(mask), dim)
            counts[:length] += mask[:length].float()
        return counts

    # ------------------------------------------------------------------ #
    # Allocation
    # ------------------------------------------------------------------ #

    def allocate_engram(
        self,
        activations: torch.Tensor,
        task_id: int,
        model: Optional[object] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """Select top-k neurons with overlap penalty.

        Parameters
        ----------
        activations:
            1-D tensor of mean activation magnitudes ``[D]`` (output of
            ``SparseEngramEncoder._collect_mean_activations``).
        task_id:
            ID of the task being allocated (used for storage).
        model:
            The ``HCKNModel`` instance.  Passed in so that ``expand_backbone``
            can be called if neurogenesis is triggered.

        Returns
        -------
        Tuple[torch.Tensor, bool]
            ``(mask, neurogenesis_triggered)`` — boolean mask ``[D]`` and
            whether the network was grown.
        """
        device = activations.device
        D = len(activations)
        k = max(1, int(self.sparsity * D))

        counts = self._claim_counts(D, device)
        adjusted = activations - self.penalty_weight * counts

        # Initial candidate mask
        topk_idx = torch.topk(adjusted, k).indices
        candidate_mask = torch.zeros(D, dtype=torch.bool, device=device)
        candidate_mask[topk_idx] = True

        neurogenesis_fired = False

        # Check if neurogenesis is needed
        if self.engram_masks and should_trigger_neurogenesis(
            candidate_mask, self.engram_masks, self.neurogenesis_threshold
        ):
            neurogenesis_fired = True
            if model is not None:
                # Grow the network
                model.expand_backbone(self.neurogenesis_new_neurons)  # type: ignore[attr-defined]
                new_D = model.backbone.output_dim  # type: ignore[attr-defined]
                # Extend activations and counts to new dimension
                extra = torch.zeros(
                    new_D - D, dtype=activations.dtype, device=device
                )
                activations = torch.cat([activations, extra])
                # Re-run allocation: new neurons have no claims, high effective score
                D = new_D
                k = max(1, int(self.sparsity * D))
                counts = self._claim_counts(D, device)
                adjusted = activations - self.penalty_weight * counts
                topk_idx = torch.topk(adjusted, k).indices
                candidate_mask = torch.zeros(D, dtype=torch.bool, device=device)
                candidate_mask[topk_idx] = True

        self.engram_masks[task_id] = candidate_mask.clone()
        return candidate_mask, neurogenesis_fired

    # ------------------------------------------------------------------ #
    # Overlap statistics
    # ------------------------------------------------------------------ #

    def overlap_stats(self) -> Dict[str, float]:
        """Compute summary overlap statistics for all stored engrams.

        Returns a dict with keys:
        ``mean_overlap``, ``max_overlap``, ``separation_score``.
        """
        if len(self.engram_masks) < 2:
            return {"mean_overlap": 0.0, "max_overlap": 0.0, "separation_score": 1.0}

        matrix = compute_overlap_matrix(self.engram_masks)
        T = matrix.shape[0]
        # Off-diagonal entries only
        off_diag = matrix[np.triu_indices(T, k=1)]
        mean_ov = float(off_diag.mean()) if len(off_diag) > 0 else 0.0
        max_ov = float(off_diag.max()) if len(off_diag) > 0 else 0.0
        return {
            "mean_overlap": mean_ov,
            "max_overlap": max_ov,
            "separation_score": 1.0 - mean_ov,
        }

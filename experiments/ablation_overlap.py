"""
Ablation: Overlap With/Without Penalty + Neurogenesis
======================================================
Compares engram overlap statistics under three conditions:
1. Baseline: random top-k allocation (no penalty)
2. Penalty only: overlap-aware allocation, no neurogenesis
3. Full: overlap-aware + neurogenesis

Usage
-----
    python experiments/ablation_overlap.py
"""

from __future__ import annotations

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hckn.separation import OverlapAwareAllocator


def simulate_tasks(
    num_tasks: int = 20,
    neuron_dim: int = 128,
    sparsity: float = 0.08,
    seed: int = 42,
):
    """Generate random activation tensors for *num_tasks* tasks."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return [
        torch.rand(neuron_dim, generator=rng)
        for _ in range(num_tasks)
    ]


def run_condition(
    name: str,
    activations_list,
    penalty_weight: float,
    neurogenesis_threshold: float,
    use_neurogenesis: bool,
) -> None:
    allocator = OverlapAwareAllocator(
        sparsity=0.08,
        penalty_weight=penalty_weight,
        neurogenesis_threshold=neurogenesis_threshold if use_neurogenesis else 1.0,
        neurogenesis_new_neurons=64,
    )
    ng_count = 0
    for task_id, acts in enumerate(activations_list):
        # Expand acts if allocator grew the network on a previous task
        if allocator.engram_masks:
            current_dim = max(m.numel() for m in allocator.engram_masks.values())
            if len(acts) < current_dim:
                extra = torch.zeros(current_dim - len(acts))
                acts = torch.cat([acts, extra])

        _, ng = allocator.allocate_engram(acts, task_id)
        if ng:
            ng_count += 1

    stats = allocator.overlap_stats()
    print(
        f"  {name:35s}  "
        f"mean_overlap={stats['mean_overlap']*100:5.1f}%  "
        f"max_overlap={stats['max_overlap']*100:5.1f}%  "
        f"separation={stats['separation_score']*100:5.1f}%  "
        f"neurogenesis_events={ng_count}"
    )


def run_ablation() -> None:
    print("\n" + "=" * 80)
    print("  Ablation: Engram Overlap — Penalty and Neurogenesis")
    print("=" * 80)

    for num_tasks in [10, 20]:
        print(f"\n  [num_tasks = {num_tasks}]")
        acts = simulate_tasks(num_tasks=num_tasks)
        run_condition("Baseline (no penalty)",         acts, 0.0, 1.0, False)
        run_condition("Penalty only (no neurogenesis)", acts, 1.0, 1.0, False)
        run_condition("Full (penalty + neurogenesis)",  acts, 1.0, 0.20, True)
    print()


if __name__ == "__main__":
    run_ablation()

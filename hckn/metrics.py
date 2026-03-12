"""
HCKN v5 Metrics
===============
Centralised metrics for reporting and ablation studies.

Covers:
- Continual learning: ACC, BWT, FWT, per-task accuracy
- Engram overlap: pairwise Jaccard, mean/max overlap, separation score
- Router: routing accuracy
- Memory: params stored per task vs. full model size
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


# ====================================================================== #
# Continual Learning Accuracy Matrix
# ====================================================================== #

class AccuracyMatrix:
    """Tracks R[i][j]: accuracy on task j right after training task i."""

    def __init__(self, num_tasks: int) -> None:
        self.num_tasks = num_tasks
        self._mat = np.full((num_tasks, num_tasks), np.nan)

    def record(self, train_task: int, eval_task: int, acc: float) -> None:
        self._mat[train_task, eval_task] = acc

    def get(self, train_task: int, eval_task: int) -> Optional[float]:
        val = self._mat[train_task, eval_task]
        return None if np.isnan(val) else float(val)

    @property
    def matrix(self) -> np.ndarray:
        return self._mat.copy()

    # Standard CL metrics --------------------------------------------

    def acc(self) -> float:
        """Average accuracy across tasks after all training."""
        last = self.num_tasks - 1
        row = self._mat[last, :]
        valid = row[~np.isnan(row)]
        return float(valid.mean()) if len(valid) > 0 else 0.0

    def forgetting(self) -> float:
        """Mean forgetting: max accuracy - final accuracy for each task."""
        vals = []
        last = self.num_tasks - 1
        for j in range(self.num_tasks - 1):
            col = self._mat[:last + 1, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 1:
                vals.append(valid.max() - valid[-1])
        return float(np.mean(vals)) if vals else 0.0

    def bwt(self) -> float:
        """Backward Transfer = 1/T Σ (R_{T,i} − R_{i,i})."""
        last = self.num_tasks - 1
        vals = []
        for i in range(last):
            r_final = self._mat[last, i]
            r_diag = self._mat[i, i]
            if not np.isnan(r_final) and not np.isnan(r_diag):
                vals.append(r_final - r_diag)
        return float(np.mean(vals)) if vals else 0.0

    def fwt(self, baselines: Optional[List[float]] = None) -> float:
        """Forward Transfer = 1/(T-1) Σ (R_{i-1,i} − b_i).

        If *baselines* is None, uses zero (relative FWT).
        """
        vals = []
        for i in range(1, self.num_tasks):
            r_prev = self._mat[i - 1, i]
            if np.isnan(r_prev):
                continue
            b = 0.0 if baselines is None else baselines[i]
            vals.append(float(r_prev) - b)
        return float(np.mean(vals)) if vals else 0.0

    def print_table(self, title: str = "Accuracy Matrix") -> None:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
        header = "     " + "  ".join(f"T{j+1:02d}" for j in range(self.num_tasks))
        print(header)
        for i in range(self.num_tasks):
            row_str = f"T{i+1:02d} |"
            for j in range(self.num_tasks):
                val = self._mat[i, j]
                if np.isnan(val):
                    row_str += "    -"
                else:
                    row_str += f" {val*100:4.1f}"
            print(row_str)
        print()


# ====================================================================== #
# Memory Efficiency
# ====================================================================== #

def model_param_count(model: torch.nn.Module) -> int:
    """Total number of scalar parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def memory_efficiency_report(
    model: torch.nn.Module,
    encoder,  # SparseEngramEncoder
    num_tasks: int,
) -> Dict[str, float]:
    """Compare engram storage to full-model storage."""
    full_params = model_param_count(model)
    engram_params_list = [
        encoder.total_params_stored(tid)
        for tid in range(num_tasks)
        if tid in encoder.engrams
    ]
    if not engram_params_list:
        return {"full_params": full_params, "engram_params_per_task": 0.0,
                "fraction_per_task": 0.0}
    avg_engram = float(np.mean(engram_params_list))
    return {
        "full_params": full_params,
        "engram_params_per_task": avg_engram,
        "fraction_per_task": avg_engram / max(full_params, 1),
    }


# ====================================================================== #
# Results Summary Printer
# ====================================================================== #

def print_results_table(
    acc_matrix: AccuracyMatrix,
    router_accuracy: Optional[float] = None,
    overlap_stats: Optional[Dict[str, float]] = None,
    memory_stats: Optional[Dict[str, float]] = None,
    transfer_summary: Optional[Dict[str, float]] = None,
) -> None:
    """Print a formatted results summary table."""
    print("\n" + "=" * 60)
    print("  HCKN v5 — Results Summary")
    print("=" * 60)

    # CL metrics
    print(f"  ACC (final avg accuracy):  {acc_matrix.acc()*100:6.2f}%")
    print(f"  Forgetting:                {acc_matrix.forgetting()*100:6.2f}%")
    print(f"  BWT:                       {acc_matrix.bwt()*100:+6.2f}%")
    if transfer_summary:
        fwt = transfer_summary.get("FWT", acc_matrix.fwt())
        print(f"  FWT:                       {fwt*100:+6.2f}%")

    if router_accuracy is not None:
        print(f"  Router accuracy:           {router_accuracy*100:6.2f}%")

    if overlap_stats:
        print(f"  Mean engram overlap:       {overlap_stats.get('mean_overlap',0)*100:6.2f}%")
        print(f"  Max engram overlap:        {overlap_stats.get('max_overlap',0)*100:6.2f}%")
        print(f"  Separation score:          {overlap_stats.get('separation_score',1)*100:6.2f}%")

    if memory_stats:
        frac = memory_stats.get("fraction_per_task", 0)
        print(f"  Memory per task:           {frac*100:6.2f}% of full model")

    print("=" * 60 + "\n")

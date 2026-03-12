"""
Ablation: Router Accuracy vs. Distance Metric
==============================================
Compares Euclidean vs. Mahalanobis distance for the CA3 Router across
several tasks and prints accuracy for each configuration.

Usage
-----
    python experiments/ablation_router.py
"""

from __future__ import annotations

import sys
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hckn import CA3Router
from hckn.model import HCKNModel


def synthetic_tasks(
    num_tasks: int = 10,
    samples_per_task: int = 200,
    feature_dim: int = 128,
    seed: int = 42,
):
    """Generate synthetic Gaussian cluster data for ablation."""
    rng = np.random.RandomState(seed)
    train_data = []
    test_data = []
    centroids = rng.randn(num_tasks, feature_dim) * 5
    for t in range(num_tasks):
        train_feat = centroids[t] + rng.randn(samples_per_task, feature_dim)
        test_feat = centroids[t] + rng.randn(samples_per_task // 4, feature_dim)
        train_data.append(torch.tensor(train_feat, dtype=torch.float32))
        test_data.append(torch.tensor(test_feat, dtype=torch.float32))
    return train_data, test_data


def evaluate_router(metric: str, train_data, test_data) -> float:
    router = CA3Router(distance_metric=metric, temperature=1.0)
    for task_id, feats in enumerate(train_data):
        router.store_prototype(task_id, feats)

    correct = total = 0
    for task_id, feats in enumerate(test_data):
        preds = router.route(feats)
        correct += sum(p == task_id for p in preds)
        total += len(preds)
    return correct / max(total, 1)


def run_ablation() -> None:
    print("\n" + "=" * 50)
    print("  Ablation: Router Distance Metric")
    print("=" * 50)

    for num_tasks in [5, 10, 20]:
        train_data, test_data = synthetic_tasks(num_tasks=num_tasks, seed=42)
        for metric in ("euclidean", "mahalanobis"):
            acc = evaluate_router(metric, train_data, test_data)
            print(f"  Tasks={num_tasks:2d}  {metric:14s}  routing_acc={acc*100:.1f}%")
    print()


if __name__ == "__main__":
    run_ablation()

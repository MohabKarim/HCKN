"""
Ablation: FWT With/Without Two-Phase Training + EWC
=====================================================
Compares forward transfer under:
1. Single-phase training (no EWC)
2. Two-phase training without EWC
3. Two-phase training with EWC (full HCKN v5)

Uses synthetic data for fast iteration.

Usage
-----
    python experiments/ablation_transfer.py
"""

from __future__ import annotations

import sys
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hckn.model import HCKNModel
from hckn.engram import SparseEngramEncoder
from hckn.cls_system import TwoPhaseTrainer, EWCRegularizer, TransferMetrics


def make_synthetic_loaders(
    num_tasks: int = 5,
    samples_per_task: int = 300,
    input_dim: int = 64,
    classes_per_task: int = 5,
    batch_size: int = 32,
    seed: int = 42,
):
    rng = torch.Generator()
    rng.manual_seed(seed)
    loaders = []
    for t in range(num_tasks):
        x = torch.randn(samples_per_task, input_dim, generator=rng)
        y = torch.randint(0, classes_per_task, (samples_per_task,), generator=rng)
        ds = TensorDataset(x, y)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
    return loaders


def evaluate(model, task_id, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feats = model.backbone(x)
            logits = model.heads[task_id](feats)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def run_condition(
    name: str,
    loaders,
    input_dim: int,
    classes_per_task: int,
    num_epochs: int,
    use_two_phase: bool,
    use_ewc: bool,
    device: torch.device,
) -> None:
    torch.manual_seed(42)
    num_tasks = len(loaders)
    model = HCKNModel(input_dim, [256, 128], classes_per_task).to(device)
    ewc = EWCRegularizer(ewc_lambda=400.0) if use_ewc else None
    tm = TransferMetrics(num_tasks)
    baselines = []

    # Compute baselines: accuracy of fresh model on each task
    for t in range(num_tasks):
        fresh = HCKNModel(input_dim, [256, 128], classes_per_task).to(device)
        fresh.add_head()
        acc = evaluate(fresh, 0, loaders[t], device)
        baselines.append(acc)
        tm.set_baseline(t, acc)

    for task_id in range(num_tasks):
        model.add_head()
        phase_split = 0.6 if use_two_phase else 1.0
        trainer = TwoPhaseTrainer(
            model=model,
            task_id=task_id,
            dataloader=loaders[task_id],
            device=device,
            num_epochs=num_epochs,
            phase_split=phase_split,
            head_lr=1e-3,
            backbone_lr_ratio=0.1,
            ewc_regularizer=ewc if task_id > 0 else None,
        )
        trainer.train()

        # Record pre-training accuracy on NEXT task (FWT measurement point)
        if task_id < num_tasks - 1:
            acc_next = evaluate(model, task_id, loaders[task_id + 1], device)
            tm.record(task_id, task_id + 1, acc_next)

        # Record post-training accuracy on THIS task (diagonal)
        acc_self = evaluate(model, task_id, loaders[task_id], device)
        tm.record(task_id, task_id, acc_self)

        # Update EWC Fisher
        if ewc is not None:
            ewc.update_fisher(model, loaders[task_id], device)

        # Final accuracy on all past tasks
        for eval_id in range(task_id + 1):
            acc = evaluate(model, eval_id, loaders[eval_id], device)
            tm.record(num_tasks - 1, eval_id, acc)

    summary = tm.summary()
    print(
        f"  {name:40s}  "
        f"ACC={summary['ACC']*100:5.1f}%  "
        f"BWT={summary['BWT']*100:+5.1f}%  "
        f"FWT={summary['FWT']*100:+5.1f}%"
    )


def run_ablation() -> None:
    device = torch.device("cpu")
    print("\n" + "=" * 75)
    print("  Ablation: Forward Transfer — Two-Phase vs. EWC")
    print("=" * 75)

    loaders = make_synthetic_loaders(num_tasks=5, input_dim=64)

    run_condition("Single-phase, no EWC",    loaders, 64, 5, 10, False, False, device)
    run_condition("Two-phase, no EWC",       loaders, 64, 5, 10, True,  False, device)
    run_condition("Two-phase + EWC (HCKN v5)", loaders, 64, 5, 10, True, True, device)
    print()


if __name__ == "__main__":
    run_ablation()

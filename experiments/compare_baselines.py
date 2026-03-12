"""
Compare Baselines: SGD, EWC, HCKN v3 (Task-IL), HCKN v5 (Class-IL)
====================================================================
Runs all four methods on a small synthetic benchmark and prints a
comparison table.

Usage
-----
    python experiments/compare_baselines.py
"""

from __future__ import annotations

import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hckn import HCKNv5, HCKNConfig
from hckn.model import HCKNModel
from hckn.cls_system import EWCRegularizer
from hckn.metrics import AccuracyMatrix


# ====================================================================== #
# Shared synthetic data
# ====================================================================== #

def make_loaders(
    num_tasks=5, samples=200, input_dim=128, cpt=5, batch_size=32, seed=42
):
    g = torch.Generator().manual_seed(seed)
    loaders = []
    for _ in range(num_tasks):
        x = torch.randn(samples, input_dim, generator=g)
        y = torch.randint(0, cpt, (samples,), generator=g)
        loaders.append(DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True))
    return loaders


def eval_model(model, task_id, loader, device):
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


# ====================================================================== #
# Baselines
# ====================================================================== #

def run_sgd(loaders, input_dim, cpt, num_epochs, device):
    """Naive SGD — no continual learning mechanism."""
    torch.manual_seed(42)
    model = HCKNModel(input_dim, [256, 128], cpt).to(device)
    criterion = nn.CrossEntropyLoss()
    acc_mat = AccuracyMatrix(len(loaders))

    for task_id, loader in enumerate(loaders):
        model.add_head()
        opt = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(num_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                feats = model.backbone(x)
                loss = criterion(model.heads[task_id](feats), y)
                loss.backward()
                opt.step()
        for eval_id in range(task_id + 1):
            acc = eval_model(model, eval_id, loaders[eval_id], device)
            acc_mat.record(task_id, eval_id, acc)

    return acc_mat


def run_ewc_baseline(loaders, input_dim, cpt, num_epochs, device):
    """EWC baseline."""
    torch.manual_seed(42)
    model = HCKNModel(input_dim, [256, 128], cpt).to(device)
    ewc = EWCRegularizer(ewc_lambda=400.0)
    criterion = nn.CrossEntropyLoss()
    acc_mat = AccuracyMatrix(len(loaders))

    for task_id, loader in enumerate(loaders):
        model.add_head()
        opt = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(num_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                feats = model.backbone(x)
                loss = criterion(model.heads[task_id](feats), y)
                if task_id > 0:
                    loss = loss + ewc.penalty(model)
                loss.backward()
                opt.step()
        ewc.update_fisher(model, loader, device)
        for eval_id in range(task_id + 1):
            acc = eval_model(model, eval_id, loaders[eval_id], device)
            acc_mat.record(task_id, eval_id, acc)

    return acc_mat


def run_hckn_v3_task_il(loaders, input_dim, cpt, num_epochs, device):
    """HCKN v3 (Task-IL): engram gating with known task ID."""
    from hckn import SparseEngramEncoder
    torch.manual_seed(42)
    model = HCKNModel(input_dim, [256, 128], cpt).to(device)
    encoder = SparseEngramEncoder(sparsity=0.08)
    criterion = nn.CrossEntropyLoss()
    acc_mat = AccuracyMatrix(len(loaders))

    for task_id, loader in enumerate(loaders):
        model.add_head()
        opt = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(num_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                feats = model.backbone(x)
                loss = criterion(model.heads[task_id](feats), y)
                loss.backward()
                opt.step()
        encoder.form_engram(model, task_id, loader, device)
        # Evaluate with known task ID (Task-IL)
        for eval_id in range(task_id + 1):
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in loaders[eval_id]:
                    x, y = x.to(device), y.to(device)
                    logits = model(
                        x, task_id=eval_id,
                        engram_mask=encoder.engrams[eval_id].mask,
                        bias_shifts=encoder.engrams[eval_id].bias_shifts,
                    )
                    correct += (logits.argmax(1) == y).sum().item()
                    total += y.size(0)
            acc_mat.record(task_id, eval_id, correct / max(total, 1))

    return acc_mat


def run_hckn_v5_class_il(loaders, input_dim, cpt, num_epochs, device):
    """HCKN v5 (Class-IL): full system, router predicts task ID."""
    cfg = HCKNConfig(
        input_dim=input_dim,
        hidden_dims=[256, 128],
        classes_per_task=cpt,
        num_tasks=len(loaders),
        num_epochs=num_epochs,
        device="cpu",
        seed=42,
        engram_sparsity=0.08,
        router_distance="mahalanobis",
    )
    system = HCKNv5(cfg)
    acc_mat = AccuracyMatrix(len(loaders))

    for task_id, loader in enumerate(loaders):
        system.train_task(task_id, loader)
        for eval_id in range(task_id + 1):
            acc = system.evaluate_task(eval_id, loaders[eval_id], use_router=True)
            acc_mat.record(task_id, eval_id, acc)

    return acc_mat


# ====================================================================== #
# Main
# ====================================================================== #

def run_comparison() -> None:
    device = torch.device("cpu")
    num_epochs = 10
    loaders = make_loaders()

    print("\n" + "=" * 65)
    print("  Baseline Comparison — Synthetic Tasks")
    print("=" * 65)
    print(f"  {'Method':<35}  {'ACC':>6}  {'Forgetting':>10}  {'BWT':>6}")
    print("-" * 65)

    for name, fn in [
        ("SGD (naive)",          lambda: run_sgd(loaders, 128, 5, num_epochs, device)),
        ("EWC",                  lambda: run_ewc_baseline(loaders, 128, 5, num_epochs, device)),
        ("HCKN v3 (Task-IL)",    lambda: run_hckn_v3_task_il(loaders, 128, 5, num_epochs, device)),
        ("HCKN v5 (Class-IL)",   lambda: run_hckn_v5_class_il(loaders, 128, 5, num_epochs, device)),
    ]:
        mat = fn()
        print(
            f"  {name:<35}  "
            f"{mat.acc()*100:>5.1f}%  "
            f"{mat.forgetting()*100:>9.1f}%  "
            f"{mat.bwt()*100:>+5.1f}%"
        )

    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_comparison()

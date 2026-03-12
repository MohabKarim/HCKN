"""
Split-CIFAR-100 Benchmark Experiment
=====================================
Runs the full HCKN v5 pipeline on Split-CIFAR-100 (100 classes → 20 tasks
of 5 classes each) and prints a comprehensive results table.

Usage
-----
    python experiments/run_split_cifar100.py

The script downloads CIFAR-100 automatically to ``./data/`` if not present.
Results are printed to stdout.
"""

from __future__ import annotations

import sys
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Allow running from repo root or from experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hckn import HCKNv5, HCKNConfig
from hckn.metrics import AccuracyMatrix, print_results_table, memory_efficiency_report


# ====================================================================== #
# Helpers
# ====================================================================== #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_split_cifar100(
    data_root: str,
    num_tasks: int,
    classes_per_task: int,
    batch_size: int,
    seed: int = 42,
):
    """Build per-task train/test DataLoaders for Split-CIFAR-100.

    Returns
    -------
    Tuple[List[DataLoader], List[DataLoader]]
        ``(train_loaders, test_loaders)`` — one per task.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                              (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                              (0.2675, 0.2565, 0.2761)),
    ])

    train_full = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_full = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform_test
    )

    # Class order: shuffle with fixed seed for reproducibility
    rng = random.Random(seed)
    all_classes = list(range(100))
    rng.shuffle(all_classes)
    task_classes = [
        all_classes[i * classes_per_task: (i + 1) * classes_per_task]
        for i in range(num_tasks)
    ]

    def indices_for_classes(dataset, classes):
        cls_set = set(classes)
        return [i for i, (_, label) in enumerate(dataset) if label in cls_set]

    def remap_label(dataset, classes):
        """Wrap dataset so labels are 0..classes_per_task-1."""
        cls_to_local = {c: j for j, c in enumerate(classes)}

        class RemappedSubset(torch.utils.data.Dataset):
            def __init__(self, ds, indices, mapping):
                self.ds = ds
                self.indices = indices
                self.mapping = mapping

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                img, label = self.ds[self.indices[idx]]
                return img, self.mapping[label]

        idx = indices_for_classes(dataset, classes)
        return RemappedSubset(dataset, idx, cls_to_local)

    train_loaders = []
    test_loaders = []
    for classes in task_classes:
        tr_ds = remap_label(train_full, classes)
        te_ds = remap_label(test_full, classes)
        train_loaders.append(
            DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=0, pin_memory=False)
        )
        test_loaders.append(
            DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                       num_workers=0, pin_memory=False)
        )

    return train_loaders, test_loaders


# ====================================================================== #
# Main Experiment
# ====================================================================== #

def run_experiment(cfg: HCKNConfig) -> None:
    set_seed(cfg.seed)
    print(f"\n{'='*60}")
    print("  HCKN v5 — Split-CIFAR-100 Benchmark")
    print(f"  Tasks: {cfg.num_tasks}  |  Classes/task: {cfg.classes_per_task}")
    print(f"  Epochs/task: {cfg.num_epochs}  |  Device: {cfg.device}")
    print(f"{'='*60}\n")

    # Build dataloaders
    train_loaders, test_loaders = make_split_cifar100(
        data_root=cfg.data_root,
        num_tasks=cfg.num_tasks,
        classes_per_task=cfg.classes_per_task,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    # Build system
    system = HCKNv5(cfg)
    acc_matrix = AccuracyMatrix(cfg.num_tasks)

    # Sequential task training
    for task_id in range(cfg.num_tasks):
        print(f"[Task {task_id+1:02d}/{cfg.num_tasks}] Training ...", end=" ", flush=True)
        log = system.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],
            val_loader=test_loaders[task_id],
        )
        ng = " [neurogenesis!]" if log.get("neurogenesis") else ""
        final_loss = (log["phase1_losses"] + log["phase2_losses"])
        final_loss_val = final_loss[-1] if final_loss else float("nan")
        print(f"done  loss={final_loss_val:.3f}{ng}")

        # Evaluate all tasks seen so far (both Task-IL and Class-IL)
        for eval_id in range(task_id + 1):
            # Class-IL (uses router)
            acc_ci = system.evaluate_task(eval_id, test_loaders[eval_id], use_router=True)
            acc_matrix.record(task_id, eval_id, acc_ci)

    # -------------------------------------------------------------- #
    # Print results
    # -------------------------------------------------------------- #
    acc_matrix.print_table("HCKN v5 — Accuracy Matrix (Class-IL)")

    overlap_stats = system.allocator.overlap_stats()
    mem_stats = memory_efficiency_report(system.model, system.encoder, cfg.num_tasks)

    print_results_table(
        acc_matrix=acc_matrix,
        router_accuracy=system.router.routing_accuracy() or None,
        overlap_stats=overlap_stats,
        memory_stats=mem_stats,
    )

    # Neurogenesis summary
    ng_count = sum(system.neurogenesis_events)
    print(f"  Neurogenesis events: {ng_count}/{cfg.num_tasks} tasks")
    print(f"  Final backbone dim: {system.model.backbone.output_dim}")
    print()


if __name__ == "__main__":
    # Default: small fast run suitable for CPU-only environments
    cfg = HCKNConfig(
        num_tasks=20,
        classes_per_task=5,
        num_epochs=10,      # increase to 50+ for full benchmark
        batch_size=64,
        hidden_dims=[512, 256, 128],
        engram_sparsity=0.08,
        router_distance="mahalanobis",
        ewc_lambda=400.0,
        phase_split=0.6,
        neurogenesis_threshold=0.20,
        device="auto",
        seed=42,
        data_root="./data",
    )
    run_experiment(cfg)

"""
HCKN v5 Configuration
=====================
Centralised hyperparameter dataclass supporting easy sweeps over all
tunable knobs in the system.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class HCKNConfig:
    """Master configuration for HCKN v5.

    All hyper-parameters are collected here so that experiments and
    ablations can override individual fields without touching code.
    """

    # ------------------------------------------------------------------ #
    # Reproducibility
    # ------------------------------------------------------------------ #
    seed: int = 42

    # ------------------------------------------------------------------ #
    # Dataset / split
    # ------------------------------------------------------------------ #
    dataset: str = "split_cifar100"
    num_classes_total: int = 100
    num_tasks: int = 20          # 100 classes / 5 per task = 20 tasks
    classes_per_task: int = 5
    data_root: str = "./data"

    # ------------------------------------------------------------------ #
    # Model architecture
    # ------------------------------------------------------------------ #
    input_dim: int = 3 * 32 * 32   # CIFAR-100 flattened
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    use_conv_backbone: bool = False  # switch to simple MLP by default

    # ------------------------------------------------------------------ #
    # Sparse Engram Encoding (SEE) — Critique-0 / base
    # ------------------------------------------------------------------ #
    engram_sparsity: float = 0.08    # top-8 % neurons form the engram
    inhibition_factor: float = 0.3   # non-engram neurons multiplied by this
    bias_scale: float = 1.0          # scale applied to stored bias shifts

    # ------------------------------------------------------------------ #
    # Critique 1 — CA3 Router
    # ------------------------------------------------------------------ #
    router_distance: Literal["euclidean", "mahalanobis"] = "mahalanobis"
    router_temperature: float = 1.0
    router_uncertainty_threshold: float = 10.0   # flag "uncertain" above this

    # ------------------------------------------------------------------ #
    # Critique 2 — Overlap-Aware Allocation + Neurogenesis
    # ------------------------------------------------------------------ #
    overlap_penalty_weight: float = 1.0
    neurogenesis_threshold: float = 0.20    # Jaccard overlap above → grow neurons
    neurogenesis_new_neurons: int = 64

    # ------------------------------------------------------------------ #
    # Critique 3 — CLS Dual-Rate / Two-Phase / EWC
    # ------------------------------------------------------------------ #
    backbone_lr_ratio: float = 0.1        # backbone LR = head_lr * this
    head_lr: float = 1e-3
    ewc_lambda: float = 400.0
    phase_split: float = 0.6              # fraction of epochs in exploration phase
    num_epochs: int = 50
    batch_size: int = 64

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    device: str = "auto"   # "auto" → detect CUDA/MPS/CPU at runtime

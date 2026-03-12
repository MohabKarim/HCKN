"""
HCKN v5 Package
===============
Biologically-Inspired Continual Learning with Sparse Engram Encoding.

Quick start
-----------
>>> from hckn import HCKNv5, HCKNConfig
>>> cfg = HCKNConfig(num_tasks=20, num_epochs=50)
>>> system = HCKNv5(cfg)

Modules
-------
- ``config``      — HCKNConfig dataclass
- ``model``       — HCKNModel (backbone + per-task heads)
- ``engram``      — SparseEngramEncoder (SEE)
- ``router``      — CA3Router (Critique 1: Task Identity)
- ``separation``  — OverlapAwareAllocator + neurogenesis (Critique 2)
- ``cls_system``  — CLSDualSystem, TwoPhaseTrainer, EWCRegularizer (Critique 3)
- ``metrics``     — AccuracyMatrix, memory, transfer metrics
- ``hckn_v5``     — HCKNv5 integrated system
"""

from .config import HCKNConfig
from .model import HCKNModel, HCKNBackbone
from .engram import SparseEngramEncoder, EngramMemory
from .router import CA3Router
from .separation import OverlapAwareAllocator, compute_overlap_matrix, should_trigger_neurogenesis
from .cls_system import EWCRegularizer, CLSDualSystem, TwoPhaseTrainer, TransferMetrics
from .metrics import AccuracyMatrix, memory_efficiency_report, print_results_table
from .hckn_v5 import HCKNv5

__all__ = [
    "HCKNConfig",
    "HCKNModel",
    "HCKNBackbone",
    "SparseEngramEncoder",
    "EngramMemory",
    "CA3Router",
    "OverlapAwareAllocator",
    "compute_overlap_matrix",
    "should_trigger_neurogenesis",
    "EWCRegularizer",
    "CLSDualSystem",
    "TwoPhaseTrainer",
    "TransferMetrics",
    "AccuracyMatrix",
    "memory_efficiency_report",
    "print_results_table",
    "HCKNv5",
]

__version__ = "5.0.0"

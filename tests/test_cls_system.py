"""Unit tests for CLS dual system: EWCRegularizer, TwoPhaseTrainer, TransferMetrics."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hckn.model import HCKNModel
from hckn.cls_system import EWCRegularizer, CLSDualSystem, TwoPhaseTrainer, TransferMetrics


# ====================================================================== #
# Helpers
# ====================================================================== #

def small_model(input_dim=32, classes_per_task=4):
    model = HCKNModel(input_dim=input_dim, hidden_dims=[64, 32], classes_per_task=classes_per_task)
    model.add_head()
    return model


def small_loader(n=60, input_dim=32, classes_per_task=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, input_dim, generator=g)
    y = torch.randint(0, classes_per_task, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


# ====================================================================== #
# EWC
# ====================================================================== #

class TestEWCRegularizer:
    def test_zero_penalty_before_update(self):
        model = small_model()
        ewc = EWCRegularizer(ewc_lambda=1.0)
        loss = ewc.penalty(model)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_penalty_positive_after_update(self):
        device = torch.device("cpu")
        model = small_model()
        loader = small_loader()
        ewc = EWCRegularizer(ewc_lambda=400.0)
        ewc.update_fisher(model, loader, device)
        # Perturb model slightly
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        penalty = ewc.penalty(model)
        assert penalty.item() > 0.0

    def test_penalty_zero_at_optimum(self):
        """If model hasn't changed, EWC penalty should be near zero."""
        device = torch.device("cpu")
        model = small_model()
        loader = small_loader()
        ewc = EWCRegularizer(ewc_lambda=400.0)
        ewc.update_fisher(model, loader, device)
        # Don't move model
        penalty = ewc.penalty(model)
        assert penalty.item() == pytest.approx(0.0, abs=1e-5)

    def test_multiple_tasks_accumulate(self):
        device = torch.device("cpu")
        model = small_model()
        ewc = EWCRegularizer(ewc_lambda=1.0)
        for i in range(3):
            ewc.update_fisher(model, small_loader(seed=i), device)
        assert len(ewc._fishers) == 3


# ====================================================================== #
# CLSDualSystem
# ====================================================================== #

class TestCLSDualSystem:
    def test_different_learning_rates(self):
        model = small_model()
        cls = CLSDualSystem(model, task_id=0, head_lr=1e-3, backbone_lr_ratio=0.1)
        opt = cls.optim
        # Two param groups: backbone and head
        assert len(opt.param_groups) == 2
        backbone_lr = opt.param_groups[0]["lr"]
        head_lr = opt.param_groups[1]["lr"]
        assert abs(head_lr - 1e-3) < 1e-9
        assert abs(backbone_lr - 1e-4) < 1e-9

    def test_head_lr_higher_than_backbone(self):
        model = small_model()
        cls = CLSDualSystem(model, task_id=0, head_lr=5e-3, backbone_lr_ratio=0.2)
        opt = cls.optim
        backbone_lr = opt.param_groups[0]["lr"]
        head_lr = opt.param_groups[1]["lr"]
        assert head_lr > backbone_lr


# ====================================================================== #
# TwoPhaseTrainer
# ====================================================================== #

class TestTwoPhaseTrainer:
    def _setup(self):
        device = torch.device("cpu")
        model = small_model()
        loader = small_loader()
        return model, loader, device

    def test_exploration_returns_losses(self):
        model, loader, device = self._setup()
        trainer = TwoPhaseTrainer(
            model=model, task_id=0, dataloader=loader,
            device=device, num_epochs=3, phase_split=1.0,
        )
        losses = trainer.exploration_phase(3)
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)

    def test_crystallisation_returns_losses(self):
        model, loader, device = self._setup()
        # Need an engram mask for crystallisation
        mask = torch.zeros(32, dtype=torch.bool)
        mask[:3] = True
        bias = torch.zeros(3)
        trainer = TwoPhaseTrainer(
            model=model, task_id=0, dataloader=loader,
            device=device, num_epochs=2, phase_split=0.0,
            engram_mask=mask, bias_shifts=bias,
        )
        losses = trainer.crystallisation_phase(2)
        assert len(losses) == 2

    def test_full_train_dict_keys(self):
        model, loader, device = self._setup()
        trainer = TwoPhaseTrainer(
            model=model, task_id=0, dataloader=loader,
            device=device, num_epochs=4, phase_split=0.5,
        )
        result = trainer.train()
        assert "phase1_losses" in result
        assert "phase2_losses" in result

    def test_backbone_unfrozen_after_crystallisation(self):
        """Backbone params should be trainable after crystallisation phase."""
        model, loader, device = self._setup()
        mask = torch.zeros(32, dtype=torch.bool)
        mask[:3] = True
        trainer = TwoPhaseTrainer(
            model=model, task_id=0, dataloader=loader,
            device=device, num_epochs=2,
            engram_mask=mask, bias_shifts=torch.zeros(3),
        )
        trainer.crystallisation_phase(2)
        for p in model.backbone.parameters():
            assert p.requires_grad


# ====================================================================== #
# TransferMetrics
# ====================================================================== #

class TestTransferMetrics:
    def test_acc_basic(self):
        tm = TransferMetrics(3)
        # After all training (train_task=2):
        tm.record(2, 0, 0.9)
        tm.record(2, 1, 0.8)
        tm.record(2, 2, 0.7)
        assert abs(tm.compute_acc() - (0.9 + 0.8 + 0.7) / 3) < 1e-6

    def test_bwt_no_forgetting(self):
        tm = TransferMetrics(3)
        # Diagonal (right after training each task)
        tm.record(0, 0, 0.8)
        tm.record(1, 1, 0.9)
        tm.record(2, 2, 0.85)
        # Final accuracy equals diagonal → no forgetting
        tm.record(2, 0, 0.8)
        tm.record(2, 1, 0.9)
        assert abs(tm.compute_bwt()) < 1e-6

    def test_bwt_negative_when_forgetting(self):
        tm = TransferMetrics(2)
        tm.record(0, 0, 0.9)  # R[0][0]
        tm.record(1, 0, 0.5)  # R[1][0] — forgot
        tm.record(1, 1, 0.8)
        assert tm.compute_bwt() < 0.0

    def test_fwt_with_baseline(self):
        tm = TransferMetrics(3)
        tm.set_baseline(1, 0.2)
        tm.set_baseline(2, 0.15)
        tm.record(0, 1, 0.4)   # R[0][1]
        tm.record(1, 2, 0.35)  # R[1][2]
        fwt = tm.compute_fwt()
        # FWT = mean(0.4-0.2, 0.35-0.15) = mean(0.2, 0.2) = 0.2
        assert abs(fwt - 0.2) < 1e-6

    def test_summary_keys(self):
        tm = TransferMetrics(2)
        tm.record(0, 0, 0.7)
        tm.record(1, 0, 0.7)
        tm.record(1, 1, 0.8)
        s = tm.summary()
        assert "ACC" in s and "BWT" in s and "FWT" in s

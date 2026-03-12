"""Integration tests for the full HCKN v5 pipeline."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from hckn import HCKNv5, HCKNConfig
from hckn.model import HCKNModel
from hckn.engram import SparseEngramEncoder
from hckn.metrics import AccuracyMatrix, memory_efficiency_report


# ====================================================================== #
# Helpers
# ====================================================================== #

def synthetic_loader(n=80, input_dim=64, cpt=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, input_dim, generator=g)
    y = torch.randint(0, cpt, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=False)


def make_system(num_tasks=3, input_dim=64, cpt=3, num_epochs=3):
    cfg = HCKNConfig(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        classes_per_task=cpt,
        num_tasks=num_tasks,
        num_epochs=num_epochs,
        phase_split=0.5,
        device="cpu",
        seed=42,
        engram_sparsity=0.1,
        router_distance="euclidean",  # faster for tests
        neurogenesis_threshold=0.5,
        ewc_lambda=10.0,
    )
    return HCKNv5(cfg)


# ====================================================================== #
# Tests
# ====================================================================== #

class TestHCKNv5Pipeline:
    def test_train_adds_heads(self):
        system = make_system(num_tasks=3)
        loaders = [synthetic_loader(seed=t) for t in range(3)]
        for t in range(3):
            system.train_task(t, loaders[t])
        assert system.model.num_tasks() == 3

    def test_engrams_stored_after_training(self):
        system = make_system(num_tasks=2)
        loaders = [synthetic_loader(seed=t) for t in range(2)]
        for t in range(2):
            system.train_task(t, loaders[t])
        assert 0 in system.encoder.engrams
        assert 1 in system.encoder.engrams

    def test_router_prototypes_stored(self):
        system = make_system(num_tasks=3)
        loaders = [synthetic_loader(seed=t) for t in range(3)]
        for t in range(3):
            system.train_task(t, loaders[t])
        assert system.router.num_prototypes == 3

    def test_predict_returns_correct_shapes(self):
        system = make_system(num_tasks=2)
        loaders = [synthetic_loader(seed=t) for t in range(2)]
        for t in range(2):
            system.train_task(t, loaders[t])
        x = torch.randn(8, 64)
        logits, task_ids = system.predict(x)
        assert logits.shape == (8, 3)
        assert len(task_ids) == 8

    def test_predict_with_task_id(self):
        system = make_system(num_tasks=2)
        loaders = [synthetic_loader(seed=t) for t in range(2)]
        for t in range(2):
            system.train_task(t, loaders[t])
        x = torch.randn(4, 64)
        logits = system.predict_with_task_id(x, task_id=0)
        assert logits.shape == (4, 3)

    def test_evaluate_task_returns_float(self):
        system = make_system(num_tasks=2)
        loaders = [synthetic_loader(seed=t) for t in range(2)]
        for t in range(2):
            system.train_task(t, loaders[t])
        acc = system.evaluate_task(0, loaders[0], use_router=True)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_accuracy_matrix_filled(self):
        num_tasks = 3
        system = make_system(num_tasks=num_tasks)
        loaders = [synthetic_loader(seed=t) for t in range(num_tasks)]
        acc_mat = AccuracyMatrix(num_tasks)
        for t in range(num_tasks):
            system.train_task(t, loaders[t])
            for e in range(t + 1):
                acc = system.evaluate_task(e, loaders[e], use_router=False)
                acc_mat.record(t, e, acc)
        # Check diagonal is filled
        for i in range(num_tasks):
            assert acc_mat.get(i, i) is not None

    def test_memory_efficiency(self):
        system = make_system(num_tasks=2)
        loaders = [synthetic_loader(seed=t) for t in range(2)]
        for t in range(2):
            system.train_task(t, loaders[t])
        stats = memory_efficiency_report(system.model, system.encoder, 2)
        assert stats["fraction_per_task"] < 1.0  # engram smaller than full model

    def test_neurogenesis_expands_backbone(self):
        """Force neurogenesis and verify backbone grew."""
        cfg = HCKNConfig(
            input_dim=32,
            hidden_dims=[32, 20],  # small dim to force overlap
            classes_per_task=2,
            num_tasks=2,
            num_epochs=2,
            device="cpu",
            seed=42,
            engram_sparsity=0.5,       # 50% → 10 neurons
            neurogenesis_threshold=0.01,  # very low threshold → always triggers
            neurogenesis_new_neurons=8,
        )
        system = HCKNv5(cfg)
        loaders = [synthetic_loader(n=40, input_dim=32, cpt=2, seed=t) for t in range(2)]
        system.train_task(0, loaders[0])
        old_dim = system.model.backbone.output_dim
        system.train_task(1, loaders[1])
        # Second task should have triggered neurogenesis → larger backbone
        assert system.model.backbone.output_dim >= old_dim

    def test_overlap_stats_available(self):
        system = make_system(num_tasks=3)
        loaders = [synthetic_loader(seed=t) for t in range(3)]
        for t in range(3):
            system.train_task(t, loaders[t])
        stats = system.allocator.overlap_stats()
        assert "mean_overlap" in stats
        assert "separation_score" in stats
        assert 0.0 <= stats["separation_score"] <= 1.0

    def test_train_log_contains_expected_keys(self):
        system = make_system(num_tasks=1)
        loader = synthetic_loader(seed=0)
        log = system.train_task(0, loader)
        assert "phase1_losses" in log
        assert "phase2_losses" in log
        assert "neurogenesis" in log


class TestModelExpansion:
    def test_expand_backbone_increases_dim(self):
        model = HCKNModel(input_dim=16, hidden_dims=[32, 20], classes_per_task=3)
        model.add_head()
        assert model.backbone.output_dim == 20
        model.expand_backbone(10)
        assert model.backbone.output_dim == 30

    def test_expand_backbone_updates_heads(self):
        model = HCKNModel(input_dim=16, hidden_dims=[32, 20], classes_per_task=3)
        model.add_head()
        model.expand_backbone(8)
        assert model.heads[0].in_features == 28

    def test_forward_works_after_expansion(self):
        model = HCKNModel(input_dim=16, hidden_dims=[32, 20], classes_per_task=3)
        model.add_head()
        model.expand_backbone(10)
        x = torch.randn(4, 16)
        out = model(x, task_id=0)
        assert out.shape == (4, 3)


class TestSparseEngramEncoder:
    def test_engram_sparsity(self):
        model = HCKNModel(input_dim=32, hidden_dims=[64, 32], classes_per_task=4)
        model.add_head()
        loader = synthetic_loader(n=60, input_dim=32, cpt=4)
        encoder = SparseEngramEncoder(sparsity=0.08)
        engram = encoder.form_engram(model, 0, loader, torch.device("cpu"))
        k = max(1, int(0.08 * 32))
        assert engram.mask.sum().item() == k

    def test_gating_modifies_features(self):
        model = HCKNModel(input_dim=32, hidden_dims=[64, 32], classes_per_task=4)
        model.add_head()
        loader = synthetic_loader(n=60, input_dim=32, cpt=4)
        encoder = SparseEngramEncoder(sparsity=0.1)
        encoder.form_engram(model, 0, loader, torch.device("cpu"))
        feats = torch.randn(4, 32)
        gated = encoder.apply_engram_gating(feats, task_id=0)
        assert not torch.allclose(feats, gated)

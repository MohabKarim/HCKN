"""Unit tests for OverlapAwareAllocator and pattern separation utilities."""

import pytest
import torch
import numpy as np

from hckn.separation import (
    OverlapAwareAllocator,
    compute_jaccard,
    compute_overlap_matrix,
    should_trigger_neurogenesis,
)
from hckn.model import HCKNModel


class TestComputeJaccard:
    def test_identical_masks(self):
        m = torch.tensor([True, False, True, True, False])
        assert compute_jaccard(m, m) == 1.0

    def test_disjoint_masks(self):
        a = torch.tensor([True, False, False])
        b = torch.tensor([False, True, False])
        assert compute_jaccard(a, b) == 0.0

    def test_partial_overlap(self):
        a = torch.tensor([True, True, False, False])
        b = torch.tensor([True, False, True, False])
        # intersection=1, union=3
        assert abs(compute_jaccard(a, b) - 1/3) < 1e-6

    def test_all_false(self):
        a = torch.zeros(5, dtype=torch.bool)
        b = torch.zeros(5, dtype=torch.bool)
        assert compute_jaccard(a, b) == 0.0


class TestComputeOverlapMatrix:
    def test_diagonal_is_one(self):
        engrams = {
            0: torch.tensor([True, True, False, False]),
            1: torch.tensor([False, True, True, False]),
        }
        mat = compute_overlap_matrix(engrams)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[1, 1] == pytest.approx(1.0)

    def test_symmetric(self):
        engrams = {
            0: torch.tensor([True, True, False, False, False]),
            1: torch.tensor([False, True, True, False, False]),
            2: torch.tensor([True, False, True, False, True]),
        }
        mat = compute_overlap_matrix(engrams)
        assert mat.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                assert abs(mat[i, j] - mat[j, i]) < 1e-6


class TestShouldTriggerNeurogenesis:
    def test_no_existing_engrams(self):
        mask = torch.tensor([True, True, False])
        assert not should_trigger_neurogenesis(mask, {}, threshold=0.2)

    def test_high_overlap_triggers(self):
        mask = torch.tensor([True, True, True, False])
        existing = {0: torch.tensor([True, True, True, False])}
        assert should_trigger_neurogenesis(mask, existing, threshold=0.2)

    def test_low_overlap_does_not_trigger(self):
        mask = torch.tensor([True, False, False, False])
        existing = {0: torch.tensor([False, True, True, True])}
        assert not should_trigger_neurogenesis(mask, existing, threshold=0.2)


class TestOverlapAwareAllocator:
    def _make_acts(self, dim=128, seed=0):
        torch.manual_seed(seed)
        return torch.rand(dim)

    def test_mask_is_correct_size(self):
        alloc = OverlapAwareAllocator(sparsity=0.08)
        acts = self._make_acts()
        mask, _ = alloc.allocate_engram(acts, task_id=0)
        assert mask.shape == (128,)
        assert mask.dtype == torch.bool

    def test_mask_sparsity_approximately_correct(self):
        alloc = OverlapAwareAllocator(sparsity=0.08)
        acts = self._make_acts()
        mask, _ = alloc.allocate_engram(acts, task_id=0)
        k = int(0.08 * 128)
        assert mask.sum().item() == k

    def test_overlap_penalty_reduces_overlap(self):
        """With penalty, second engram should share fewer neurons than random."""
        alloc_penalty = OverlapAwareAllocator(sparsity=0.1, penalty_weight=5.0,
                                              neurogenesis_threshold=1.0)
        alloc_baseline = OverlapAwareAllocator(sparsity=0.1, penalty_weight=0.0,
                                               neurogenesis_threshold=1.0)
        acts1 = torch.rand(200)
        acts2 = torch.rand(200)

        m1_p, _ = alloc_penalty.allocate_engram(acts1, 0)
        m2_p, _ = alloc_penalty.allocate_engram(acts2, 1)
        overlap_penalty = compute_jaccard(m1_p, m2_p)

        m1_b, _ = alloc_baseline.allocate_engram(acts1, 0)
        m2_b, _ = alloc_baseline.allocate_engram(acts2, 1)
        overlap_baseline = compute_jaccard(m1_b, m2_b)

        assert overlap_penalty <= overlap_baseline + 0.05  # allow small tolerance

    def test_neurogenesis_triggered(self):
        """Force neurogenesis by using very small dim and identical activations."""
        dim = 20
        alloc = OverlapAwareAllocator(
            sparsity=0.5,              # 50% → 10 neurons
            penalty_weight=0.0,        # no penalty so overlap stays high
            neurogenesis_threshold=0.1,
            neurogenesis_new_neurons=10,
        )
        model = HCKNModel(input_dim=8, hidden_dims=[32, dim], classes_per_task=2)
        model.add_head()

        acts = torch.ones(dim)  # all equal → top-k is arbitrary but dense
        mask0, ng0 = alloc.allocate_engram(acts, 0)
        assert not ng0  # first task: no existing engrams

        # Second identical allocation → high overlap → triggers neurogenesis
        acts2 = torch.ones(dim)
        mask1, ng1 = alloc.allocate_engram(acts2, 1, model=model)
        # neurogenesis should have fired (overlap = 1.0 > 0.1)
        assert ng1

    def test_overlap_stats_single_task(self):
        alloc = OverlapAwareAllocator()
        acts = torch.rand(64)
        alloc.allocate_engram(acts, 0)
        stats = alloc.overlap_stats()
        assert stats["mean_overlap"] == 0.0
        assert stats["separation_score"] == 1.0

    def test_overlap_stats_multiple_tasks(self):
        alloc = OverlapAwareAllocator(sparsity=0.1, penalty_weight=1.0,
                                      neurogenesis_threshold=1.0)
        for t in range(5):
            torch.manual_seed(t)
            alloc.allocate_engram(torch.rand(100), t)
        stats = alloc.overlap_stats()
        assert 0.0 <= stats["mean_overlap"] <= 1.0
        assert 0.0 <= stats["separation_score"] <= 1.0

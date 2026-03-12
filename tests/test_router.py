"""Unit tests for CA3Router."""

import pytest
import torch
import numpy as np

from hckn.router import CA3Router


def _make_clustered_features(num_tasks: int, n_per_task: int = 100, dim: int = 64, seed: int = 0):
    """Generate clustered Gaussian features."""
    rng = np.random.RandomState(seed)
    centroids = rng.randn(num_tasks, dim) * 10
    all_feats = []
    all_labels = []
    for t in range(num_tasks):
        f = centroids[t] + rng.randn(n_per_task, dim) * 0.5
        all_feats.append(torch.tensor(f, dtype=torch.float32))
        all_labels.extend([t] * n_per_task)
    return all_feats, all_labels


class TestCA3RouterEuclidean:
    def setup_method(self):
        self.router = CA3Router(distance_metric="euclidean")

    def test_single_prototype(self):
        feats = torch.randn(50, 32)
        self.router.store_prototype(0, feats)
        assert self.router.num_prototypes == 1

    def test_routes_to_nearest(self):
        task_feats, _ = _make_clustered_features(3)
        for t, f in enumerate(task_feats):
            self.router.store_prototype(t, f)
        # Query near centroid 1
        query = task_feats[1][:5]
        preds = self.router.route(query)
        assert all(p == 1 for p in preds)

    def test_routing_accuracy_perfect(self):
        task_feats, _ = _make_clustered_features(5)
        for t, f in enumerate(task_feats):
            self.router.store_prototype(t, f)
        for t, f in enumerate(task_feats):
            preds = self.router.route(f)
            self.router.update_accuracy(preds, [t] * len(preds))
        assert self.router.routing_accuracy() > 0.90

    def test_confidence_output_shape(self):
        feats = torch.randn(50, 16)
        self.router.store_prototype(0, feats)
        self.router.store_prototype(1, feats + 100)
        query = torch.randn(4, 16)
        task_ids, confs = self.router.route_with_confidence(query)
        assert len(task_ids) == 4
        assert len(confs) == 4
        assert all(0.0 <= c <= 1.0 for c in confs)

    def test_no_prototypes_raises(self):
        router = CA3Router()
        with pytest.raises(RuntimeError):
            router.route(torch.randn(2, 8))

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            CA3Router(distance_metric="cosine")

    def test_reset_accuracy(self):
        feats = torch.randn(20, 8)
        self.router.store_prototype(0, feats)
        self.router.update_accuracy([0, 0], [0, 1])
        self.router.reset_accuracy()
        assert self.router.routing_accuracy() == 0.0


class TestCA3RouterMahalanobis:
    def test_routes_correctly(self):
        router = CA3Router(distance_metric="mahalanobis")
        task_feats, _ = _make_clustered_features(4, n_per_task=200, dim=32)
        for t, f in enumerate(task_feats):
            router.store_prototype(t, f)
        # Query near centroid 2
        preds = router.route(task_feats[2][:10])
        assert sum(p == 2 for p in preds) >= 8  # allow 2 misses

    def test_singular_covariance_fallback(self):
        """All features identical → covariance is singular → fallback to Euclidean."""
        router = CA3Router(distance_metric="mahalanobis")
        feats = torch.ones(10, 8)  # rank-0 covariance
        router.store_prototype(0, feats)
        router.store_prototype(1, feats + 5)
        preds = router.route(torch.ones(3, 8))
        # Should not crash; should return some predictions
        assert len(preds) == 3

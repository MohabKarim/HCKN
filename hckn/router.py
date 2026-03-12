"""
CA3-Inspired Prototype Router  (Critique 1 — Task Identity)
============================================================
Converts HCKN from Task-IL to Class-IL: the router predicts the task ID
from input features alone, without any external label.

Neuroscience background
-----------------------
CA3 in the hippocampus performs **pattern completion**: a partial or noisy
input cue is compared against stored attractor basins (memory prototypes).
Recurrent collateral connections pull the network state toward the nearest
attractor, retrieving the full stored pattern.  No external context label
is required — the match is driven entirely by feature similarity.
(Rolls, Progress in Neurobiology 2013; Nakazawa et al., Science 2002)

The router stores one prototype centroid per task (the mean feature
vector from that task's training/validation data) and routes inference
samples to the nearest prototype via Mahalanobis or Euclidean distance.
If the nearest-prototype distance exceeds a threshold the router flags the
sample as "uncertain" — analogous to CA3/DG pattern separation detecting
novelty that does not fit any existing memory trace.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class CA3Router:
    """Prototype-distance task router inspired by CA3 pattern completion.

    Parameters
    ----------
    distance_metric:
        ``'mahalanobis'`` (default) or ``'euclidean'``.
    temperature:
        Softmax temperature for confidence scores (lower → sharper).
    uncertainty_threshold:
        If the minimum distance to any prototype exceeds this value the
        router returns confidence < 0.5 and flags the sample as uncertain.
    """

    def __init__(
        self,
        distance_metric: str = "mahalanobis",
        temperature: float = 1.0,
        uncertainty_threshold: float = 10.0,
    ) -> None:
        if distance_metric not in ("mahalanobis", "euclidean"):
            raise ValueError(
                f"distance_metric must be 'mahalanobis' or 'euclidean', "
                f"got '{distance_metric}'"
            )
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.uncertainty_threshold = uncertainty_threshold

        # task_id → centroid  [D]  (stored on CPU as numpy for scipy compat)
        self._centroids: Dict[int, np.ndarray] = {}
        # task_id → inverse covariance  [D, D]
        self._inv_covs: Dict[int, np.ndarray] = {}
        # Routing history for accuracy tracking
        self._correct: int = 0
        self._total: int = 0

    # ------------------------------------------------------------------ #
    # Prototype storage
    # ------------------------------------------------------------------ #

    def store_prototype(
        self,
        task_id: int,
        features: torch.Tensor,
    ) -> None:
        """Compute and store the centroid for *task_id*.

        Parameters
        ----------
        task_id:
            Integer task index.
        features:
            Feature matrix ``[N, D]`` from the task's training or
            validation data (backbone outputs).

        Neuroscience analogy
        --------------------
        Each centroid represents the CA3 attractor basin centre for the
        episodic memory of that task.  Storing it is analogous to the
        consolidation of spatial context representations in place cells.
        """
        feats_np = features.detach().cpu().numpy().astype(np.float64)
        centroid = feats_np.mean(axis=0)
        self._centroids[task_id] = centroid

        if self.distance_metric == "mahalanobis":
            self._inv_covs[task_id] = self._compute_inv_cov(feats_np)

    def _compute_inv_cov(self, feats: np.ndarray) -> np.ndarray:
        """Regularised inverse covariance matrix (falls back to identity)."""
        try:
            cov = np.cov(feats, rowvar=False)
            if cov.ndim == 0:  # scalar (single feature)
                cov = np.array([[float(cov)]])
            # Add small ridge regularisation for numerical stability
            reg = 1e-6 * np.eye(cov.shape[0])
            inv_cov = np.linalg.inv(cov + reg)
            return inv_cov
        except np.linalg.LinAlgError:
            D = feats.shape[1]
            return np.eye(D)

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def route(self, features: torch.Tensor) -> List[int]:
        """Predict task IDs for a batch of feature vectors.

        Parameters
        ----------
        features:
            ``[B, D]`` backbone feature tensor.

        Returns
        -------
        List[int]
            Predicted task ID for each sample in the batch.
        """
        task_ids, _ = self._route_internal(features)
        return task_ids

    def route_with_confidence(
        self, features: torch.Tensor
    ) -> Tuple[List[int], List[float]]:
        """Predict task IDs and return per-sample confidence scores.

        Confidence is computed as a softmax over negative distances:
        high confidence means the sample is clearly closest to one centroid.
        A confidence < 0.5 and distance > threshold indicates novelty.

        Returns
        -------
        Tuple[List[int], List[float]]
            ``(task_ids, confidences)`` — one entry per batch element.
        """
        return self._route_internal(features)

    def _route_internal(
        self, features: torch.Tensor
    ) -> Tuple[List[int], List[float]]:
        if not self._centroids:
            raise RuntimeError("No prototypes stored — call store_prototype first.")

        feats_np = features.detach().cpu().numpy().astype(np.float64)
        task_ids_stored = sorted(self._centroids.keys())

        pred_tasks: List[int] = []
        confidences: List[float] = []

        for feat in feats_np:
            dists = np.array([
                self._distance(feat, tid) for tid in task_ids_stored
            ])  # [T]
            best_idx = int(np.argmin(dists))
            pred_task = task_ids_stored[best_idx]
            pred_tasks.append(pred_task)

            # Softmax over negative distances (lower distance → higher score)
            scores = np.exp(-dists / self.temperature)
            scores /= scores.sum() + 1e-12
            confidence = float(scores[best_idx])
            confidences.append(confidence)

        return pred_tasks, confidences

    def _distance(self, feat: np.ndarray, task_id: int) -> float:
        """Compute distance from *feat* to the centroid of *task_id*."""
        centroid = self._centroids[task_id]
        diff = feat - centroid
        if self.distance_metric == "mahalanobis":
            inv_cov = self._inv_covs.get(task_id)
            if inv_cov is not None:
                return float(np.sqrt(np.maximum(diff @ inv_cov @ diff, 0.0)))
        # Euclidean fallback
        return float(np.sqrt(np.dot(diff, diff)))

    # ------------------------------------------------------------------ #
    # Accuracy tracking
    # ------------------------------------------------------------------ #

    def update_accuracy(
        self, predicted_task_ids: List[int], true_task_ids: List[int]
    ) -> None:
        """Record routing predictions for accuracy computation."""
        for p, t in zip(predicted_task_ids, true_task_ids):
            self._total += 1
            if p == t:
                self._correct += 1

    def routing_accuracy(self) -> float:
        """Return cumulative routing accuracy (fraction correct)."""
        if self._total == 0:
            return 0.0
        return self._correct / self._total

    def reset_accuracy(self) -> None:
        self._correct = 0
        self._total = 0

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #

    @property
    def num_prototypes(self) -> int:
        return len(self._centroids)

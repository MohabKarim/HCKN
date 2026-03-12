"""
Hippocampal Replay Buffer  (v6 — Mechanism 1: Sharp-Wave Ripples)
=================================================================
Implements DER++ (Dark Experience Replay++) style memory replay to
prevent catastrophic forgetting.

Neuroscience background
-----------------------
During slow-wave sleep (SWS), the hippocampus re-activates memories via
**sharp-wave ripple** (SWR) events.  These replay sequences are transmitted
to the neocortex, which uses the replayed patterns to consolidate them into
long-term storage.  Critically, the brain replays ~50% old content during
consolidation — ensuring old memories are regularly re-practiced alongside
new learning.  Without this replay, each new task overwrites old cortical
representations (Diekelmann & Born, Nature Reviews Neuroscience 2010).

DER++ analogy
-------------
Instead of literally sleeping, we maintain a fixed-size ring buffer of
past experiences.  At each training step, a batch of "replayed" memories
is drawn from the buffer and added to the current loss.  The buffer also
stores **soft logits** (dark knowledge) and **feature vectors** at
storage time:

- Logit distillation (β · MSE): preserves the decision-boundary geometry
  of old tasks — akin to the brain re-consolidating *how* it categorised
  past stimuli, not just whether it saw them.
- Feature distillation (γ · MSE): directly prevents backbone drift by
  constraining the internal representation to remain stable for old
  inputs — the neural equivalent of synaptic tagging (Frey & Morris, 1997).

References
----------
Buzzega et al., "Dark Experience for General Continual Learning", NeurIPS 2020.
Diekelmann & Born, "The memory function of sleep", Nature Reviews Neuroscience 2010.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch


class HippocampalReplayBuffer:
    """Fixed-size experience replay buffer with logit and feature storage.

    Implements **reservoir sampling** to ensure balanced task representation
    over time: when the buffer is full, each new sample has an equal
    probability of replacing any existing entry.  This mirrors the
    hippocampus's capacity-limited but temporally balanced replay.

    Parameters
    ----------
    buffer_size:
        Maximum number of ``(input, label, task_id, logits, features)``
        tuples stored.  Default 2000 (≈20 tasks × 5 classes × 20 samples).
    device:
        Compute device on which tensors are reconstructed during sampling.

    Neuroscience analogy
    --------------------
    The bounded buffer capacity mirrors the limited hippocampal workspace.
    Reservoir sampling ensures all past tasks are represented proportionally
    regardless of their recency — analogous to the hippocampus tagging
    memories by recency *and* importance rather than overwriting old traces.
    """

    def __init__(
        self,
        buffer_size: int = 2000,
        device: Optional[torch.device] = None,
    ) -> None:
        self.buffer_size = buffer_size
        self.device = device or torch.device("cpu")

        # Ring-buffer storage (list of dicts for flexibility)
        self._inputs: List[torch.Tensor] = []
        self._labels: List[int] = []
        self._task_ids: List[int] = []
        self._logits: List[torch.Tensor] = []
        self._features: List[torch.Tensor] = []

        # Total number of items ever seen (for reservoir sampling)
        self._num_seen: int = 0

    # ------------------------------------------------------------------ #
    # Buffer update
    # ------------------------------------------------------------------ #

    def update_buffer(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
        logits: torch.Tensor,
        features: torch.Tensor,
    ) -> None:
        """Add a batch of new samples to the buffer using reservoir sampling.

        Each call presents a new batch.  Reservoir sampling maintains a
        uniform distribution over all *ever-seen* samples.

        Parameters
        ----------
        inputs:
            Raw input batch ``[B, *]`` (moved to CPU for storage).
        labels:
            Class labels ``[B]``.
        task_id:
            Integer task index for all samples in this batch.
        logits:
            Model soft outputs at storage time ``[B, C]``.  These are the
            "dark knowledge" targets for logit distillation.
        features:
            Backbone feature vectors at storage time ``[B, D]``.  Used for
            feature distillation.

        Neuroscience analogy
        --------------------
        SWR-triggered replay is selective: only a fraction of experiences
        are replayed.  Reservoir sampling provides a principled way to
        decide which memories to retain as new ones arrive.
        """
        inputs_cpu = inputs.detach().cpu()
        logits_cpu = logits.detach().cpu()
        features_cpu = features.detach().cpu()
        labels_cpu = labels.detach().cpu()

        batch_size = inputs_cpu.size(0)

        for i in range(batch_size):
            self._num_seen += 1
            if len(self._inputs) < self.buffer_size:
                # Buffer not yet full — simply append
                self._inputs.append(inputs_cpu[i])
                self._labels.append(int(labels_cpu[i].item()))
                self._task_ids.append(task_id)
                self._logits.append(logits_cpu[i])
                self._features.append(features_cpu[i])
            else:
                # Reservoir sampling: replace a random existing entry
                replace_idx = random.randint(0, self._num_seen - 1)
                if replace_idx < self.buffer_size:
                    self._inputs[replace_idx] = inputs_cpu[i]
                    self._labels[replace_idx] = int(labels_cpu[i].item())
                    self._task_ids[replace_idx] = task_id
                    self._logits[replace_idx] = logits_cpu[i]
                    self._features[replace_idx] = features_cpu[i]

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def sample_replay_batch(
        self,
        batch_size: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor, torch.Tensor]]:
        """Sample a mixed batch of past experiences for replay training.

        Samples are drawn uniformly at random — **not** stratified by task.
        Reservoir sampling during insertion already ensures approximate
        balance across tasks.

        Parameters
        ----------
        batch_size:
            Number of replay samples to draw.

        Returns
        -------
        ``(inputs, labels, task_ids, logits, features)`` or ``None`` if the
        buffer is empty.

        Neuroscience analogy
        --------------------
        During SWR-triggered replay in slow-wave sleep, the hippocampus
        reactivates a subset of past experiences in compressed time, allowing
        the neocortex to consolidate them.  This method is called once per
        training step — the ''mini-sleep'' that counteracts forgetting.
        """
        n = len(self._inputs)
        if n == 0:
            return None

        actual_batch = min(batch_size, n)
        indices = random.sample(range(n), actual_batch)

        inputs = torch.stack([self._inputs[i] for i in indices]).to(self.device)
        labels = torch.tensor([self._labels[i] for i in indices],
                               dtype=torch.long, device=self.device)
        task_ids = [self._task_ids[i] for i in indices]
        logits = torch.stack([self._logits[i] for i in indices]).to(self.device)
        features = torch.stack([self._features[i] for i in indices]).to(self.device)

        return inputs, labels, task_ids, logits, features

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        """Current number of entries in the buffer."""
        return len(self._inputs)

    @property
    def num_seen(self) -> int:
        """Total number of samples ever presented (for reservoir stats)."""
        return self._num_seen

    def task_counts(self) -> dict:
        """Return a dict mapping task_id → count in the buffer."""
        counts: dict = {}
        for tid in self._task_ids:
            counts[tid] = counts.get(tid, 0) + 1
        return counts

    def __repr__(self) -> str:
        return (
            f"HippocampalReplayBuffer("
            f"size={self.size}/{self.buffer_size}, "
            f"tasks={sorted(self.task_counts().keys())})"
        )

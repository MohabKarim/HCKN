"""
HCKN v5 Base Model
==================
Backbone + per-task classification heads.

The backbone is a simple MLP (configurable hidden dimensions).  Each task
gets its own linear output head that is appended dynamically as tasks
arrive.  The backbone can be grown (neurogenesis) by appending extra
hidden units at the *last* hidden layer.

Neuroscience analogy
--------------------
The backbone corresponds to the neocortex — a shared representation
that evolves slowly across tasks.  The per-task heads correspond to
hippocampal-to-prefrontal projection pathways that are fast and
task-specific.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional


class HCKNBackbone(nn.Module):
    """Shared MLP feature extractor (neocortex analog).

    Parameters
    ----------
    input_dim:
        Dimensionality of the flattened input.
    hidden_dims:
        List of hidden-layer widths, e.g. ``[512, 256, 128]``.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature vector for a batch of inputs."""
        x = x.view(x.size(0), -1)
        return self.layers(x)

    def get_linear_layers(self) -> List[nn.Linear]:
        """Return all Linear sub-modules in order."""
        return [m for m in self.layers if isinstance(m, nn.Linear)]

    def freeze_early_layers(self, num_layers_to_freeze: int) -> None:
        """Freeze the first *num_layers_to_freeze* linear layers (perineuronal nets).

        Once early-layer representations have been consolidated across
        multiple tasks, wrapping them in ''perineuronal nets'' prevents
        further synaptic modification — leaving later layers plastic for
        new learning.

        Neuroscience analogy
        --------------------
        Perineuronal nets (PNNs) are extracellular matrix structures that
        physically surround stabilised neurons, closing the critical period
        for that cortical layer (Sorg et al., J Neurosci 2016).  Early
        sensory cortex closes first; higher association areas remain plastic
        longer.  This creates a stability-plasticity hierarchy that is the
        key to life-long learning.

        Parameters
        ----------
        num_layers_to_freeze:
            Number of linear layers (counting from the input end) to make
            non-trainable.  Clamped to ``len(linear_layers) - 1`` so at
            least one layer stays plastic.
        """
        linears = self.get_linear_layers()
        n = min(max(0, num_layers_to_freeze), len(linears) - 1)
        for layer in linears[:n]:
            layer.requires_grad_(False)

    def unfreeze_all(self) -> None:
        """Unfreeze all layers (re-open plasticity for exploration phase).

        Neuroscience analogy
        --------------------
        Memory reconsolidation transiently de-stabilises synapses, making
        them labile again before re-locking.  Similarly, we temporarily
        remove the PNN-equivalent freezing at the start of exploration so
        the backbone can adapt to the new task.
        """
        for param in self.parameters():
            param.requires_grad_(True)

    def expand_last_layer(self, num_new_neurons: int) -> None:
        """Grow the last hidden layer by *num_new_neurons* (neurogenesis).

        Existing weights are preserved; new rows/columns are initialised
        with He-normal initialisation.

        Neuroscience analogy
        --------------------
        Adult neurogenesis in the dentate gyrus adds new granule cells that
        provide fresh, uncontaminated representational capacity when existing
        cells are saturated (Aimone et al., Neuron 2011).
        """
        # Identify the last Linear and the one before the output head
        linears = self.get_linear_layers()
        last_linear = linears[-1]

        old_out = last_linear.out_features
        new_out = old_out + num_new_neurons

        # Build expanded layer
        new_layer = nn.Linear(last_linear.in_features, new_out,
                              bias=last_linear.bias is not None)
        # Copy existing weights
        with torch.no_grad():
            new_layer.weight[:old_out] = last_linear.weight
            nn.init.kaiming_normal_(new_layer.weight[old_out:])
            if last_linear.bias is not None:
                new_layer.bias[:old_out] = last_linear.bias
                nn.init.zeros_(new_layer.bias[old_out:])

        # Replace in the sequential container
        # Find the index of last_linear inside self.layers
        idx = None
        for i, m in enumerate(self.layers):
            if m is last_linear:
                idx = i
                break
        assert idx is not None
        self.layers[idx] = new_layer  # type: ignore[index]
        self.hidden_dims[-1] = new_out
        self.output_dim = new_out


class HCKNModel(nn.Module):
    """Full HCKN model: shared backbone + dynamic per-task heads.

    Parameters
    ----------
    input_dim:
        Flattened input size.
    hidden_dims:
        Backbone hidden layer widths.
    classes_per_task:
        Output size for each task head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        classes_per_task: int,
    ) -> None:
        super().__init__()
        self.backbone = HCKNBackbone(input_dim, hidden_dims)
        self.classes_per_task = classes_per_task
        self.heads: nn.ModuleList = nn.ModuleList()

    # ------------------------------------------------------------------ #
    # Head management
    # ------------------------------------------------------------------ #

    def add_head(self) -> int:
        """Add a new linear classification head and return its index."""
        head = nn.Linear(self.backbone.output_dim, self.classes_per_task)
        nn.init.kaiming_normal_(head.weight)
        nn.init.zeros_(head.bias)
        self.heads.append(head)
        return len(self.heads) - 1

    def num_tasks(self) -> int:
        return len(self.heads)

    # ------------------------------------------------------------------ #
    # Forward passes
    # ------------------------------------------------------------------ #

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone feature vector (before any head)."""
        return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
        task_id: int,
        engram_mask: Optional[torch.Tensor] = None,
        bias_shifts: Optional[torch.Tensor] = None,
        inhibition_factor: float = 0.3,
    ) -> torch.Tensor:
        """Full forward pass with optional engram gating.

        Parameters
        ----------
        x:
            Input batch ``[B, *]``.
        task_id:
            Which task head to use.
        engram_mask:
            Boolean tensor of shape ``[hidden_dim_last]`` indicating engram
            neurons.  ``None`` → no gating (exploration phase).
        bias_shifts:
            Float tensor ``[num_engram_neurons]``.  Added to engram-neuron
            activations.
        inhibition_factor:
            Multiplicative suppression applied to *non*-engram neurons.

        Neuroscience analogy
        --------------------
        Non-engram neurons are suppressed by inhibitory interneurons
        (Stefanelli et al., Nature Neuroscience 2016).  Engram neurons
        receive extra excitatory drive that reactivates the stored memory
        trace (Josselyn & Tonegawa, Science 2020).
        """
        features = self.backbone(x)

        if engram_mask is not None:
            feat_dim = features.shape[1]
            mask_dim = engram_mask.shape[0]
            if feat_dim > mask_dim:
                # Backbone grew after this engram was stored — pad with False
                padded_mask = torch.zeros(feat_dim, dtype=torch.bool,
                                          device=features.device)
                padded_mask[:mask_dim] = engram_mask.to(features.device)
                engram_mask = padded_mask
            elif mask_dim > feat_dim:
                engram_mask = engram_mask[:feat_dim]

            # Inhibit non-engram neurons
            gated = features.clone()
            gated[:, ~engram_mask] *= inhibition_factor
            # Boost engram neurons
            if bias_shifts is not None:
                engram_indices = engram_mask.nonzero(as_tuple=True)[0]
                valid = min(len(bias_shifts), len(engram_indices))
                if valid > 0:
                    gated[:, engram_indices[:valid]] += (
                        bias_shifts[:valid].to(features.device).unsqueeze(0)
                    )
            features = gated

        return self.heads[task_id](features)

    # ------------------------------------------------------------------ #
    # Neurogenesis delegation
    # ------------------------------------------------------------------ #

    def expand_backbone(self, num_new_neurons: int) -> None:
        """Grow the backbone's last hidden layer and update all heads."""
        old_dim = self.backbone.output_dim
        self.backbone.expand_last_layer(num_new_neurons)
        new_dim = self.backbone.output_dim

        # Expand every existing head to accept the larger feature vector
        new_heads = nn.ModuleList()
        for head in self.heads:
            new_head = nn.Linear(new_dim, self.classes_per_task)
            with torch.no_grad():
                new_head.weight[:, :old_dim] = head.weight
                nn.init.kaiming_normal_(new_head.weight[:, old_dim:])
                new_head.bias.copy_(head.bias)
            new_heads.append(new_head)
        self.heads = new_heads

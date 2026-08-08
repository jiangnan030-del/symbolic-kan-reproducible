# SPDX-License-Identifier: MIT
# Derived research software; see NOTICE.md for upstream attribution.

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SelectionTerms:
    sharpness: torch.Tensor
    entropy: torch.Tensor
    nms: torch.Tensor
    off_mass: torch.Tensor

    def weighted(
        self,
        *,
        sharpness_weight: float,
        entropy_weight: float,
        nms_weight: float,
        off_mass_weight: float,
    ) -> torch.Tensor:
        return (
            sharpness_weight * self.sharpness
            + entropy_weight * self.entropy
            + nms_weight * self.nms
            + off_mass_weight * self.off_mass
        )


def selection_terms(primitive_probabilities: list[torch.Tensor]) -> SelectionTerms:
    """Compute sharpness, entropy, true pairwise NMS, and upstream-style off-mass.

    Every list item has shape ``[units, edges, primitives]`` for one block. NMS is the
    mean dot-product overlap between different edge distributions in the same unit,
    matching the pairwise definition in the manuscript rather than reusing off-mass.
    """

    if not primitive_probabilities:
        zero = torch.tensor(0.0)
        return SelectionTerms(zero, zero, zero, zero)

    sharpness_values: list[torch.Tensor] = []
    entropy_values: list[torch.Tensor] = []
    nms_values: list[torch.Tensor] = []
    off_mass_values: list[torch.Tensor] = []

    for probabilities in primitive_probabilities:
        eps = torch.finfo(probabilities.dtype).eps
        maxima = probabilities.max(dim=-1).values
        sharpness_values.append(1.0 - maxima.mean())
        entropy_values.append(
            -(probabilities * (probabilities + eps).log()).sum(dim=-1).mean()
        )
        off_mass_values.append((1.0 - maxima).mean())

        edge_count = probabilities.shape[1]
        if edge_count > 1:
            overlap = probabilities @ probabilities.transpose(-1, -2)
            mask = torch.triu(
                torch.ones(edge_count, edge_count, dtype=torch.bool, device=overlap.device),
                diagonal=1,
            )
            nms_values.append(overlap[:, mask].mean())
        else:
            nms_values.append(probabilities.new_zeros(()))

    return SelectionTerms(
        sharpness=torch.stack(sharpness_values).mean(),
        entropy=torch.stack(entropy_values).mean(),
        nms=torch.stack(nms_values).mean(),
        off_mass=torch.stack(off_mass_values).mean(),
    )


def unit_gate_penalty(unit_probabilities: list[torch.Tensor]) -> torch.Tensor:
    if not unit_probabilities:
        return torch.tensor(0.0)
    return torch.cat([value.reshape(-1) for value in unit_probabilities]).mean()


def primitive_bias_penalty(model: torch.nn.Module) -> torch.Tensor:
    values = [
        parameter.square().sum()
        for name, parameter in model.named_parameters()
        if name.endswith("bias_raw")
    ]
    if not values:
        reference = next(model.parameters())
        return reference.new_zeros(())
    return torch.stack(values).sum()

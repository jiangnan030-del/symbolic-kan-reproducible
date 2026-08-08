# SPDX-License-Identifier: MIT
# Architecture derived from sfaroughi3/Pub_Symbolic_KANs at 9481a82.
# See NOTICE.md for full authorship, license, and non-endorsement information.

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .gates import one_hot_argmax, primitive_probabilities, straight_through_argmax
from .primitives import SymbolicPrimitive
from .reproducibility import resolve_dtype


@dataclass(slots=True)
class GateState:
    """Differentiable structural diagnostics returned by a forward pass."""

    primitive_probabilities: list[torch.Tensor]
    edge_probabilities: list[torch.Tensor]
    unit_probabilities: list[torch.Tensor]


class GatedPrimitiveEdge(nn.Module):
    """One scalar projection followed by a gated library of primitives."""

    def __init__(
        self,
        primitive_names: tuple[str, ...],
        *,
        tau: float,
        hard_sample: bool,
        use_beta: bool,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.primitive_names = primitive_names
        self.primitives = nn.ModuleList(
            [
                SymbolicPrimitive(name, use_beta=use_beta, use_bias=use_bias)
                for name in primitive_names
            ]
        )
        self.logits = nn.Parameter(0.005 * torch.randn(len(primitive_names)))
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.register_buffer("is_hardened", torch.tensor(False))
        self.hard_sample = hard_sample

    def set_temperature(self, tau: float) -> None:
        if tau <= 0:
            raise ValueError("temperature must be positive")
        self.tau.fill_(float(tau))

    def probabilities(self) -> torch.Tensor:
        return primitive_probabilities(
            self.logits,
            float(self.tau.item()),
            training=self.training,
            hard_sample=self.hard_sample,
            hardened=bool(self.is_hardened.item()),
        )

    def deterministic_probabilities(self, *, hard: bool = False) -> torch.Tensor:
        if hard or bool(self.is_hardened.item()):
            return one_hot_argmax(self.logits)
        return torch.softmax(self.logits / self.tau, dim=-1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.probabilities()
        values = torch.stack([primitive(value) for primitive in self.primitives], dim=-1)
        return (values * probabilities).sum(dim=-1), probabilities

    @torch.no_grad()
    def harden(self, snap_strength: float = 8.0, freeze: bool = True) -> None:
        winner = int(self.logits.argmax().item())
        self.logits.fill_(-snap_strength)
        self.logits[winner] = snap_strength
        self.is_hardened.fill_(True)
        if freeze:
            self.logits.requires_grad_(False)


class SymbolicKANBlock(nn.Module):
    """A layer of units, each selecting one edge and one analytic primitive."""

    def __init__(self, input_dim: int, config: ModelConfig, tau: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_units = config.hidden_units
        self.edges_per_unit = config.edges_per_unit
        self.residual = config.residual
        self.edge_selection = config.edge_selection
        self.edge_temperature = config.edge_temperature
        self.use_unit_gates = config.unit_gates

        self.projection_weight = nn.Parameter(
            torch.randn(config.hidden_units, config.edges_per_unit, input_dim) * 0.4
        )
        if config.projection_bias:
            self.projection_bias = nn.Parameter(
                torch.zeros(config.hidden_units, config.edges_per_unit)
            )
        else:
            self.register_buffer(
                "projection_bias", torch.zeros(config.hidden_units, config.edges_per_unit)
            )

        self.edges = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        GatedPrimitiveEdge(
                            config.primitives,
                            tau=tau,
                            hard_sample=config.gumbel_hard,
                            use_beta=config.primitive_beta,
                            use_bias=config.primitive_bias,
                        )
                        for _ in range(config.edges_per_unit)
                    ]
                )
                for _ in range(config.hidden_units)
            ]
        )

        if self.use_unit_gates:
            self.unit_logits = nn.Parameter(torch.zeros(config.hidden_units))
        else:
            self.register_parameter("unit_logits", None)
        self.register_buffer(
            "edge_mask", torch.zeros(config.hidden_units, config.edges_per_unit)
        )
        self.register_buffer("edge_mask_active", torch.tensor(False))
        self.register_buffer("unit_gates_hardened", torch.tensor(False))

    def set_temperature(self, tau: float) -> None:
        for unit_edges in self.edges:
            for edge in unit_edges:
                edge.set_temperature(tau)

    def _edge_mask(self, confidence: torch.Tensor, unit_index: int) -> torch.Tensor:
        if bool(self.edge_mask_active.item()):
            return self.edge_mask[unit_index]
        soft = torch.softmax(confidence / self.edge_temperature, dim=-1)
        if self.training:
            if self.edge_selection == "soft":
                return soft
            return straight_through_argmax(soft)
        if self.edge_selection == "soft":
            return soft
        return one_hot_argmax(soft)

    def _unit_probabilities(self) -> torch.Tensor:
        if self.unit_logits is None:
            return self.projection_weight.new_ones(self.hidden_units)
        probabilities = torch.sigmoid(self.unit_logits)
        if bool(self.unit_gates_hardened.item()):
            return (probabilities > 0.5).to(probabilities.dtype)
        return probabilities

    def forward(
        self, value: torch.Tensor, residual_input: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        unit_outputs: list[torch.Tensor] = []
        primitive_by_unit: list[torch.Tensor] = []
        edge_by_unit: list[torch.Tensor] = []
        unit_probabilities = self._unit_probabilities()

        for unit_index, unit_edges in enumerate(self.edges):
            edge_outputs: list[torch.Tensor] = []
            edge_primitive_probabilities: list[torch.Tensor] = []
            edge_confidences: list[torch.Tensor] = []

            for edge_index, edge in enumerate(unit_edges):
                projection = (
                    value @ self.projection_weight[unit_index, edge_index].unsqueeze(-1)
                    + self.projection_bias[unit_index, edge_index]
                ).squeeze(-1)
                output, probabilities = edge(projection)
                edge_outputs.append(output)
                edge_primitive_probabilities.append(probabilities)
                edge_confidences.append(probabilities.max())

            edge_values = torch.stack(edge_outputs, dim=-1)
            confidence = torch.stack(edge_confidences)
            mask = self._edge_mask(confidence, unit_index)
            selected = (edge_values * mask).sum(dim=-1, keepdim=True)
            selected = selected * unit_probabilities[unit_index]

            unit_outputs.append(selected)
            primitive_by_unit.append(torch.stack(edge_primitive_probabilities))
            edge_by_unit.append(mask)

        hidden = torch.cat(unit_outputs, dim=-1)
        if self.residual and residual_input is not None and residual_input.shape == hidden.shape:
            hidden = residual_input + hidden

        return (
            hidden,
            torch.stack(primitive_by_unit),
            torch.stack(edge_by_unit),
            unit_probabilities,
        )

    @torch.no_grad()
    def harden(self, *, snap_strength: float = 8.0, freeze: bool = True) -> None:
        mask = torch.zeros_like(self.edge_mask)
        for unit_index, unit_edges in enumerate(self.edges):
            confidences = torch.stack(
                [edge.deterministic_probabilities().max() for edge in unit_edges]
            )
            winner = int(confidences.argmax().item())
            mask[unit_index, winner] = 1.0
            for edge in unit_edges:
                edge.harden(snap_strength=snap_strength, freeze=freeze)
        self.edge_mask.copy_(mask)
        self.edge_mask_active.fill_(True)

        if self.unit_logits is not None:
            alive = (torch.sigmoid(self.unit_logits) > 0.5).to(self.unit_logits.dtype)
            eps = torch.finfo(self.unit_logits.dtype).eps
            snapped = torch.log(alive.clamp_min(eps)) - torch.log((1.0 - alive).clamp_min(eps))
            self.unit_logits.copy_(snapped)
            self.unit_gates_hardened.fill_(True)
            if freeze:
                self.unit_logits.requires_grad_(False)


class FixedSumReadout(nn.Module):
    """Paper-aligned readout: an untrained sum of final activations."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.sum(dim=-1, keepdim=True)


class SymbolicKAN(nn.Module):
    """Symbolic Kolmogorov–Arnold Network with explicit discrete structure."""

    def __init__(self, config: ModelConfig, *, initial_temperature: float = 4.0) -> None:
        super().__init__()
        self.config = config
        blocks: list[SymbolicKANBlock] = []
        for block_index in range(config.num_blocks):
            input_dim = config.input_dim if block_index == 0 else config.hidden_units
            blocks.append(SymbolicKANBlock(input_dim, config, initial_temperature))
        self.blocks = nn.ModuleList(blocks)

        if config.readout == "fixed_sum":
            self.readout: nn.Module = FixedSumReadout()
        else:
            self.readout = nn.Linear(
                config.hidden_units, 1, bias=bool(config.readout_bias)
            )

        self.inverse_parameters_raw = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(float(config.inverse_parameter_init)))
                for _ in range(config.inverse_parameter_count)
            ]
        )
        self.to(dtype=resolve_dtype(config.dtype))

    def set_temperature(self, tau: float) -> None:
        for block in self.blocks:
            block.set_temperature(tau)

    def inverse_parameters(self) -> tuple[torch.Tensor, ...]:
        if self.config.inverse_parameter_constraint == "positive":
            return tuple(F.softplus(value) + 1e-8 for value in self.inverse_parameters_raw)
        return tuple(self.inverse_parameters_raw)

    def forward(
        self, value: torch.Tensor, *, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, GateState]:
        hidden: torch.Tensor | None = None
        primitive_probabilities_by_block: list[torch.Tensor] = []
        edge_probabilities_by_block: list[torch.Tensor] = []
        unit_probabilities_by_block: list[torch.Tensor] = []

        for block in self.blocks:
            block_input = value if hidden is None else hidden
            hidden, primitive_probabilities, edge_probabilities, unit_probabilities = block(
                block_input, hidden
            )
            primitive_probabilities_by_block.append(primitive_probabilities)
            edge_probabilities_by_block.append(edge_probabilities)
            if block.use_unit_gates:
                unit_probabilities_by_block.append(unit_probabilities)

        if hidden is None:
            raise RuntimeError("SymbolicKAN requires at least one block")
        prediction = self.readout(hidden)
        if not return_aux:
            return prediction
        return prediction, GateState(
            primitive_probabilities=primitive_probabilities_by_block,
            edge_probabilities=edge_probabilities_by_block,
            unit_probabilities=unit_probabilities_by_block,
        )

    @torch.no_grad()
    def harden(self, *, snap_strength: float = 8.0, freeze: bool = True) -> "SymbolicKAN":
        """Freeze primitive, edge, and optional unit decisions deterministically."""

        for block in self.blocks:
            block.harden(snap_strength=snap_strength, freeze=freeze)
        return self

    @property
    def is_hardened(self) -> bool:
        return all(bool(block.edge_mask_active.item()) for block in self.blocks)


def cosine_temperature(epoch: int, total_epochs: int, start: float, end: float) -> float:
    if total_epochs <= 1:
        return end
    progress = (epoch - 1) / (total_epochs - 1)
    return end + 0.5 * (start - end) * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())

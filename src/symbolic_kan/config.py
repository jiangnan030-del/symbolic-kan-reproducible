# SPDX-License-Identifier: MIT
# Derived research software; see NOTICE.md for upstream attribution.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

ReadoutMode = Literal["fixed_sum", "trainable_linear"]
EdgeSelectionMode = Literal["soft", "straight_through"]
InverseConstraint = Literal["unconstrained", "positive"]


@dataclass(slots=True)
class ModelConfig:
    """Serializable Symbolic-KAN architecture configuration."""

    input_dim: int = 1
    hidden_units: int = 6
    edges_per_unit: int = 3
    num_blocks: int = 4
    primitives: tuple[str, ...] = ("sin", "cos", "exp", "x", "x2")
    residual: bool = True
    projection_bias: bool = True
    primitive_beta: bool = False
    primitive_bias: bool = True
    unit_gates: bool = False
    readout: ReadoutMode = "fixed_sum"
    readout_bias: bool = False
    edge_selection: EdgeSelectionMode = "straight_through"
    edge_temperature: float = 1.0
    gumbel_hard: bool = False
    inverse_parameter_count: int = 0
    inverse_parameter_init: float = 0.1
    inverse_parameter_constraint: InverseConstraint = "unconstrained"
    dtype: Literal["float32", "float64"] = "float64"

    def __post_init__(self) -> None:
        self.primitives = tuple(self.primitives)
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.hidden_units < 1 or self.edges_per_unit < 1 or self.num_blocks < 1:
            raise ValueError("hidden_units, edges_per_unit, and num_blocks must be positive")
        if not self.primitives:
            raise ValueError("at least one primitive is required")
        if self.readout not in {"fixed_sum", "trainable_linear"}:
            raise ValueError(f"unsupported readout: {self.readout}")
        if self.edge_selection not in {"soft", "straight_through"}:
            raise ValueError(f"unsupported edge selection: {self.edge_selection}")
        if self.edge_temperature <= 0:
            raise ValueError("edge_temperature must be positive")
        if self.inverse_parameter_count < 0:
            raise ValueError("inverse_parameter_count cannot be negative")


@dataclass(slots=True)
class TrainingConfig:
    """Two-stage optimization and structure-selection settings."""

    seed: int = 42
    adam_epochs: int = 1000
    learning_rate: float = 5e-3
    weight_decay: float = 0.0
    gate_learning_rate_scale: float = 0.2
    gate_weight_decay: float = 1e-2
    tau_start: float = 4.0
    tau_end: float = 0.2
    selection_start_fraction: float = 0.5
    selection_weight_start: float = 0.0
    selection_weight_end: float = 0.5
    sharpness_weight: float = 1.0
    entropy_weight: float = 0.9
    nms_weight: float = 0.9
    off_mass_weight: float = 0.0
    unit_gate_weight: float = 0.0
    primitive_bias_weight: float = 1e-4
    gradient_clip_norm: float | None = 5.0
    validate_every: int = 1
    lbfgs_steps: int = 0
    lbfgs_learning_rate: float = 0.3
    lbfgs_max_iter: int = 15
    lbfgs_history_size: int = 50
    lbfgs_clip_norm: float | None = None
    deterministic_algorithms: bool = False

    def __post_init__(self) -> None:
        if self.adam_epochs < 0 or self.lbfgs_steps < 0:
            raise ValueError("optimizer step counts cannot be negative")
        if self.learning_rate <= 0 or self.gate_learning_rate_scale <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= self.selection_start_fraction <= 1:
            raise ValueError("selection_start_fraction must be in [0, 1]")
        if self.tau_start <= 0 or self.tau_end <= 0:
            raise ValueError("temperatures must be positive")
        if self.validate_every < 1:
            raise ValueError("validate_every must be positive")


@dataclass(slots=True)
class ExperimentConfig:
    """Complete experiment configuration with provenance metadata."""

    name: str
    profile: Literal["legacy", "paper", "corrected", "smoke"]
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    problem: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        data = dict(value)
        data["model"] = ModelConfig(**dict(data.get("model", {})))
        data["training"] = TrainingConfig(**dict(data.get("training", {})))
        data["problem"] = dict(data.get("problem", {}))
        data["provenance"] = dict(data.get("provenance", {}))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError(f"configuration must be a mapping: {config_path}")
    return ExperimentConfig.from_mapping(raw)

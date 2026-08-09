"""Unofficial, attributed Symbolic-KAN research package.

The Symbolic-KAN method and upstream implementation are credited in NOTICE.md.
This package is not an official release by the original authors.
"""

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointLoadResult,
    checkpoint_summary,
    load_checkpoint,
    save_checkpoint,
)
from .config import ExperimentConfig, ModelConfig, TrainingConfig, load_experiment_config
from .export import export_expression, export_structure
from .inspection import (
    PrimitiveCandidate,
    build_symbolic_report,
    prune_edges,
    prune_units,
    rank_primitive_candidates,
    structure_diagnostics,
    write_structure_svg,
    write_symbolic_report,
)
from .model import GateState, SymbolicKAN
from .training import FitResult, fit_supervised

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointLoadResult",
    "ExperimentConfig",
    "FitResult",
    "GateState",
    "ModelConfig",
    "PrimitiveCandidate",
    "SymbolicKAN",
    "TrainingConfig",
    "build_symbolic_report",
    "checkpoint_summary",
    "export_expression",
    "export_structure",
    "fit_supervised",
    "load_checkpoint",
    "load_experiment_config",
    "prune_edges",
    "prune_units",
    "rank_primitive_candidates",
    "save_checkpoint",
    "structure_diagnostics",
    "write_structure_svg",
    "write_symbolic_report",
]

__version__ = "0.1.0a2"

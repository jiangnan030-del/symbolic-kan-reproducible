"""Unofficial, attributed Symbolic-KAN research package.

The Symbolic-KAN method and upstream implementation are credited in NOTICE.md.
This package is not an official release by the original authors.
"""

from .config import ExperimentConfig, ModelConfig, TrainingConfig, load_experiment_config
from .export import export_expression, export_structure
from .model import GateState, SymbolicKAN
from .training import FitResult, fit_supervised

__all__ = [
    "ExperimentConfig",
    "FitResult",
    "GateState",
    "ModelConfig",
    "SymbolicKAN",
    "TrainingConfig",
    "export_expression",
    "export_structure",
    "fit_supervised",
    "load_experiment_config",
]

__version__ = "0.1.0a1"

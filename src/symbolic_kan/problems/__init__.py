"""Physics problem helpers for attributed Symbolic-KAN experiments."""

from .laplace import LaplaceProblem
from .reaction_diffusion import ReactionDiffusionProblem
from .volterra import VariableLimitGaussLegendre, VolterraProblem

__all__ = [
    "LaplaceProblem",
    "ReactionDiffusionProblem",
    "VariableLimitGaussLegendre",
    "VolterraProblem",
]

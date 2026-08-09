"""Physics problem helpers for attributed Symbolic-KAN experiments."""

from .conservation_law import ConservationLawProblem
from .laplace import LaplaceProblem
from .reaction_diffusion import ReactionDiffusionProblem
from .volterra import VariableLimitGaussLegendre, VolterraProblem

__all__ = [
    "ConservationLawProblem",
    "LaplaceProblem",
    "ReactionDiffusionProblem",
    "VariableLimitGaussLegendre",
    "VolterraProblem",
]

# SPDX-License-Identifier: MIT
# Adapted from the conservation-law tutorial in KindXiaoming/pykan (Physics_2A).
# This is a derivative-package adapter; it is not upstream source code.

"""Scalar conservation-law benchmark.

Solves ``u_t + f(u)_x = 0`` with flux ``f(u) = 0.5 * u^2`` (inviscid Burgers')
on a periodic domain.  The training signal combines a PDE residual evaluated at
interior collocation points with an initial-condition data-fit term.

The case mirrors the pedagogical setup in ``tutorials/Physics/Physics_2A_conservation_law.ipynb``
from the original pykan repository, but uses the Symbolic-KAN gated-primitive
architecture instead of B-spline KAN layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..model import GateState, SymbolicKAN


def _gradient(output: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Differentiate *output* w.r.t. *coords*, keeping the graph alive."""

    grad: torch.Tensor | None = None
    if output.requires_grad:
        grad = torch.autograd.grad(
            output,
            coords,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            allow_unused=True,
        )[0]
    if grad is None:
        grad = torch.zeros_like(coords)
    return grad + 0.0 * coords


@dataclass(slots=True)
class ConservationLawProblem:
    """1-D scalar conservation law (inviscid Burgers').

    PDE:  ``u_t + u * u_x = 0``
    IC:   ``u(x, 0) = sin(2*pi*x)``
    Domain: ``x in [0, 1)`` periodic, ``t in [0, t_end]``
    """

    domain_low: float = 0.0
    domain_high: float = 1.0
    t_end: float = 0.2

    def __post_init__(self) -> None:
        if self.domain_high <= self.domain_low:
            raise ValueError("domain_high must be greater than domain_low")
        if self.t_end <= 0:
            raise ValueError("t_end must be positive")

    @staticmethod
    def _validate_coordinates(coordinates: torch.Tensor) -> None:
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape [N, 2] (x, t)")

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """u(x, 0) = sin(2*pi*x)."""
        pi = x.new_tensor(math.pi)
        return torch.sin(2.0 * pi * x)

    def sample(
        self,
        *,
        collocation_count: int,
        ic_count: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample interior collocation points and initial-condition points."""

        if collocation_count < 1:
            raise ValueError("collocation_count must be positive")
        if ic_count < 1:
            raise ValueError("ic_count must be positive")

        span = self.domain_high - self.domain_low
        eps = torch.finfo(dtype).eps

        collocation = (
            torch.rand(collocation_count, 2, device=device, dtype=dtype, generator=generator)
            * torch.tensor([span, self.t_end], device=device, dtype=dtype)
            + torch.tensor([self.domain_low + eps, eps], device=device, dtype=dtype)
        )

        ic_x = (
            torch.rand(ic_count, 1, device=device, dtype=dtype, generator=generator) * span
            + self.domain_low
        )

        return {
            "collocation_xt": collocation,
            "ic_x": ic_x,
            "ic_u": self.initial_condition(ic_x),
        }

    def loss(
        self,
        model: SymbolicKAN,
        batch: dict[str, torch.Tensor],
        *,
        pde_weight: float = 1.0,
        ic_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], GateState]:
        """Compute PDE residual loss and initial-condition data loss.

        The model receives ``(x, t)`` and predicts ``u(x, t)``.
        Residual: ``u_t + u * u_x``
        """

        coordinates = batch["collocation_xt"].detach().clone().requires_grad_(True)
        prediction, gates = model(coordinates, return_aux=True)

        # Spatial and temporal gradients
        grad = _gradient(prediction, coordinates)
        u_x = grad[:, 0:1]
        u_t = grad[:, 1:2]

        # Conservation law residual: u_t + u * u_x = 0
        residual = u_t + prediction * u_x
        pde_loss = residual.square().mean()

        # Initial condition loss: u(x, 0) = sin(2*pi*x)
        ic_x = batch["ic_x"].detach().clone().requires_grad_(True)
        ic_t = torch.zeros_like(ic_x)
        ic_coords = torch.cat([ic_x, ic_t], dim=1)
        ic_pred = model(ic_coords)
        ic_loss = (ic_pred - batch["ic_u"]).square().mean()

        total = pde_weight * pde_loss + ic_weight * ic_loss
        return total, {"pde": pde_loss, "ic": ic_loss}, gates

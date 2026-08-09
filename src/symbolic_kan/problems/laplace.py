# SPDX-License-Identifier: MIT
# Benchmark equation and settings are transcribed from the Symbolic-KAN manuscript.
# No matching Laplace experiment was present in the audited upstream code at 9481a82.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..model import GateState, SymbolicKAN


def _coordinate_gradient(output: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Differentiate while keeping constant/linear branches second-derivative safe."""

    gradient: torch.Tensor | None = None
    if output.requires_grad:
        gradient = torch.autograd.grad(
            output,
            coordinates,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            allow_unused=True,
        )[0]
    if gradient is None:
        gradient = torch.zeros_like(coordinates)
    # Preserve a zero-valued graph connection so another derivative is always valid.
    return gradient + 0.0 * coordinates


@dataclass(slots=True)
class LaplaceProblem:
    """Two-dimensional harmonic benchmark described in the Symbolic-KAN paper.

    The target is ``u(x, y) = sin(pi*x) * sinh(pi*y)`` on a square domain. This
    adapter is part of the derivative package; it is not recovered upstream source code.
    """

    domain_low: float = 0.0
    domain_high: float = 1.0

    def __post_init__(self) -> None:
        if self.domain_high <= self.domain_low:
            raise ValueError("domain_high must be greater than domain_low")

    @staticmethod
    def _validate_coordinates(coordinates: torch.Tensor) -> None:
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape [N, 2]")

    def exact_solution(self, coordinates: torch.Tensor) -> torch.Tensor:
        self._validate_coordinates(coordinates)
        x = coordinates[:, 0:1]
        y = coordinates[:, 1:2]
        pi = coordinates.new_tensor(math.pi)
        return torch.sin(pi * x) * torch.sinh(pi * y)

    def sample(
        self,
        *,
        collocation_count: int,
        boundary_count: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample interior points and exactly ``boundary_count`` boundary points."""

        if collocation_count < 1:
            raise ValueError("collocation_count must be positive")
        if boundary_count < 0:
            raise ValueError("boundary_count cannot be negative")

        span = self.domain_high - self.domain_low
        eps = torch.finfo(dtype).eps
        collocation = (
            torch.rand(
                collocation_count,
                2,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * (span - 2.0 * eps)
            + self.domain_low
            + eps
        )

        base, remainder = divmod(boundary_count, 4)
        side_counts = [base + int(index < remainder) for index in range(4)]

        def uniform(count: int) -> torch.Tensor:
            return (
                torch.rand(
                    count,
                    1,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                * span
                + self.domain_low
            )

        def fixed(count: int, value: float) -> torch.Tensor:
            return torch.full((count, 1), value, device=device, dtype=dtype)

        left = torch.cat(
            [fixed(side_counts[0], self.domain_low), uniform(side_counts[0])], dim=1
        )
        right = torch.cat(
            [fixed(side_counts[1], self.domain_high), uniform(side_counts[1])], dim=1
        )
        bottom = torch.cat(
            [uniform(side_counts[2]), fixed(side_counts[2], self.domain_low)], dim=1
        )
        top = torch.cat(
            [uniform(side_counts[3]), fixed(side_counts[3], self.domain_high)], dim=1
        )
        boundary = torch.cat([left, right, bottom, top], dim=0)

        return {
            "collocation_xy": collocation,
            "boundary_xy": boundary,
            "boundary_u": self.exact_solution(boundary),
        }

    def loss(
        self,
        model: SymbolicKAN,
        batch: dict[str, torch.Tensor],
        *,
        pde_weight: float = 1.0,
        boundary_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], GateState]:
        """Return the weighted Laplace residual and Dirichlet boundary loss."""

        coordinates = (
            batch["collocation_xy"].detach().clone().requires_grad_(True)
        )
        prediction, gates = model(coordinates, return_aux=True)
        first = _coordinate_gradient(prediction, coordinates)
        second_x = _coordinate_gradient(first[:, 0:1], coordinates)[:, 0:1]
        second_y = _coordinate_gradient(first[:, 1:2], coordinates)[:, 1:2]
        residual = second_x + second_y
        pde_loss = residual.square().mean()

        boundary_coordinates = batch["boundary_xy"]
        if boundary_coordinates.shape[0] == 0:
            boundary_loss = prediction.new_zeros(())
        else:
            boundary_prediction = model(boundary_coordinates)
            boundary_loss = (
                boundary_prediction - batch["boundary_u"]
            ).square().mean()

        total = pde_weight * pde_loss + boundary_weight * boundary_loss
        return total, {"pde": pde_loss, "boundary": boundary_loss}, gates

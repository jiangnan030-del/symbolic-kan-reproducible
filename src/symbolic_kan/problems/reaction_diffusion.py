# SPDX-License-Identifier: MIT
# Problem definition derived from sfaroughi3/Pub_Symbolic_KANs at 9481a82.

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model import GateState, SymbolicKAN


@dataclass(slots=True)
class ReactionDiffusionProblem:
    """Inverse 1D reaction–diffusion benchmark from the upstream repository."""

    diffusion: float = 0.01
    kappa_true: float = 0.7
    domain_low: float = -2.0
    domain_high: float = 2.0

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(6.0 * x).pow(3)

    def forcing(self, x: torch.Tensor) -> torch.Tensor:
        sine = torch.sin(6.0 * x)
        diffusion_term = 1.08 * sine * (2.0 - 3.0 * sine.square())
        return diffusion_term + self.kappa_true * torch.tanh(sine.pow(3))

    def sample(
        self,
        *,
        collocation_count: int,
        measurement_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        eps = torch.finfo(dtype).eps
        span = self.domain_high - self.domain_low
        collocation = (
            torch.rand(collocation_count, 1, device=device, dtype=dtype)
            * (span - 2.0 * eps)
            + self.domain_low
            + eps
        )
        measurement_x = (
            torch.rand(measurement_count, 1, device=device, dtype=dtype)
            * (span - 2.0 * eps)
            + self.domain_low
            + eps
        )
        boundary_x = torch.tensor(
            [[self.domain_low], [self.domain_high]], device=device, dtype=dtype
        )
        return {
            "collocation_x": collocation,
            "measurement_x": measurement_x,
            "measurement_y": self.exact_solution(measurement_x),
            "boundary_x": boundary_x,
            "boundary_y": self.exact_solution(boundary_x),
        }

    def loss(
        self,
        model: SymbolicKAN,
        batch: dict[str, torch.Tensor],
        *,
        pde_weight: float = 0.1,
        boundary_weight: float = 1.0,
        data_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], GateState]:
        x = batch["collocation_x"].detach().clone().requires_grad_(True)
        u, gates = model(x, return_aux=True)
        first = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        second = torch.autograd.grad(
            first, x, grad_outputs=torch.ones_like(first), create_graph=True
        )[0]
        inverse = model.inverse_parameters()
        kappa = inverse[0] if inverse else x.new_tensor(self.kappa_true)
        residual = self.diffusion * second + kappa * torch.tanh(u) - self.forcing(x)
        pde_loss = residual.square().mean()

        boundary_prediction = model(batch["boundary_x"])
        boundary_loss = (boundary_prediction - batch["boundary_y"]).square().mean()
        if batch["measurement_x"].numel():
            data_prediction = model(batch["measurement_x"])
            data_loss = (data_prediction - batch["measurement_y"]).square().mean()
        else:
            data_loss = x.new_zeros(())

        total = (
            pde_weight * pde_loss
            + boundary_weight * boundary_loss
            + data_weight * data_loss
        )
        return total, {"pde": pde_loss, "boundary": boundary_loss, "data": data_loss}, gates

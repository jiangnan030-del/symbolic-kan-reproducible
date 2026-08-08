# SPDX-License-Identifier: MIT
# Problem definition derived from sfaroughi3/Pub_Symbolic_KANs at 9481a82.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..model import GateState, SymbolicKAN


class VariableLimitGaussLegendre(nn.Module):
    """Gauss–Legendre integration on every interval ``[0, x_i]``.

    The upstream implementation masks one global quadrature rule. Here, each row's nodes
    and weights are correctly affinely mapped to its own upper limit. Storage is O(NQ).
    """

    def __init__(self, order: int = 64) -> None:
        super().__init__()
        if order < 2:
            raise ValueError("quadrature order must be at least two")
        nodes, weights = np.polynomial.legendre.leggauss(order)
        self.register_buffer("nodes", torch.from_numpy(nodes))
        self.register_buffer("weights", torch.from_numpy(weights))

    def forward(
        self,
        integrand: Callable[[torch.Tensor], torch.Tensor],
        upper_limits: torch.Tensor,
    ) -> torch.Tensor:
        if upper_limits.ndim != 2 or upper_limits.shape[1] != 1:
            raise ValueError("upper_limits must have shape [N, 1]")
        nodes = self.nodes.to(device=upper_limits.device, dtype=upper_limits.dtype)
        base_weights = self.weights.to(device=upper_limits.device, dtype=upper_limits.dtype)
        mapped_nodes = 0.5 * upper_limits * (nodes.reshape(1, -1) + 1.0)
        mapped_weights = 0.5 * upper_limits * base_weights.reshape(1, -1)
        values = integrand(mapped_nodes.reshape(-1, 1)).reshape(mapped_nodes.shape)
        return (mapped_weights * values).sum(dim=1, keepdim=True)


@dataclass(slots=True)
class VolterraProblem:
    """MINPO-style nonlinear Volterra benchmark used by the upstream code."""

    kappa: float = 1.0
    domain_end: float = 10.0
    quadrature_order: int = 64

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        root = x.new_tensor(self.kappa).sqrt()
        return torch.exp(-x) * torch.cosh(root * x)

    def exact_memory(self, x: torch.Tensor) -> torch.Tensor:
        root = x.new_tensor(self.kappa).sqrt()
        return torch.exp(-x) * torch.sinh(root * x) / root

    def loss(
        self,
        model: SymbolicKAN,
        collocation_x: torch.Tensor,
        *,
        residual_weight: float = 1.0,
        initial_weight: float = 1.0,
        memory_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], GateState]:
        x = collocation_x.detach().clone().requires_grad_(True)
        memory, gates = model(x, return_aux=True)
        memory_x = torch.autograd.grad(
            memory, x, grad_outputs=torch.ones_like(memory), create_graph=True
        )[0]
        solution = memory + memory_x
        solution_x = torch.autograd.grad(
            solution, x, grad_outputs=torch.ones_like(solution), create_graph=True
        )[0]
        residual = solution_x + solution - self.kappa * memory
        residual_loss = residual.square().mean()

        x0 = x.new_zeros((1, 1), requires_grad=True)
        memory0 = model(x0)
        memory0_x = torch.autograd.grad(
            memory0, x0, grad_outputs=torch.ones_like(memory0), create_graph=True
        )[0]
        solution0 = memory0 + memory0_x
        initial_loss = memory0.square().mean() + (solution0 - 1.0).square().mean()

        quadrature = VariableLimitGaussLegendre(self.quadrature_order).to(
            device=x.device, dtype=x.dtype
        )

        def kernel_integrand(tau: torch.Tensor) -> torch.Tensor:
            tau = tau.requires_grad_(True)
            memory_tau = model(tau)
            derivative_tau = torch.autograd.grad(
                memory_tau,
                tau,
                grad_outputs=torch.ones_like(memory_tau),
                create_graph=True,
            )[0]
            solution_tau = memory_tau + derivative_tau
            # x is supplied row-wise outside; return u(tau), then apply kernel explicitly.
            return solution_tau

        # Evaluate mapped nodes directly so the exponential kernel can use each upper limit.
        nodes = quadrature.nodes.to(device=x.device, dtype=x.dtype)
        weights = quadrature.weights.to(device=x.device, dtype=x.dtype)
        tau = 0.5 * x * (nodes.reshape(1, -1) + 1.0)
        tau_flat = tau.reshape(-1, 1).requires_grad_(True)
        u_tau = kernel_integrand(tau_flat).reshape(tau.shape)
        mapped_weights = 0.5 * x * weights.reshape(1, -1)
        kernel = torch.exp(tau - x)
        reconstructed_memory = (mapped_weights * kernel * u_tau).sum(dim=1, keepdim=True)
        memory_loss = (reconstructed_memory - memory).square().mean()

        total = (
            residual_weight * residual_loss
            + initial_weight * initial_loss
            + memory_weight * memory_loss
        )
        return total, {"residual": residual_loss, "initial": initial_loss, "memory": memory_loss}, gates

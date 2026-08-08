import torch

from symbolic_kan.problems import VariableLimitGaussLegendre


def test_variable_limit_quadrature_matches_exponential_kernel_constant() -> None:
    rule = VariableLimitGaussLegendre(order=32).to(dtype=torch.float64)
    x = torch.linspace(0.0, 3.0, 20, dtype=torch.float64).reshape(-1, 1)

    def integrand(tau: torch.Tensor) -> torch.Tensor:
        # Integrate exp(tau - x) row-wise by folding exp(tau) here and exp(-x) outside.
        return torch.exp(tau)

    integral_exp_tau = rule(integrand, x)
    result = torch.exp(-x) * integral_exp_tau
    expected = 1.0 - torch.exp(-x)
    assert torch.allclose(result, expected, atol=1e-12, rtol=1e-12)

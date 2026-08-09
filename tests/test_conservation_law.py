import math
from pathlib import Path

import torch

from symbolic_kan import ModelConfig, SymbolicKAN, load_experiment_config
from symbolic_kan.problems import ConservationLawProblem


def test_initial_condition() -> None:
    problem = ConservationLawProblem()
    x = torch.tensor([[0.0], [0.25], [0.5], [0.75]], dtype=torch.float64)
    u0 = problem.initial_condition(x)
    expected = torch.sin(2.0 * math.pi * x)
    assert torch.allclose(u0, expected, atol=1e-12)


def test_sampling_shapes() -> None:
    problem = ConservationLawProblem()
    batch = problem.sample(
        collocation_count=20,
        ic_count=10,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert batch["collocation_xt"].shape == (20, 2)
    assert batch["ic_x"].shape == (10, 1)
    assert batch["ic_u"].shape == (10, 1)
    # Collocation t must be in (0, t_end)
    t = batch["collocation_xt"][:, 1]
    assert bool((t > 0).all()) and bool((t < problem.t_end).all())
    # Collocation x must be in domain
    x = batch["collocation_xt"][:, 0]
    assert bool((x >= problem.domain_low).all()) and bool((x < problem.domain_high).all())


def test_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(5)
    problem = ConservationLawProblem()
    model = SymbolicKAN(
        ModelConfig(
            input_dim=2,
            hidden_units=2,
            edges_per_unit=2,
            num_blocks=1,
            primitives=("x", "x2", "sin", "tanh"),
            readout="fixed_sum",
            dtype="float64",
        ),
        initial_temperature=2.0,
    )
    batch = problem.sample(
        collocation_count=8,
        ic_count=6,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    loss, components, _ = problem.loss(model, batch)
    assert torch.isfinite(loss)
    assert set(components) == {"pde", "ic"}
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_smoke_profile_loads() -> None:
    path = Path(__file__).parents[1] / "experiments/conservation_law/configs/smoke.yaml"
    config = load_experiment_config(path)
    assert config.profile == "smoke"
    assert config.model.input_dim == 2
    assert config.problem["type"] == "conservation_law"

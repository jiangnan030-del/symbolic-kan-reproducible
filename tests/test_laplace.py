from pathlib import Path

import torch

from symbolic_kan import ModelConfig, SymbolicKAN, load_experiment_config
from symbolic_kan.problems import LaplaceProblem


def _gradient(output: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        coordinates,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
    )[0]


def test_exact_solution_is_harmonic() -> None:
    problem = LaplaceProblem()
    coordinates = torch.tensor(
        [[0.17, 0.23], [0.43, 0.61], [0.82, 0.37]],
        dtype=torch.float64,
        requires_grad=True,
    )
    solution = problem.exact_solution(coordinates)
    first = _gradient(solution, coordinates)
    second_x = _gradient(first[:, 0:1], coordinates)[:, 0:1]
    second_y = _gradient(first[:, 1:2], coordinates)[:, 1:2]
    assert torch.allclose(second_x + second_y, torch.zeros_like(solution), atol=1e-11)


def test_sampling_respects_square_boundary_and_count() -> None:
    problem = LaplaceProblem()
    generator = torch.Generator().manual_seed(7)
    batch = problem.sample(
        collocation_count=11,
        boundary_count=13,
        device=torch.device("cpu"),
        dtype=torch.float64,
        generator=generator,
    )
    assert batch["collocation_xy"].shape == (11, 2)
    assert batch["boundary_xy"].shape == (13, 2)
    assert batch["boundary_u"].shape == (13, 1)
    boundary = batch["boundary_xy"]
    on_boundary = (
        torch.isclose(boundary[:, 0], boundary.new_tensor(0.0))
        | torch.isclose(boundary[:, 0], boundary.new_tensor(1.0))
        | torch.isclose(boundary[:, 1], boundary.new_tensor(0.0))
        | torch.isclose(boundary[:, 1], boundary.new_tensor(1.0))
    )
    assert bool(on_boundary.all())


def test_laplace_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(3)
    problem = LaplaceProblem()
    model = SymbolicKAN(
        ModelConfig(
            input_dim=2,
            hidden_units=2,
            edges_per_unit=2,
            num_blocks=1,
            primitives=("x", "x2", "sin", "sinh"),
            readout="fixed_sum",
            dtype="float64",
        ),
        initial_temperature=2.0,
    )
    batch = problem.sample(
        collocation_count=6,
        boundary_count=8,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    loss, components, _ = problem.loss(model, batch)
    assert torch.isfinite(loss)
    assert set(components) == {"pde", "boundary"}
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_laplace_smoke_profile_loads() -> None:
    path = Path(__file__).parents[1] / "experiments/laplace/configs/smoke.yaml"
    config = load_experiment_config(path)
    assert config.profile == "smoke"
    assert config.model.input_dim == 2
    assert config.problem["type"] == "laplace"
    assert "manuscript" in config.provenance["source_scope"]

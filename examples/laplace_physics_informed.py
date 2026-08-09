"""Train the paper-described 2-D Laplace case with an auditable short loop.

This is a derivative-package example, not released upstream experiment code and not a
claim that the paper's reported metric has been reproduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from symbolic_kan import (
    SymbolicKAN,
    load_experiment_config,
    save_checkpoint,
    write_symbolic_report,
)
from symbolic_kan.model import cosine_temperature
from symbolic_kan.problems import LaplaceProblem
from symbolic_kan.regularization import (
    primitive_bias_penalty,
    selection_terms,
    unit_gate_penalty,
)
from symbolic_kan.reproducibility import resolve_dtype, seed_everything
from symbolic_kan.training import build_adamw


def selection_scale(step: int, total_steps: int, start_fraction: float, end: float) -> float:
    start = int(start_fraction * max(total_steps, 1))
    if step < start:
        return 0.0
    progress = (step - start) / max(1, total_steps - start)
    return end * progress**2


def relative_grid_error(
    model: SymbolicKAN,
    problem: LaplaceProblem,
    *,
    device: torch.device,
    dtype: torch.dtype,
    grid_size: int = 33,
) -> float:
    axis = torch.linspace(
        problem.domain_low,
        problem.domain_high,
        grid_size,
        device=device,
        dtype=dtype,
    )
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
    with torch.no_grad():
        prediction = model(coordinates)
        target = problem.exact_solution(coordinates)
    return float(
        (
            torch.linalg.vector_norm(prediction - target)
            / (torch.linalg.vector_norm(target) + 1e-12)
        ).item()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/laplace/configs/smoke.yaml",
        help="Laplace YAML profile",
    )
    parser.add_argument("--output", default="outputs/laplace-smoke")
    parser.add_argument("--steps", type=int, help="override Adam steps from the profile")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    experiment = load_experiment_config(args.config)
    if experiment.problem.get("type") != "laplace":
        raise ValueError("the selected configuration is not a Laplace profile")
    if experiment.model.input_dim != 2:
        raise ValueError("the Laplace example requires model.input_dim == 2")

    training = experiment.training
    steps = training.adam_epochs if args.steps is None else args.steps
    if steps < 1:
        raise ValueError("steps must be positive")

    seed_everything(training.seed, training.deterministic_algorithms)
    device = torch.device(args.device)
    dtype = resolve_dtype(experiment.model.dtype)
    model = SymbolicKAN(
        experiment.model,
        initial_temperature=training.tau_start,
    ).to(device)
    domain = experiment.problem.get("domain", [0.0, 1.0])
    problem = LaplaceProblem(float(domain[0]), float(domain[1]))
    batch = problem.sample(
        collocation_count=int(experiment.problem.get("collocation_count", 32)),
        boundary_count=int(experiment.problem.get("boundary_count", 16)),
        device=device,
        dtype=dtype,
    )
    optimizer = build_adamw(model, training)
    history: list[dict[str, float | int | str]] = []

    for step in range(1, steps + 1):
        model.train()
        temperature = cosine_temperature(
            step,
            steps,
            training.tau_start,
            training.tau_end,
        )
        model.set_temperature(temperature)
        optimizer.zero_grad(set_to_none=True)
        physics, components, gates = problem.loss(
            model,
            batch,
            pde_weight=float(experiment.problem.get("pde_weight", 1.0)),
            boundary_weight=float(experiment.problem.get("boundary_weight", 1.0)),
        )
        terms = selection_terms(gates.primitive_probabilities)
        selection = terms.weighted(
            sharpness_weight=training.sharpness_weight,
            entropy_weight=training.entropy_weight,
            nms_weight=training.nms_weight,
            off_mass_weight=training.off_mass_weight,
        )
        scale = selection_scale(
            step,
            steps,
            training.selection_start_fraction,
            training.selection_weight_end,
        )
        unit_penalty = unit_gate_penalty(gates.unit_probabilities).to(physics.device)
        total = (
            physics
            + scale * selection
            + training.unit_gate_weight * unit_penalty
            + training.primitive_bias_weight * primitive_bias_penalty(model)
        )
        total.backward()
        if training.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
        optimizer.step()
        history.append(
            {
                "phase": "adam",
                "step": step,
                "loss": float(total.detach().item()),
                "pde": float(components["pde"].detach().item()),
                "boundary": float(components["boundary"].detach().item()),
                "selection": float(selection.detach().item()),
                "temperature": float(temperature),
            }
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": experiment.name,
        "profile": experiment.profile,
        "scope": "paper-described derivative Laplace adapter; not upstream code",
    }

    model.eval()
    soft_error = relative_grid_error(
        model,
        problem,
        device=device,
        dtype=dtype,
    )
    save_checkpoint(
        output / "checkpoint_soft.pt",
        model,
        phase="soft",
        training_config=training,
        optimizer=optimizer,
        history=history,
        metadata=metadata,
    )
    write_symbolic_report(
        model,
        output / "soft_report",
        variables=["x", "y"],
        history=history,
        metadata=metadata,
    )

    model.harden(freeze=True).eval()
    hardened_error = relative_grid_error(
        model,
        problem,
        device=device,
        dtype=dtype,
    )
    save_checkpoint(
        output / "checkpoint_hardened.pt",
        model,
        phase="hardened",
        training_config=training,
        history=history,
        metadata=metadata,
    )
    artifacts = write_symbolic_report(
        model,
        output / "hardened_report",
        variables=["x", "y"],
        history=history,
        metadata=metadata,
    )

    print(
        json.dumps(
            {
                "experiment": experiment.name,
                "profile": experiment.profile,
                "steps": steps,
                "soft_relative_grid_error": soft_error,
                "hardened_relative_grid_error": hardened_error,
                "last_training_entry": history[-1],
                "artifacts": {name: str(path) for name, path in artifacts.items()},
                "scientific_claim": "none; validate long runs and multiple seeds separately",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

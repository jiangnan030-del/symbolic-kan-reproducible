"""Train the conservation-law PDE case with an auditable short loop.

Solves ``u_t + u * u_x = 0`` (inviscid Burgers') with initial condition
``u(x, 0) = sin(2*pi*x)`` on a periodic domain.

Adapted from KindXiaoming/pykan Physics_2A_conservation_law tutorial.

This is a derivative-package example, not upstream experiment code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from symbolic_kan import (
    SymbolicKAN,
    load_experiment_config,
    plot_2d_prediction,
    plot_training_history,
    save_checkpoint,
    write_symbolic_report,
)
from symbolic_kan.model import cosine_temperature
from symbolic_kan.problems import ConservationLawProblem
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/conservation_law/configs/smoke.yaml",
        help="Conservation law YAML profile",
    )
    parser.add_argument("--output", default="outputs/conservation-law-smoke")
    parser.add_argument("--steps", type=int, help="override Adam steps from the profile")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    experiment = load_experiment_config(args.config)
    if experiment.problem.get("type") != "conservation_law":
        raise ValueError("the selected configuration is not a conservation_law profile")
    if experiment.model.input_dim != 2:
        raise ValueError("the conservation law example requires model.input_dim == 2")

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
    t_end = float(experiment.problem.get("t_end", 0.2))
    problem = ConservationLawProblem(float(domain[0]), float(domain[1]), t_end)
    batch = problem.sample(
        collocation_count=int(experiment.problem.get("collocation_count", 32)),
        ic_count=int(experiment.problem.get("ic_count", 16)),
        device=device,
        dtype=dtype,
    )
    optimizer = build_adamw(model, training)
    history: list[dict[str, float | int | str]] = []

    for step in range(1, steps + 1):
        model.train()
        temperature = cosine_temperature(step, steps, training.tau_start, training.tau_end)
        model.set_temperature(temperature)
        optimizer.zero_grad(set_to_none=True)
        physics, components, gates = problem.loss(
            model,
            batch,
            pde_weight=float(experiment.problem.get("pde_weight", 1.0)),
            ic_weight=float(experiment.problem.get("ic_weight", 1.0)),
        )
        terms = selection_terms(gates.primitive_probabilities)
        selection = terms.weighted(
            sharpness_weight=training.sharpness_weight,
            entropy_weight=training.entropy_weight,
            nms_weight=training.nms_weight,
            off_mass_weight=training.off_mass_weight,
        )
        scale = selection_scale(
            step, steps,
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
        history.append({
            "phase": "adam",
            "step": step,
            "loss": float(total.detach().item()),
            "pde": float(components["pde"].detach().item()),
            "ic": float(components["ic"].detach().item()),
            "selection": float(selection.detach().item()),
            "temperature": float(temperature),
        })

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": experiment.name,
        "profile": experiment.profile,
        "scope": "derivative conservation-law adapter; not upstream code",
    }

    model.eval()
    save_checkpoint(
        output / "checkpoint_soft.pt", model, phase="soft",
        training_config=training, optimizer=optimizer, history=history, metadata=metadata,
    )
    write_symbolic_report(
        model, output / "soft_report", variables=["x", "t"],
        history=history, metadata=metadata,
    )

    model.harden(freeze=True).eval()
    save_checkpoint(
        output / "checkpoint_hardened.pt", model, phase="hardened",
        training_config=training, history=history, metadata=metadata,
    )
    artifacts = write_symbolic_report(
        model, output / "hardened_report", variables=["x", "t"],
        history=history, metadata=metadata,
    )

    # -- Generate PNG plots --
    plot_training_history(
        history, output / "training_history.png",
        title=f"Conservation Law: u_t + u*u_x = 0 ({experiment.name})",
    )

    def target_fn(X, Y):
        # u(x, 0) = sin(2*pi*x) -- plot the IC slice at t=0
        return np.sin(2.0 * np.pi * X)

    # For the 2-input model, show prediction at t=0 slice
    model_for_plot = SymbolicKAN(experiment.model)
    ckpt_path = output / "checkpoint_soft.pt"
    try:
        soft_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        soft_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_for_plot.load_state_dict(soft_state["model_state"])
    model_for_plot.eval()
    plot_2d_prediction(
        model_for_plot, target_fn, output / "prediction_soft.png",
        x_range=(domain[0], domain[1]), y_range=(0.0, t_end),
        title="Conservation Law - Soft Model",
        var_names=("x", "t"),
    )
    plot_2d_prediction(
        model, target_fn, output / "prediction_hardened.png",
        x_range=(domain[0], domain[1]), y_range=(0.0, t_end),
        title="Conservation Law - Hardened Model",
        var_names=("x", "t"),
    )

    print(json.dumps({
        "experiment": experiment.name,
        "profile": experiment.profile,
        "steps": steps,
        "last_training_entry": history[-1],
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "plots": {
            "training_history": str(output / "training_history.png"),
            "prediction_soft": str(output / "prediction_soft.png"),
            "prediction_hardened": str(output / "prediction_hardened.png"),
        },
        "scientific_claim": "none; validate long runs and multiple seeds separately",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

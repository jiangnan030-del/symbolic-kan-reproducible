"""Discover a deep nested analytic formula with Symbolic-KAN.

Target:  y = exp(sin(x^2) + cos(x))

This formula requires three levels of composition (x -> x^2 -> sin/cos -> + -> exp),
making it a good benchmark for multi-block Symbolic-KAN architectures.

Inspired by Example_3_deep_formula from KindXiaoming/pykan, but using the
gated-primitive Symbolic-KAN architecture instead of B-spline KAN layers.

This is a derivative-package example, not upstream experiment code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from symbolic_kan import (
    ModelConfig,
    SymbolicKAN,
    TrainingConfig,
    fit_supervised,
    plot_1d_prediction,
    plot_training_history,
    rank_primitive_candidates,
    save_checkpoint,
    write_symbolic_report,
)


def main() -> int:
    torch.manual_seed(42)

    # -- Data: y = exp(sin(x^2) + cos(x)) --
    dtype = torch.float64
    x = (2.0 * torch.rand(512, 1, dtype=dtype) - 1.0) * 2.0  # x in [-2, 2]
    y = torch.exp(torch.sin(x.square()) + torch.cos(x))
    x_train, x_val = x[:384], x[384:]
    y_train, y_val = y[:384], y[384:]

    print("=" * 60)
    print("Deep Formula Discovery")
    print("Target: y = exp(sin(x^2) + cos(x))")
    print("=" * 60)
    print(f"Train: {x_train.shape[0]} samples, Val: {x_val.shape[0]} samples")
    print()

    # -- Model: 4 blocks to handle the 3-level composition --
    model = SymbolicKAN(
        ModelConfig(
            input_dim=1,
            hidden_units=6,
            edges_per_unit=3,
            num_blocks=4,
            primitives=("x", "x2", "sin", "cos", "exp", "const"),
            readout="fixed_sum",
            dtype="float64",
        )
    )
    training = TrainingConfig(
        seed=42,
        adam_epochs=200,
        lbfgs_steps=5,
        tau_start=4.0,
        tau_end=0.2,
        selection_weight_end=0.3,
        deterministic_algorithms=False,
    )

    print("Training...")
    result = fit_supervised(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        training,
        output_directory="outputs/deep-formula",
    )
    print(f"Best soft validation loss: {result.best_soft_validation_loss:.6e}")
    print(f"Best hardened validation loss: {result.best_hardened_validation_loss:.6e}")
    print()

    # -- Inspect candidates --
    soft_model = SymbolicKAN(model.config)
    soft_state_path = Path("outputs/deep-formula/model_best_soft.pt")
    try:
        soft_state = torch.load(soft_state_path, map_location="cpu", weights_only=True)
    except TypeError:
        soft_state = torch.load(soft_state_path, map_location="cpu")
    soft_model.load_state_dict(soft_state)
    soft_model.eval()

    candidates = rank_primitive_candidates(soft_model, top_k=3)
    print("Top primitive candidates (soft model):")
    for c in candidates[:6]:
        d = c.to_dict()
        print(f"  block={d['block']} unit={d['unit']} edge={d['edge']} "
              f"rank={d['rank']} prim={d['primitive']} score={d['score']:.4f}")
    print()

    # -- Export reports --
    output = Path("outputs/deep-formula")
    save_checkpoint(
        output / "checkpoint_soft.pt",
        soft_model,
        phase="soft",
        training_config=training,
        history=result.history,
        metadata={"profile": "deep-formula", "seed": 42, "target": "exp(sin(x^2)+cos(x))"},
    )
    soft_paths = write_symbolic_report(
        soft_model,
        output / "soft_report",
        variables=["x"],
        history=result.history,
        metadata={"profile": "deep-formula", "seed": 42, "target": "exp(sin(x^2)+cos(x))"},
    )
    save_checkpoint(
        output / "checkpoint_hardened.pt",
        result.model,
        phase="hardened",
        training_config=training,
        history=result.history,
        metadata={"profile": "deep-formula", "seed": 42, "target": "exp(sin(x^2)+cos(x))"},
    )
    hardened_paths = write_symbolic_report(
        result.model,
        output / "hardened_report",
        variables=["x"],
        history=result.history,
        metadata={"profile": "deep-formula", "seed": 42, "target": "exp(sin(x^2)+cos(x))"},
    )

    # -- Generate PNG plots --
    plot_training_history(
        result.history,
        output / "training_history.png",
        title="Deep Formula: y = exp(sin(x^2) + cos(x))",
    )

    target_fn = lambda x: np.exp(np.sin(x**2) + np.cos(x))
    plot_1d_prediction(
        soft_model, target_fn, output / "prediction_soft.png",
        x_range=(-2.0, 2.0),
        title="Deep Formula - Soft Model vs Ground Truth",
    )
    plot_1d_prediction(
        result.model, target_fn, output / "prediction_hardened.png",
        x_range=(-2.0, 2.0),
        title="Deep Formula - Hardened Model vs Ground Truth",
    )

    print("Artifacts:")
    print(json.dumps(
        {
            "soft_report": str(soft_paths["html"]),
            "hardened_report": str(hardened_paths["html"]),
            "hardened_expression": str(hardened_paths["expression"]),
            "training_history_png": str(output / "training_history.png"),
            "prediction_soft_png": str(output / "prediction_soft.png"),
            "prediction_hardened_png": str(output / "prediction_hardened.png"),
        },
        indent=2,
    ))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

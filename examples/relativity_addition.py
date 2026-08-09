"""Discover the relativistic velocity-addition formula with Symbolic-KAN.

Target:  v = tanh(artanh(v1) + artanh(v2))

This formula arises from special relativity: when two velocities are composed
in the same direction, the resulting velocity is not simply v1+v2 but follows
the Einstein velocity-addition law.  The formula can be rewritten as:

    v = (v1 + v2) / (1 + v1*v2)

but the tanh(log) form is more natural for KAN discovery since ``artanh`` is
``0.5 * log((1+x)/(1-x))`` and the model has ``log``, ``exp``, and ``tanh``
primitives available.

Inspired by Example_10_relativity-addition from KindXiaoming/pykan.

This is a derivative-package example, not upstream experiment code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from symbolic_kan import (
    ModelConfig,
    SymbolicKAN,
    TrainingConfig,
    fit_supervised,
    plot_2d_prediction,
    plot_training_history,
    rank_primitive_candidates,
    save_checkpoint,
    write_symbolic_report,
)


def main() -> int:
    torch.manual_seed(42)

    # -- Data: v = tanh(artanh(v1) + artanh(v2)) --
    dtype = torch.float64
    # v1, v2 in (-0.95, 0.95) to avoid artanh singularities
    n = 600
    v = 2.0 * torch.rand(n, 2, dtype=dtype) - 1.0
    v = v * 0.95
    v1, v2 = v[:, [0]], v[:, [1]]
    # Exact relativistic addition: (v1+v2)/(1+v1*v2)
    y = (v1 + v2) / (1.0 + v1 * v2)

    x_train, x_val = v[:400], v[400:]
    y_train, y_val = y[:400], y[400:]

    print("=" * 60)
    print("Relativistic Velocity Addition")
    print("Target: v = (v1 + v2) / (1 + v1*v2)")
    print("       = tanh(artanh(v1) + artanh(v2))")
    print("=" * 60)
    print(f"Train: {x_train.shape[0]} samples, Val: {x_val.shape[0]} samples")
    print()

    # -- Model: 2 inputs, 3 blocks for the tanh/log composition --
    model = SymbolicKAN(
        ModelConfig(
            input_dim=2,
            hidden_units=5,
            edges_per_unit=3,
            num_blocks=3,
            primitives=("x", "x2", "const", "log", "exp", "tanh", "inv", "sin", "cos"),
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
        output_directory="outputs/relativity-addition",
    )
    print(f"Best soft validation loss: {result.best_soft_validation_loss:.6e}")
    print(f"Best hardened validation loss: {result.best_hardened_validation_loss:.6e}")
    print()

    # -- Inspect candidates --
    soft_model = SymbolicKAN(model.config)
    soft_state_path = Path("outputs/relativity-addition/model_best_soft.pt")
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

    # -- Evaluate accuracy --
    model.eval()
    with torch.no_grad():
        pred = model(x_val)
        rel_error = (torch.linalg.vector_norm(pred - y_val)
                     / torch.linalg.vector_norm(y_val)).item()
    print(f"Hardened relative error on validation: {rel_error:.6e}")
    print()

    # -- Export reports --
    output = Path("outputs/relativity-addition")
    save_checkpoint(
        output / "checkpoint_soft.pt",
        soft_model,
        phase="soft",
        training_config=training,
        history=result.history,
        metadata={"profile": "relativity-addition", "seed": 42,
                  "target": "tanh(artanh(v1)+artanh(v2))"},
    )
    soft_paths = write_symbolic_report(
        soft_model,
        output / "soft_report",
        variables=["v1", "v2"],
        history=result.history,
        metadata={"profile": "relativity-addition", "seed": 42,
                  "target": "tanh(artanh(v1)+artanh(v2))"},
    )
    save_checkpoint(
        output / "checkpoint_hardened.pt",
        result.model,
        phase="hardened",
        training_config=training,
        history=result.history,
        metadata={"profile": "relativity-addition", "seed": 42,
                  "target": "tanh(artanh(v1)+artanh(v2))"},
    )
    hardened_paths = write_symbolic_report(
        result.model,
        output / "hardened_report",
        variables=["v1", "v2"],
        history=result.history,
        metadata={"profile": "relativity-addition", "seed": 42,
                  "target": "tanh(artanh(v1)+artanh(v2))"},
    )

    # -- Generate PNG plots --
    plot_training_history(
        result.history,
        output / "training_history.png",
        title="Relativity Addition: v = (v1+v2)/(1+v1*v2)",
    )

    target_fn = lambda X, Y: (X + Y) / (1.0 + X * Y)
    plot_2d_prediction(
        soft_model, target_fn, output / "prediction_soft.png",
        x_range=(-0.95, 0.95), y_range=(-0.95, 0.95),
        title="Relativity Addition - Soft Model",
        var_names=("v1", "v2"),
    )
    plot_2d_prediction(
        result.model, target_fn, output / "prediction_hardened.png",
        x_range=(-0.95, 0.95), y_range=(-0.95, 0.95),
        title="Relativity Addition - Hardened Model",
        var_names=("v1", "v2"),
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

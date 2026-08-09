"""Generate PNG plots for the Hello Symbolic-KAN tutorial outputs.

This script loads the already-trained checkpoints from
``tutorials/outputs/hello-symbolic-kan/`` and produces visualizations.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from symbolic_kan import ModelConfig, SymbolicKAN, plot_1d_prediction, plot_training_history


def main() -> int:
    output = Path("tutorials/outputs/hello-symbolic-kan")
    if not (output / "model_best_soft.pt").exists():
        print("Error: tutorial outputs not found. Run the notebook first.")
        return 1

    # Reconstruct the model config (must match the notebook)
    config = ModelConfig(
        input_dim=2, hidden_units=4, edges_per_unit=2, num_blocks=2,
        primitives=("x", "x2", "sin", "cos", "exp"),
        readout="fixed_sum", dtype="float64",
    )

    # Load history
    import json
    history_path = output / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []

    if history:
        plot_training_history(
            history, output / "training_history.png",
            title="Hello Symbolic-KAN: y = exp(sin(pi*x0) + x1^2)",
        )
        print(f"Saved: {output / 'training_history.png'}")

    # Target function: y = exp(sin(pi*x0) + x1^2)
    # For 1-D slice, fix x1=0.5
    target_fn = lambda x0: np.exp(np.sin(np.pi * x0) + 0.5**2)

    # Load soft model
    soft_model = SymbolicKAN(config)
    try:
        soft_state = torch.load(output / "model_best_soft.pt", map_location="cpu", weights_only=True)
    except TypeError:
        soft_state = torch.load(output / "model_best_soft.pt", map_location="cpu")
    soft_model.load_state_dict(soft_state)
    soft_model.eval()

    # Load hardened model
    hardened_model = SymbolicKAN(config)
    try:
        hardened_state = torch.load(output / "model_best_hardened.pt", map_location="cpu", weights_only=True)
    except TypeError:
        hardened_state = torch.load(output / "model_best_hardened.pt", map_location="cpu")
    hardened_model.load_state_dict(hardened_state)
    hardened_model.eval()

    # For 2-input model, create a wrapper that fixes x1=0.5
    class FixedX1Model:
        def __init__(self, model, x1_fixed=0.5):
            self.model = model
            self.x1_fixed = x1_fixed

        def eval(self):
            return self

        def __call__(self, x):
            x_full = torch.cat([
                x,
                torch.full_like(x, self.x1_fixed),
            ], dim=1)
            with torch.no_grad():
                return self.model(x_full)

    soft_wrapper = FixedX1Model(soft_model)
    hardened_wrapper = FixedX1Model(hardened_model)

    # We need to adapt plot_1d_prediction for the wrapper
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(-1.0, 1.0, 300)
    x_tensor = torch.tensor(x, dtype=torch.float64).reshape(-1, 1)
    true = target_fn(x)

    with torch.no_grad():
        soft_pred = soft_wrapper(x_tensor).numpy().flatten()
        hard_pred = hardened_wrapper(x_tensor).numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, pred, title in [
        (axes[0], soft_pred, "Soft Model (x1=0.5)"),
        (axes[1], hard_pred, "Hardened Model (x1=0.5)"),
    ]:
        ax.plot(x, true, "b-", linewidth=2, label="Ground truth")
        ax.plot(x, pred, "r--", linewidth=1.5, label="Prediction")
        ax.set_xlabel("x0")
        ax.set_ylabel("y")
        ax.set_title(f"Hello Symbolic-KAN - {title}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output / "prediction_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output / 'prediction_comparison.png'}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

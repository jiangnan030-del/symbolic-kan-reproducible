# SPDX-License-Identifier: MIT
# Visualization helpers for Symbolic-KAN experiment outputs.

"""Generate PNG plots for Symbolic-KAN experiments.

Each plot shows training loss history and model prediction vs. ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from symbolic_kan import SymbolicKAN


def plot_training_history(
    history: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Training History",
) -> None:
    """Plot training loss and components over steps."""

    steps = [h["step"] for h in history if h.get("phase") == "adam"]
    losses = [h["loss"] for h in history if h.get("phase") == "adam"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total loss (log scale)
    axes[0].semilogy(steps, losses, "b-", linewidth=1.5, label="total loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss (log)")
    axes[0].set_title(f"{title} - Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: loss components
    component_keys = [k for k in (history[0] if history else {}).keys()
                      if k not in ("phase", "step", "loss", "selection", "temperature")]
    for key in component_keys:
        values = [h.get(key, float("nan")) for h in history if h.get("phase") == "adam"]
        axes[1].semilogy(steps, values, linewidth=1.2, label=key)
    if "selection" in (history[0] if history else {}):
        sel_vals = [h.get("selection", float("nan")) for h in history if h.get("phase") == "adam"]
        axes[1].semilogy(steps, sel_vals, linewidth=1.2, label="selection", linestyle="--")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss component (log)")
    axes[1].set_title(f"{title} - Loss Components")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_1d_prediction(
    model: SymbolicKAN,
    target_fn,
    output_path: Path,
    *,
    x_range: tuple[float, float] = (-2.0, 2.0),
    title: str = "Prediction vs Ground Truth",
    variable_label: str = "x",
) -> None:
    """Plot model prediction against the analytic target for 1-D problems."""

    model.eval()
    x = torch.linspace(x_range[0], x_range[1], 300, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        pred = model(x).numpy().flatten()
    true = target_fn(x.numpy().flatten())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x.numpy().flatten(), true, "b-", linewidth=2, label="Ground truth")
    ax.plot(x.numpy().flatten(), pred, "r--", linewidth=1.5, label="Model prediction")
    ax.set_xlabel(variable_label)
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_prediction(
    model: SymbolicKAN,
    target_fn,
    output_path: Path,
    *,
    grid_size: int = 50,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
    title: str = "Prediction vs Ground Truth",
    var_names: tuple[str, str] = ("x", "y"),
) -> None:
    """Plot model prediction and ground truth as 2-D heatmaps for 2-input models."""

    model.eval()
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(xs, ys)
    coords = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float64
    )
    with torch.no_grad():
        pred = model(coords).numpy().reshape(grid_size, grid_size)
    true = target_fn(X, Y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ground truth
    im0 = axes[0].imshow(true, extent=[*x_range, *y_range], origin="lower", cmap="viridis", aspect="auto")
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel(var_names[0])
    axes[0].set_ylabel(var_names[1])
    plt.colorbar(im0, ax=axes[0])

    # Prediction
    im1 = axes[1].imshow(pred, extent=[*x_range, *y_range], origin="lower", cmap="viridis", aspect="auto")
    axes[1].set_title("Model Prediction")
    axes[1].set_xlabel(var_names[0])
    axes[1].set_ylabel(var_names[1])
    plt.colorbar(im1, ax=axes[1])

    # Absolute error
    error = np.abs(pred - true)
    im2 = axes[2].imshow(error, extent=[*x_range, *y_range], origin="lower", cmap="hot", aspect="auto")
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel(var_names[0])
    axes[2].set_ylabel(var_names[1])
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

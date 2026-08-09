"""Smoke tests for the deep-formula and relativity-addition examples."""

import torch

from symbolic_kan import ModelConfig, SymbolicKAN, TrainingConfig, fit_supervised


def test_deep_formula_short_run() -> None:
    """A 3-step training run should produce finite losses and a fitted model."""
    torch.manual_seed(99)
    dtype = torch.float64
    x = (2.0 * torch.rand(32, 1, dtype=dtype) - 1.0) * 2.0
    y = torch.exp(torch.sin(x.square()) + torch.cos(x))
    model = SymbolicKAN(
        ModelConfig(
            input_dim=1,
            hidden_units=3,
            edges_per_unit=2,
            num_blocks=2,
            primitives=("x", "x2", "sin", "cos", "exp"),
            readout="fixed_sum",
            dtype="float64",
        )
    )
    training = TrainingConfig(
        seed=99, adam_epochs=3, lbfgs_steps=0, tau_start=2.0, tau_end=0.5,
        selection_weight_end=0.01,
    )
    result = fit_supervised(model, x[:24], y[:24], x[24:], y[24:], training)
    assert torch.isfinite(torch.tensor(result.best_soft_validation_loss))
    adam_entries = [h for h in result.history if h.get("phase") == "adam"]
    assert len(adam_entries) == 3


def test_relativity_addition_short_run() -> None:
    """A 3-step training run on the velocity-addition target."""
    torch.manual_seed(77)
    dtype = torch.float64
    v = (2.0 * torch.rand(40, 2, dtype=dtype) - 1.0) * 0.9
    v1, v2 = v[:, [0]], v[:, [1]]
    y = (v1 + v2) / (1.0 + v1 * v2)
    model = SymbolicKAN(
        ModelConfig(
            input_dim=2,
            hidden_units=3,
            edges_per_unit=2,
            num_blocks=2,
            primitives=("x", "x2", "const", "log", "exp", "tanh", "inv"),
            readout="fixed_sum",
            dtype="float64",
        )
    )
    training = TrainingConfig(
        seed=77, adam_epochs=3, lbfgs_steps=0, tau_start=2.0, tau_end=0.5,
        selection_weight_end=0.01,
    )
    result = fit_supervised(model, v[:30], y[:30], v[30:], y[30:], training)
    assert torch.isfinite(torch.tensor(result.best_soft_validation_loss))
    adam_entries = [h for h in result.history if h.get("phase") == "adam"]
    assert len(adam_entries) == 3

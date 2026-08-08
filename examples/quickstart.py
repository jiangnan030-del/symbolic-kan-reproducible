"""Minimal API example for the unofficial attributed Symbolic-KAN package."""

import torch

from symbolic_kan import ModelConfig, SymbolicKAN, export_expression

config = ModelConfig(
    input_dim=1,
    hidden_units=4,
    edges_per_unit=2,
    num_blocks=2,
    primitives=("x", "x2", "sin", "cos", "exp"),
    readout="fixed_sum",
)
model = SymbolicKAN(config)
model.eval()
x = torch.linspace(-1.0, 1.0, 32).reshape(-1, 1).to(next(model.parameters()).dtype)
print(model(x).shape)
model.harden()
print(export_expression(model, variables=["x"]))

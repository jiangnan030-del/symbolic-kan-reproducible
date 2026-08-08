import torch

from symbolic_kan import ModelConfig, SymbolicKAN, export_expression, export_structure


def _model() -> SymbolicKAN:
    return SymbolicKAN(
        ModelConfig(
            input_dim=1,
            hidden_units=3,
            edges_per_unit=2,
            num_blocks=2,
            primitives=("x", "x2", "sin"),
            readout="fixed_sum",
            dtype="float64",
        )
    )


def test_evaluation_is_deterministic() -> None:
    model = _model().eval()
    x = torch.linspace(-1, 1, 16, dtype=torch.float64).reshape(-1, 1)
    first = model(x)
    second = model(x)
    assert torch.equal(first, second)


def test_hardening_creates_one_edge_per_unit() -> None:
    model = _model().harden()
    for block in model.blocks:
        assert block.edge_mask_active.item()
        assert torch.equal(block.edge_mask.sum(dim=1), torch.ones(block.hidden_units, dtype=block.edge_mask.dtype))
    structure = export_structure(model)
    assert structure["hardened"] is True
    assert export_expression(model, variables=["x"])

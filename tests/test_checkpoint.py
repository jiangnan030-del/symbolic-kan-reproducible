import torch

from symbolic_kan import (
    CHECKPOINT_SCHEMA_VERSION,
    ModelConfig,
    SymbolicKAN,
    checkpoint_summary,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_round_trip_is_device_agnostic(tmp_path) -> None:
    model = SymbolicKAN(
        ModelConfig(
            input_dim=1,
            hidden_units=2,
            edges_per_unit=2,
            num_blocks=1,
            primitives=("x", "x2"),
            dtype="float64",
        )
    ).eval()
    x = torch.linspace(-1, 1, 8, dtype=torch.float64).reshape(-1, 1)
    expected = model(x).detach()
    path = save_checkpoint(
        tmp_path / "model.pt",
        model,
        phase="soft",
        history=[{"phase": "test", "step": 1}],
        metadata={"profile": "smoke"},
    )

    loaded = load_checkpoint(path, device="cpu")
    actual = loaded.model(x).detach()
    assert torch.allclose(expected, actual)
    assert loaded.phase == "soft"
    assert loaded.metadata["profile"] == "smoke"

    summary = checkpoint_summary(path)
    assert summary["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert summary["history_entries"] == 1
    assert summary["provenance"]["upstream_commit"].startswith("9481a82")

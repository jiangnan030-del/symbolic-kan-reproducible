import json

from symbolic_kan import (
    ModelConfig,
    SymbolicKAN,
    prune_edges,
    prune_units,
    rank_primitive_candidates,
    write_symbolic_report,
)


def _model() -> SymbolicKAN:
    return SymbolicKAN(
        ModelConfig(
            input_dim=2,
            hidden_units=2,
            edges_per_unit=2,
            num_blocks=1,
            primitives=("x", "x2", "sin"),
            unit_gates=True,
            dtype="float64",
        )
    ).eval()


def test_candidate_ranking_is_complete_and_bounded() -> None:
    model = _model()
    candidates = rank_primitive_candidates(model, top_k=2)
    assert len(candidates) == 2 * 2 * 2
    assert {candidate.rank for candidate in candidates} == {1, 2}
    assert all(0.0 <= candidate.probability <= 1.0 for candidate in candidates)


def test_pruning_and_report_are_auditable(tmp_path) -> None:
    model = _model()
    unit_report = prune_units(model, threshold=0.5)
    edge_report = prune_edges(model, min_confidence=0.99)
    assert model.is_hardened
    assert unit_report["blocks"][0]["unit_gates"] is True
    assert len(edge_report["blocks"][0]["units"]) == 2

    paths = write_symbolic_report(
        model,
        tmp_path,
        variables=["x_0", "x_1"],
        metadata={"profile": "smoke"},
    )
    assert set(paths) == {"report", "expression", "structure", "html"}
    assert all(path.exists() for path in paths.values())
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["provenance"]["upstream_commit"].startswith("9481a82")
    assert report["summary"]["hardened"] is True
    assert "<svg" in paths["structure"].read_text(encoding="utf-8")
    assert "unofficial derivative" in paths["html"].read_text(encoding="utf-8").lower()

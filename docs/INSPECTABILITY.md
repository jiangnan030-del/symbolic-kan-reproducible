# Inspectability and discovery workflow

Version `0.1.0a2` makes model inspection a first-class, deterministic stage rather than a side effect of training.

## Workflow

```text
fit soft gates
  → inspect native primitive candidates
  → prune units and edges
  → harden discrete decisions
  → optionally refit continuous parameters
  → export a versioned checkpoint and audit report
```

## Native candidate evidence

`rank_primitive_candidates` ranks primitives already present on each edge. The score is deterministic gate probability minus an explicit complexity prior. It is not an R² value and does not prove that a selected primitive is the true governing law.

```python
from symbolic_kan import rank_primitive_candidates

candidates = rank_primitive_candidates(model, top_k=3, complexity_weight=0.02)
```

For scientific claims, compare selections across seeds and evaluate interpolation, extrapolation, derivatives, and dimensional consistency.

## Deterministic pruning

```python
from symbolic_kan import prune_edges, prune_units

unit_report = prune_units(model, threshold=0.5)
edge_report = prune_edges(model, min_confidence=0.75)
```

At least one unit per gated block is retained. Low-confidence edge winners are retained but explicitly marked so they cannot be silently mistaken for confident discoveries.

## Audit artifacts

```python
from symbolic_kan import write_symbolic_report

paths = write_symbolic_report(
    model,
    "outputs/run-001",
    variables=["x", "t"],
    metadata={"profile": "corrected", "seed": 42},
)
```

The output contains:

- `checkpoint_soft.pt`: pre-hardening gate evidence when created by `fit-demo`;
- `checkpoint_hardened.pt`: versioned, device-agnostic selected model state;
- `symbolic_report.json`: structure, candidate evidence, provenance, and metadata;
- `structure.svg`: dependency-free selected-structure diagram;
- `expression.txt`: exported hierarchical expression;
- `report.html`: portable human-readable audit report.

## CLI

```bash
symkan inspect --checkpoint checkpoint_soft.pt --output soft-report/
symkan prune --checkpoint checkpoint_soft.pt --output pruned.pt
symkan plot --checkpoint pruned.pt --output structure.svg
symkan export --checkpoint pruned.pt --output export/
```

Only load checkpoints from trusted sources because PyTorch checkpoints use pickle-compatible serialization.

## Scientific-integrity boundary

An exported formula is a result of this derivative implementation. It must not be presented as a value or formula supplied by the upstream authors unless independently verified against the upstream source and paper.

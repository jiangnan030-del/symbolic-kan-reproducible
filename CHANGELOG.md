# Changelog

All notable changes to this unofficial derivative are recorded here.

## 0.1.0-alpha.1 — 2026-08-08

### Added

- Standard `src/` Python package and `pyproject.toml`.
- Deterministic evaluation separate from stochastic Gumbel training.
- Explicit `fixed_sum` and `trainable_linear` readout modes.
- True pairwise edge-overlap (NMS) regularization and separately named off-mass term.
- Safe inverse primitive at zero.
- Structure hardening and hierarchical expression export.
- Correct variable-limit Gauss–Legendre Volterra quadrature with O(NQ) storage.
- Legacy, paper-aligned, corrected, and smoke configuration profiles.
- Unit tests, CI, provenance notice, citation metadata, and academic-integrity guidance.

### Important

This alpha release has not reproduced every table in the paper. Numerical results generated
with it must be labelled as derivative-package results and must not be attributed to the
upstream authors.

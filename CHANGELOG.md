# Changelog

All notable changes to this unofficial derivative are recorded here.

## 0.1.0-alpha.2 — 2026-08-09

### Added

- Deterministic native primitive candidate ranking with an explicit complexity prior.
- Auditable edge/unit pruning reports and confidence flags.
- Dependency-free selected-structure SVG output.
- JSON, text, SVG, and portable HTML symbolic audit reports.
- Versioned, atomic, device-agnostic checkpoints with model, optimizer, RNG, environment, and provenance metadata.
- `symkan inspect`, `prune`, `plot`, and `export` commands.
- A `hello_symbolic_kan` tutorial notebook and inspectability documentation.
- MkDocs documentation configuration and GitHub Pages workflow.
- Related-work documentation distinguishing PyKAN spline symbolicization from native discrete Symbolic-KAN selection.

### Changed

- `fit-demo` now writes soft and hardened checkpoints plus audit-report bundles.
- Package and citation versions advance to `0.1.0a2`.

### Scientific-integrity boundary

Candidate rankings are gate evidence from this derivative model, not proof of a governing law and not an upstream paper result. No PyKAN source code is copied in this release.

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
- Repository cover and social-preview artwork.
- English and Simplified Chinese repository overviews, badges, architecture diagram, and star history.
- Security policy, issue forms, CODEOWNERS, Dependabot, roadmap, and pre-release notes.

### Important

This alpha release has not reproduced every table in the paper. Numerical results generated
with it must be labelled as derivative-package results and must not be attributed to the
upstream authors.

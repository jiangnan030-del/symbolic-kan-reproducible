# Roadmap

This roadmap applies to the unofficial `symbolic-kan-reproducible` derivative package. It does not describe commitments by the upstream Symbolic-KAN authors.

## 0.1.0a1 — package and provenance

- [x] Standard `src/` package layout and CLI.
- [x] Deterministic evaluation and structure hardening.
- [x] Explicit `legacy`, `paper`, `corrected`, and `smoke` profiles.
- [x] Pairwise NMS, safe primitives, and mapped Volterra quadrature.
- [x] Unit tests, CI, attribution, citation, and academic-integrity guidance.
- [x] Repository cover, bilingual overview, badges, and community templates.
- [x] Publish the `v0.1.0a1` pre-release.

## 0.1.0a2 — inspectability and discovery

- [x] Rank native primitive candidates from deterministic gate evidence.
- [x] Add confidence-aware unit and edge pruning.
- [x] Add schema-versioned, device-agnostic checkpoints.
- [x] Export JSON, expression text, selected-structure SVG, and portable HTML reports.
- [x] Add `inspect`, `prune`, `plot`, and `export` CLI stages.
- [x] Add a CPU tutorial, related-work boundary, and MkDocs/Pages workflow.
- [x] Keep PyKAN inspiration separate from code origin; no PyKAN source copied.
- [ ] Confirm the full CI matrix on Python 3.10 and 3.12 for the final a2 commit.
- [ ] Run multi-seed stability and derivative-error validation before tagging a2.

## 0.1 stable — validation and distribution

- [ ] Add multi-seed reaction–diffusion and Volterra validation reports.
- [ ] Publish wheel and source distribution.
- [ ] Add coverage reporting after the CI baseline is stable.
- [ ] Archive a release with a DOI after the validation boundary is documented.
- [ ] Add benchmark artifacts without presenting derivative results as upstream results.

## 0.2 — research extensions

- [ ] Add PySINDy/PySR comparison adapters.
- [ ] Add optional PyKAN teacher/benchmark integration with pinned versions and third-party notices where required.
- [ ] Add dimensional and monotonicity constraints.
- [ ] Report structure-selection frequency across seeds.
- [ ] Add documented adapters for new scientific problems.

## Non-goals

- Reconstructing unpublished experiments and presenting them as author-provided code.
- Claiming exact paper reproduction without matching configurations, seeds, environments, and validation evidence.
- Implying endorsement by the original authors.

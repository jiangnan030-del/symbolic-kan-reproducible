# Roadmap

This roadmap applies to the unofficial `symbolic-kan-reproducible` derivative package. It does not describe commitments by the upstream Symbolic-KAN authors.

## 0.1 alpha — package and provenance

- [x] Standard `src/` package layout and CLI.
- [x] Deterministic evaluation and structure hardening.
- [x] Explicit `legacy`, `paper`, `corrected`, and `smoke` profiles.
- [x] Pairwise NMS, safe primitives, and mapped Volterra quadrature.
- [x] Unit tests, CI, attribution, citation, and academic-integrity guidance.
- [x] Repository cover, bilingual overview, badges, and community templates.
- [ ] Confirm the full CI matrix on Python 3.10 and 3.12.
- [ ] Publish signed `v0.1.0a1` pre-release artifacts.

## 0.1 stable — validation and distribution

- [ ] Add multi-seed reaction–diffusion and Volterra validation reports.
- [ ] Publish wheel and source distribution.
- [ ] Add coverage reporting after the CI baseline is stable.
- [ ] Archive a release with a DOI after the validation boundary is documented.
- [ ] Add benchmark artifacts without presenting derivative results as upstream results.

## 0.2 — research extensions

- [ ] Add PySINDy/PySR comparison adapters.
- [ ] Add dimensional and monotonicity constraints.
- [ ] Report structure-selection frequency across seeds.
- [ ] Add documented adapters for new scientific problems.

## Non-goals

- Reconstructing unpublished experiments and presenting them as author-provided code.
- Claiming exact paper reproduction without matching configurations, seeds, environments, and validation evidence.
- Implying endorsement by the original authors.

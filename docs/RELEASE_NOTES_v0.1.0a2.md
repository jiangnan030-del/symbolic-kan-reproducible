# v0.1.0a2 — Inspectability and discovery workflow

> **Unofficial derivative pre-release.** Symbolic-KAN, its paper, and the original experiments are the work of Salah A. Faroughi, Farinaz Mostajeran, Amirhossein Arzani, and Shirko Faroughi. This package is not endorsed by the original authors.

## Provenance

- Upstream repository: https://github.com/sfaroughi3/Pub_Symbolic_KANs
- Audited upstream commit: `9481a822e73e5a7520c6c0a425a8a402f2878c03`
- Derivative version: `0.1.0a2`

## Highlights

- Native primitive candidates ranked from deterministic gate evidence and an explicit complexity prior.
- Deterministic unit/edge pruning with uncertainty flags rather than silent overclaiming.
- Selected-structure SVG, JSON audit data, expression text, and portable HTML report.
- Versioned device-agnostic checkpoints with provenance, environment, RNG, optimizer, and history metadata.
- New `inspect`, `prune`, `plot`, and `export` CLI stages.
- Tutorial notebook, inspectability guide, related-work boundary, and GitHub Pages configuration.

## Related-work boundary

The workflow is informed by PyKAN's inspect–prune–symbolicize user experience, but this release does not copy PyKAN source code. PyKAN learns edge splines and may fit symbolic functions afterward; this package ranks and hardens its own native discrete primitive library.

## Validation boundary

Local static compilation and configuration validation are supplemented by the GitHub CI matrix. Long numerical reproduction runs, multi-seed structural stability, and paper-level benchmark claims remain out of scope for this alpha.

Candidate scores are not proof of the true governing equation. Report the exact profile, commit, seed, environment, and validation data with every scientific result.

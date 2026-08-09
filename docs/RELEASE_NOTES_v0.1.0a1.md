# v0.1.0a1 — Initial attributed alpha

> **Unofficial derivative release.** Symbolic-KAN, its paper, and the original experiments are the work of Salah A. Faroughi, Farinaz Mostajeran, Amirhossein Arzani, and Shirko Faroughi. This release is not endorsed by the original authors.

## Provenance

- Upstream repository: https://github.com/sfaroughi3/Pub_Symbolic_KANs
- Audited upstream commit: `9481a822e73e5a7520c6c0a425a8a402f2878c03`
- Derivative package version: `0.1.0a1`
- License: MIT; original copyright retained

## Highlights

- Installable `src/symbolic_kan` Python package and `symkan` CLI.
- Deterministic evaluation separated from stochastic Gumbel training.
- Structure hardening and hierarchical expression export.
- Explicit `fixed_sum` and `trainable_linear` readouts.
- Pairwise edge-overlap NMS with a separately named off-mass term.
- Safe inverse primitive and configurable positive parameterization.
- Variable-limit mapped Gauss–Legendre Volterra quadrature with O(NQ) storage.
- `legacy`, `paper`, `corrected`, and `smoke` profiles.
- Tests, CI, provenance metadata, and academic-integrity documentation.

## Installation

```bash
pip install "symbolic-kan-reproducible @ git+https://github.com/jiangnan030-del/symbolic-kan-reproducible.git@v0.1.0a1"
```

Use the commit URL instead until the tag is published.

## Validation boundary

The source package passed Python syntax compilation, TOML/YAML parsing, provenance checks, archive-integrity checks, and a common-secret-pattern scan in the packaging environment. Full PyTorch runtime tests and long numerical reproduction runs must be verified by GitHub Actions or a suitable scientific-computing environment.

This alpha does **not** claim independent reproduction of every table, seed, or long-run metric in the paper. Results generated with `corrected` are derivative-package results and must not be attributed to the upstream authors.

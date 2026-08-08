# Provenance and attribution notice

This repository is an **unofficial derivative research-software refactor** of the public
Symbolic-KAN implementation. It is not an official release by the paper authors and must
not be presented as one.

## Upstream work

- **Method/paper:** *Symbolic–KAN: Kolmogorov-Arnold networks with discrete symbolic
  structure for interpretable learning*
- **Authors:** Salah A. Faroughi, Farinaz Mostajeran, Amirhossein Arzani, and Shirko Faroughi
- **Original repository:** https://github.com/sfaroughi3/Pub_Symbolic_KANs
- **Audited source commit:** `9481a822e73e5a7520c6c0a425a8a402f2878c03`
- **Original code credits stated upstream:** Prof. Salah A. Faroughi (original author) and
  Dr. Farinaz Mostajeran (code modifications and improvements)
- **Original license:** MIT, Copyright (c) 2024 Prof. Salah A. Faroughi

The upstream repository did not contain a root-level license file at the audited commit,
but each inspected Python source file included the MIT license text. That original notice
is preserved in this repository's `LICENSE` file.

## What this repository adds

This derivative introduces a standard Python package layout, deterministic evaluation,
configuration objects, explicit legacy/paper/corrected experiment profiles, tests,
continuous integration, safer primitives, true pairwise edge-overlap regularization,
fixed-sum and trainable readout modes, corrected variable-limit Volterra quadrature,
and reproducibility metadata.

Some modules are refactored or cleanly reimplemented from the public upstream behavior.
No claim is made that the package maintainer invented Symbolic-KAN, authored the original
paper, produced the paper's reported results, or is endorsed by the upstream authors.

## Required scholarly practice

When using this software in a paper, thesis, report, presentation, or derived repository:

1. Cite the original Symbolic-KAN paper and its authors as the source of the method.
2. Cite or link the original GitHub repository and record the upstream commit.
3. Identify this repository as an unofficial derivative/refactor.
4. Report this package's version/commit and the exact experiment profile used.
5. Distinguish upstream results from results produced with `legacy`, `paper`, or
   `corrected` configurations in this package.
6. Do not imply endorsement by the original authors.

See `CITATION.cff` and `docs/ACADEMIC_INTEGRITY.md` for a ready-to-use citation statement.

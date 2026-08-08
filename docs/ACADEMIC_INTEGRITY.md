# Academic integrity and citation guidance

## Ownership

The Symbolic-KAN method and original public implementation are credited to the paper
and upstream repository listed below. This derivative package does not transfer or erase
that authorship.

- Salah A. Faroughi, Farinaz Mostajeran, Amirhossein Arzani, Shirko Faroughi,
  *Symbolic–KAN: Kolmogorov-Arnold networks with discrete symbolic structure for
  interpretable learning* (2026).
- Upstream code: https://github.com/sfaroughi3/Pub_Symbolic_KANs
- Audited commit: `9481a822e73e5a7520c6c0a425a8a402f2878c03`

## Required reporting

For every published experiment, record:

1. the original paper citation;
2. the upstream repository URL and commit;
3. this derivative repository's commit and package version;
4. the exact YAML configuration and random seed;
5. whether the profile is `legacy`, `paper`, `corrected`, or `smoke`;
6. hardware, Python, PyTorch, dtype, and deterministic settings;
7. all algorithmic differences that can affect results.

Do not reuse tables, plots, or numerical values from the paper without explicit citation.
Do not label modified or newly generated results as “the authors’ results.” Do not imply
that the upstream authors reviewed or endorsed this refactor.

## Ready-to-use methods statement

> The model architecture follows Symbolic-KAN as introduced by Faroughi et al. We used
> the authors’ public repository (`sfaroughi3/Pub_Symbolic_KANs`, commit `9481a82`) as
> the provenance baseline. Our experiments were executed with the unofficial
> `symbolic-kan-reproducible` derivative at commit `<DERIVATIVE_SHA>` using the
> `<PROFILE>` configuration. The derivative changes include `<LIST MATERIAL CHANGES>`;
> consequently, reported values are our results and should not be interpreted as values
> produced or validated by the original authors.

## Redistributing code

The upstream files contain an MIT license. Keep `LICENSE`, `NOTICE.md`, the upstream URL,
and the audited commit in every substantial redistribution. Preserve source-level SPDX
and provenance comments when copying package modules.

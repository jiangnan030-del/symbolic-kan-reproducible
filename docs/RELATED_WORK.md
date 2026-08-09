# Related work and code-origin boundary

## Symbolic-KAN upstream

The method and original public implementation used as the provenance baseline are:

- *Symbolic–KAN: Kolmogorov-Arnold networks with discrete symbolic structure for interpretable learning*
- Salah A. Faroughi, Farinaz Mostajeran, Amirhossein Arzani, Shirko Faroughi
- <https://github.com/sfaroughi3/Pub_Symbolic_KANs>
- audited commit `9481a822e73e5a7520c6c0a425a8a402f2878c03`

See `NOTICE.md` and `ACADEMIC_INTEGRITY.md` for required attribution.

## PyKAN

[PyKAN](https://github.com/KindXiaoming/pykan) is the MIT-licensed reference repository associated with KAN and KAN 2.0. It learns edge spline functions and supports visualization, sparsification, pruning, symbolic fitting, refinement, and scientific tutorials.

The inspectability workflow in `0.1.0a2` was informed by PyKAN's user-facing sequence—inspect, prune, symbolicize, and export—but does **not** copy PyKAN source code. This package instead ranks its own native discrete primitive gates and retains the Symbolic-KAN architecture and provenance boundary.

Reference inspected:

- repository: <https://github.com/KindXiaoming/pykan>
- commit: `ecde4ec3274d3bef1ad737479cf126aed38ab530`
- release: `v0.2.8`
- license: MIT, Copyright (c) 2024 Ziming Liu

If future work copies or modifies PyKAN code, the affected files must retain its MIT notice and be listed in a third-party notices file. Benchmark use must record the exact PyKAN version, commit, configuration, and seed.

## Important conceptual distinction

- PyKAN: learns flexible spline functions on edges and may fit symbolic functions afterward.
- This package: searches a discrete analytic primitive library directly, then hardens and audits that structure.

Comparisons are useful, but the two implementations and their numerical results must not be conflated.

# Two-dimensional Laplace case

> **Scope:** this is a new adapter in the unofficial derivative package. The benchmark equation and settings are transcribed from the Symbolic-KAN manuscript. A matching Laplace experiment was not present in the audited upstream repository at commit `9481a82`, so outputs from this adapter must not be presented as upstream results or as a verified paper reproduction.

## Problem

The case solves the harmonic boundary-value problem

\[
\nabla^2 u = u_{xx} + u_{yy} = 0,\qquad (x,y)\in[0,1]^2,
\]

with Dirichlet data from

\[
u(x,y)=\sin(\pi x)\sinh(\pi y).
\]

`LaplaceProblem` provides:

- exact-solution evaluation;
- uniform interior sampling;
- an exact requested number of samples distributed over all four boundaries;
- second-order automatic-differentiation residuals;
- PDE and boundary-loss components.

The derivative helper keeps a zero-valued graph connection for constant or linear selected branches, so evaluating second derivatives remains valid during symbolic structure changes.

## Profiles

| Profile | Purpose | Important boundary |
| --- | --- | --- |
| `smoke` | Five-step API and CI exercise | Not a scientific benchmark |
| `paper` | Manuscript-described architecture, point counts, and primitive library | Some optimizer details are derivative defaults where the manuscript is not explicit |
| `corrected` | Float64, deterministic algorithms, and soft edge exploration | Results belong to this derivative implementation |

Configurations live under `experiments/laplace/configs/`.

## Run the case

From the repository root:

```bash
python examples/laplace_physics_informed.py \
  --config experiments/laplace/configs/smoke.yaml \
  --output outputs/laplace-smoke
```

The example runs a compact Adam structure-search loop, reports soft and hardened grid errors, and writes versioned checkpoints plus JSON, expression, SVG, and HTML audit bundles.

The generic construction smoke test also accepts the new two-input profile:

```bash
symkan smoke --config experiments/laplace/configs/smoke.yaml
```

That command validates construction, deterministic evaluation, and hardening only; it does not train the PDE case.

## Validation before scientific use

A serious experiment should additionally record:

1. interpolation and dense-grid relative error;
2. the Laplacian residual on an independent grid;
3. boundary error by side;
4. soft-to-hardened prediction drift;
5. selected structure and confidence for every seed;
6. at least 8–20 seeds and environment metadata;
7. a clear distinction between manuscript-described settings and derivative corrections.

The included unit tests verify that the analytic target is harmonic, boundary sampling has the requested size, the physics loss is finite/differentiable, and the smoke profile parses.

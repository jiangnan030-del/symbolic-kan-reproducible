# Reproducibility protocol

## Minimum record for every run

The run directory should contain:

- resolved YAML configuration;
- package version and Git commit;
- Python, PyTorch, CUDA/MPS, OS, device, and dtype information;
- all random seeds and deterministic flags;
- optimizer parameter groups, including effective weight decay;
- best soft and best hardened checkpoints;
- epoch-level metrics without stale values;
- selected primitive/edge/unit structure;
- exported expression and validation-grid equivalence error.

## Recommended workflow

1. Run `symkan smoke` with the intended environment.
2. Run the `legacy` profile to establish behavioral continuity where practical.
3. Run `paper` without silently borrowing corrected settings.
4. Run `corrected` as a separate experiment and report changes.
5. Use at least five seeds for development and at least twenty for structural claims.
6. Report mean, standard deviation, failures, and primitive/edge selection frequencies.
7. Archive the exact commit, configuration, and raw logs with each table or figure.

## Numeric checks

- Repeated evaluation of one checkpoint must be deterministic.
- Hardened network output must match the exported discrete evaluator on a fixed grid.
- Variable-limit Volterra quadrature must converge against an analytic integral.
- Second-derivative problems should be compared in float32 and float64.
- Long L-BFGS runs should not use gradient clipping unless the deviation is documented.

## Current alpha limits

The package supplies core architecture, a two-stage supervised trainer, physics problem
helpers, and smoke configurations. It does not claim completed long-run reproduction of
all manuscript tables. Missing upstream experiments are not reconstructed and presented
as if they were author-provided code.

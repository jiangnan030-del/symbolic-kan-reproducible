# Validation status

## Completed in the packaging environment

- All Python source, tests, and examples pass `compileall` syntax compilation.
- `pyproject.toml` parses successfully with Python's `tomllib`.
- Every YAML profile parses successfully and records the audited upstream commit.
- A targeted scan found no common credential or private-key patterns.
- Root `LICENSE`, `NOTICE.md`, `CITATION.cff`, and source-level provenance comments are present.

## Pending automated runtime checks

The packaging sandbox did not contain PyTorch and had no outbound package-network access,
so tensor-level pytest execution and wheel construction were not run locally. GitHub
Actions is configured to install dependencies, run the test suite on Python 3.10 and
3.12, and build the distribution after the repository is uploaded.

Until that CI run passes, this artifact should be treated as `0.1.0a1`, not as a stable or
paper-reproducing release.

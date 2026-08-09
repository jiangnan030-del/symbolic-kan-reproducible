# Source manifest scope

`SOURCE_MANIFEST.json` is a frozen integrity manifest for the initial `0.1.0a1` source-package snapshot. Its `version` field is intentionally `0.1.0a1`, and its file hashes must not be interpreted as hashes for current `main` or `0.1.0a2`.

Reference snapshot:

- version: `0.1.0a1`
- repository commit represented by the initial release preparation: `aee554b746dd6869fa85ad82a3f08f17f27df340`
- upstream provenance commit: `9481a822e73e5a7520c6c0a425a8a402f2878c03`

Before tagging `v0.1.0a2`, generate a new complete SHA-256 manifest from the exact release tree and record the release commit. Do not edit old hashes to make them appear current.

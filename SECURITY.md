# Security policy

## Supported versions

This project is early-stage research software. Security fixes are provided on a best-effort basis for the latest commit on `main` and the latest published alpha release.

| Version | Supported |
| --- | --- |
| `main` | Best effort |
| `0.1.x` alpha | Best effort |
| Older snapshots | No |

## Reporting a vulnerability

Please use GitHub's **Security → Report a vulnerability** flow when it is available. Include:

- affected version or commit;
- a minimal reproduction;
- likely impact;
- suggested mitigation, if known.

Do not publish secrets, credentials, weaponized exploits, or private research data in a public issue. If private vulnerability reporting is unavailable, open a minimal public issue requesting a private maintainer contact without including exploit details.

## Research-software scope

This package is not intended for safety-critical, clinical, autonomous-control, or production security decisions. Numerical instability or scientific-regression concerns should normally use the reproducibility issue form rather than the security channel.

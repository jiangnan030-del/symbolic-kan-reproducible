# SPDX-License-Identifier: MIT
# Derived research software; see NOTICE.md for upstream attribution.

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy, CPU, CUDA, and MPS-visible PyTorch state."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=True)
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name}")


def environment_snapshot() -> dict[str, Any]:
    """Return environment metadata suitable for a run manifest."""

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }

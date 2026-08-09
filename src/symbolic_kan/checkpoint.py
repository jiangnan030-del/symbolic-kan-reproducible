# SPDX-License-Identifier: MIT
# Derived research software; see NOTICE.md for upstream attribution.

"""Versioned, device-agnostic checkpoints for auditable research runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .model import SymbolicKAN
from .reproducibility import environment_snapshot

CHECKPOINT_SCHEMA_VERSION = 1
UPSTREAM_REPOSITORY = "https://github.com/sfaroughi3/Pub_Symbolic_KANs"
UPSTREAM_COMMIT = "9481a822e73e5a7520c6c0a425a8a402f2878c03"


@dataclass(slots=True)
class CheckpointLoadResult:
    """Loaded model plus recoverable training and provenance state."""

    model: SymbolicKAN
    phase: str
    history: list[dict[str, Any]]
    optimizer_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    metadata: dict[str, Any]
    checkpoint: dict[str, Any]


def _package_version() -> str:
    try:
        return version("symbolic-kan-reproducible")
    except PackageNotFoundError:
        return "development"


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError("configuration metadata must be a dataclass, mapping, or None")


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _load_payload(path: str | Path, map_location: str | torch.device) -> dict[str, Any]:
    # Checkpoints use pickle internally. Only load files from trusted sources.
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported checkpoint schema: "
            f"{payload.get('schema_version')!r}; expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    return payload


def save_checkpoint(
    path: str | Path,
    model: SymbolicKAN,
    *,
    phase: str,
    training_config: Any = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    history: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save a CPU-portable model and optional training state."""

    if not phase.strip():
        raise ValueError("phase must be non-empty")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "package": "symbolic-kan-reproducible",
        "package_version": _package_version(),
        "status": "unofficial derivative",
        "provenance": {
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
        },
        "phase": phase,
        "model_config": asdict(model.config),
        "training_config": _as_mapping(training_config),
        "model_state": _to_cpu(model.state_dict()),
        "optimizer_state": None if optimizer is None else _to_cpu(optimizer.state_dict()),
        "scheduler_state": None if scheduler is None else _to_cpu(scheduler.state_dict()),
        "history": list(history or []),
        "metadata": dict(metadata or {}),
        "environment": environment_snapshot(),
        "rng_state": {
            "torch_cpu": torch.get_rng_state().cpu(),
            "torch_cuda": None
            if not torch.cuda.is_available()
            else [state.cpu() for state in torch.cuda.get_rng_state_all()],
        },
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return output


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = False,
) -> CheckpointLoadResult:
    """Load a trusted versioned checkpoint on a caller-selected device."""

    payload = _load_payload(path, map_location=device)
    config = ModelConfig(**dict(payload["model_config"]))
    model = SymbolicKAN(config)
    model.load_state_dict(payload["model_state"], strict=strict)
    model.to(device)

    if model.is_hardened:
        for block in model.blocks:
            for unit_edges in block.edges:
                for edge in unit_edges:
                    if bool(edge.is_hardened.item()):
                        edge.logits.requires_grad_(False)
            if block.unit_logits is not None and bool(block.unit_gates_hardened.item()):
                block.unit_logits.requires_grad_(False)
    model.eval()

    if restore_rng:
        rng_state = payload.get("rng_state", {})
        cpu_state = rng_state.get("torch_cpu")
        if cpu_state is not None:
            torch.set_rng_state(cpu_state.cpu())
        cuda_states = rng_state.get("torch_cuda")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)

    return CheckpointLoadResult(
        model=model,
        phase=str(payload.get("phase", "unknown")),
        history=list(payload.get("history", [])),
        optimizer_state=payload.get("optimizer_state"),
        scheduler_state=payload.get("scheduler_state"),
        metadata=dict(payload.get("metadata", {})),
        checkpoint=payload,
    )


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    """Inspect trusted checkpoint metadata without constructing the model."""

    payload = _load_payload(path, map_location="cpu")
    model_state = payload.get("model_state", {})
    return {
        "schema_version": payload["schema_version"],
        "package": payload.get("package"),
        "package_version": payload.get("package_version"),
        "status": payload.get("status"),
        "phase": payload.get("phase"),
        "model_config": payload.get("model_config"),
        "training_config": payload.get("training_config"),
        "provenance": payload.get("provenance"),
        "environment": payload.get("environment"),
        "history_entries": len(payload.get("history", [])),
        "state_tensor_count": sum(
            isinstance(value, torch.Tensor) for value in model_state.values()
        ),
        "state_value_count": sum(
            int(value.numel())
            for value in model_state.values()
            if isinstance(value, torch.Tensor)
        ),
        "metadata": payload.get("metadata", {}),
    }

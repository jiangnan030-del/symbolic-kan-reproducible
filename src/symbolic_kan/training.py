# SPDX-License-Identifier: MIT
# Training flow derived from sfaroughi3/Pub_Symbolic_KANs at 9481a82.
# See NOTICE.md for full authorship, license, and non-endorsement information.

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import TrainingConfig
from .model import SymbolicKAN, cosine_temperature
from .regularization import primitive_bias_penalty, selection_terms, unit_gate_penalty
from .reproducibility import environment_snapshot, seed_everything


@dataclass(slots=True)
class FitResult:
    model: SymbolicKAN
    history: list[dict[str, float | int | str]]
    best_soft_validation_loss: float
    best_hardened_validation_loss: float
    output_directory: Path | None


def _is_gate_parameter(name: str) -> bool:
    return name.endswith(".logits") or name.endswith(".unit_logits")


def build_adamw(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """Build explicit parameter groups so configured decay cannot be silently ignored."""

    gate_parameters: list[nn.Parameter] = []
    normal_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (gate_parameters if _is_gate_parameter(name) else normal_parameters).append(parameter)

    groups: list[dict[str, Any]] = []
    if normal_parameters:
        groups.append(
            {
                "params": normal_parameters,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            }
        )
    if gate_parameters:
        groups.append(
            {
                "params": gate_parameters,
                "lr": config.learning_rate * config.gate_learning_rate_scale,
                "weight_decay": config.gate_weight_decay,
                "betas": (0.0, 0.999),
            }
        )
    return torch.optim.AdamW(groups)


def _relative_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(prediction - target) / (torch.linalg.vector_norm(target) + 1e-12)).item())


def _selection_weight(epoch: int, config: TrainingConfig) -> float:
    start = int(config.selection_start_fraction * max(config.adam_epochs, 1))
    if epoch < start:
        return 0.0
    progress = (epoch - start) / max(1, config.adam_epochs - start)
    return config.selection_weight_start + (
        config.selection_weight_end - config.selection_weight_start
    ) * progress**2


def _validation(model: SymbolicKAN, features: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(features)
        mse = float(torch.mean((prediction - target).square()).item())
        relative = _relative_error(prediction, target)
    return mse, relative


def _write_run_metadata(
    directory: Path,
    model: SymbolicKAN,
    config: TrainingConfig,
    history: list[dict[str, float | int | str]],
    optimizer: torch.optim.Optimizer,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "package": "symbolic-kan-reproducible",
        "upstream_repository": "https://github.com/sfaroughi3/Pub_Symbolic_KANs",
        "upstream_commit": "9481a822e73e5a7520c6c0a425a8a402f2878c03",
        "model": asdict(model.config),
        "training": asdict(config),
        "environment": environment_snapshot(),
        "optimizer_groups": [
            {
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
                "betas": group.get("betas"),
                "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in optimizer.param_groups
        ],
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    (directory / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def fit_supervised(
    model: SymbolicKAN,
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    validation_features: torch.Tensor,
    validation_target: torch.Tensor,
    config: TrainingConfig,
    *,
    output_directory: str | Path | None = None,
) -> FitResult:
    """Fit a supervised Symbolic-KAN with AdamW, hardening, and optional L-BFGS.

    Physics-informed callers can use the same model/regularization API in a custom loop.
    This function deliberately restores the best soft state before hardening and the best
    hardened state before returning/exporting.
    """

    seed_everything(config.seed, config.deterministic_algorithms)
    directory = Path(output_directory) if output_directory is not None else None
    optimizer = build_adamw(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.adam_epochs, 1)
    )

    history: list[dict[str, float | int | str]] = []
    best_soft_loss = float("inf")
    best_soft_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.adam_epochs + 1):
        model.train()
        model.set_temperature(
            cosine_temperature(
                epoch, config.adam_epochs, config.tau_start, config.tau_end
            )
        )
        optimizer.zero_grad(set_to_none=True)
        prediction, gates = model(train_features, return_aux=True)
        mse = torch.mean((prediction - train_target).square())
        terms = selection_terms(gates.primitive_probabilities)
        selection = terms.weighted(
            sharpness_weight=config.sharpness_weight,
            entropy_weight=config.entropy_weight,
            nms_weight=config.nms_weight,
            off_mass_weight=config.off_mass_weight,
        )
        selection_scale = _selection_weight(epoch, config)
        unit_penalty = unit_gate_penalty(gates.unit_probabilities).to(mse.device)
        bias_penalty = primitive_bias_penalty(model)
        total = (
            mse
            + selection_scale * selection
            + config.unit_gate_weight * unit_penalty
            + config.primitive_bias_weight * bias_penalty
        )
        total.backward()
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        scheduler.step()

        validation_loss, relative_error = _validation(
            model, validation_features, validation_target
        )
        if validation_loss < best_soft_loss:
            best_soft_loss = validation_loss
            best_soft_state = copy.deepcopy(model.state_dict())
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)
                torch.save(best_soft_state, directory / "model_best_soft.pt")

        history.append(
            {
                "phase": "adam",
                "step": epoch,
                "loss": float(total.detach().item()),
                "train_mse": float(mse.detach().item()),
                "selection": float(selection.detach().item()),
                "selection_scale": float(selection_scale),
                "validation_mse": validation_loss,
                "relative_error": relative_error,
                "temperature": float(config.tau_end if config.adam_epochs <= 1 else cosine_temperature(epoch, config.adam_epochs, config.tau_start, config.tau_end)),
            }
        )

    if best_soft_state is not None:
        model.load_state_dict(best_soft_state)
    model.harden(freeze=True)

    best_hardened_loss, hardened_relative_error = _validation(
        model, validation_features, validation_target
    )
    best_hardened_state = copy.deepcopy(model.state_dict())
    history.append(
        {
            "phase": "harden",
            "step": 0,
            "loss": best_hardened_loss,
            "validation_mse": best_hardened_loss,
            "relative_error": hardened_relative_error,
        }
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.lbfgs_steps > 0 and trainable:
        lbfgs = torch.optim.LBFGS(
            trainable,
            lr=config.lbfgs_learning_rate,
            max_iter=config.lbfgs_max_iter,
            history_size=config.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )
        for step in range(1, config.lbfgs_steps + 1):
            state: dict[str, float] = {}

            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                prediction = model(train_features)
                loss = torch.mean((prediction - train_target).square())
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite L-BFGS objective")
                loss.backward()
                if config.lbfgs_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(trainable, config.lbfgs_clip_norm)
                state["loss"] = float(loss.detach().item())
                return loss

            lbfgs.step(closure)
            validation_loss, relative_error = _validation(
                model, validation_features, validation_target
            )
            if validation_loss < best_hardened_loss:
                best_hardened_loss = validation_loss
                best_hardened_state = copy.deepcopy(model.state_dict())
            history.append(
                {
                    "phase": "lbfgs",
                    "step": step,
                    "loss": state.get("loss", float("nan")),
                    "validation_mse": validation_loss,
                    "relative_error": relative_error,
                }
            )

    model.load_state_dict(best_hardened_state)
    model.eval()
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(best_hardened_state, directory / "model_best_hardened.pt")
        _write_run_metadata(directory, model, config, history, optimizer)

    return FitResult(
        model=model,
        history=history,
        best_soft_validation_loss=best_soft_loss,
        best_hardened_validation_loss=best_hardened_loss,
        output_directory=directory,
    )

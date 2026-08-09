# SPDX-License-Identifier: MIT
# Derived research software; see NOTICE.md for upstream attribution.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .checkpoint import checkpoint_summary, load_checkpoint, save_checkpoint
from .config import load_experiment_config
from .export import export_expression, export_structure
from .inspection import prune_edges, write_structure_svg, write_symbolic_report
from .model import SymbolicKAN
from .reproducibility import resolve_dtype
from .training import fit_supervised

UPSTREAM = "https://github.com/sfaroughi3/Pub_Symbolic_KANs"
UPSTREAM_COMMIT = "9481a822e73e5a7520c6c0a425a8a402f2878c03"


def _model_from_config(path: str) -> tuple[SymbolicKAN, Any]:
    experiment = load_experiment_config(path)
    return SymbolicKAN(experiment.model, initial_temperature=experiment.training.tau_start), experiment


def _variables(model: SymbolicKAN) -> list[str]:
    if model.config.input_dim == 1:
        return ["x"]
    return [f"x_{index}" for index in range(model.config.input_dim)]


def _stringify_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {name: str(path) for name, path in paths.items()}


def _load_internal_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a state dict written by this process during the current fit."""

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"expected a state-dict mapping: {path}")
    return state


def command_info(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "package": "symbolic-kan-reproducible",
                "version": __version__,
                "status": "unofficial derivative alpha",
                "upstream": UPSTREAM,
                "upstream_commit": UPSTREAM_COMMIT,
                "citation_notice": "Cite the original paper and repository; see NOTICE.md.",
            },
            indent=2,
        )
    )
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    model, experiment = _model_from_config(args.config)
    dtype = resolve_dtype(experiment.model.dtype)
    domain = experiment.problem.get("domain", [-1.0, 1.0])
    x = torch.linspace(
        float(domain[0]), float(domain[1]), args.points, dtype=dtype
    ).reshape(-1, 1)
    if experiment.model.input_dim > 1:
        x = x.repeat(1, experiment.model.input_dim)
    model.eval()
    with torch.no_grad():
        first = model(x)
        second = model(x)
    deterministic_delta = float((first - second).abs().max().item())
    model.harden()
    with torch.no_grad():
        hardened = model(x)
    report = {
        "experiment": experiment.name,
        "profile": experiment.profile,
        "deterministic_eval_max_delta": deterministic_delta,
        "hardened_output_shape": list(hardened.shape),
        "hardened_output_finite": bool(torch.isfinite(hardened).all().item()),
        "selected_structure": export_structure(model),
        "expression": export_expression(model, variables=_variables(model)),
    }
    print(json.dumps(report, indent=2))
    return 0 if deterministic_delta == 0.0 and report["hardened_output_finite"] else 1


def command_fit_demo(args: argparse.Namespace) -> int:
    model, experiment = _model_from_config(args.config)
    dtype = resolve_dtype(experiment.model.dtype)
    generator = torch.Generator().manual_seed(experiment.training.seed)
    x_train = (
        2.0 * torch.rand(args.train_points, 1, generator=generator, dtype=dtype) - 1.0
    )
    y_train = x_train.square()
    x_val = torch.linspace(-1.0, 1.0, args.validation_points, dtype=dtype).reshape(
        -1, 1
    )
    y_val = x_val.square()
    result = fit_supervised(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        experiment.training,
        output_directory=args.output,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"experiment": experiment.name, "profile": experiment.profile}
    soft_state_path = output / "model_best_soft.pt"
    soft_checkpoint: Path | None = None
    soft_artifacts: dict[str, Path] | None = None
    if soft_state_path.exists():
        soft_model, _ = _model_from_config(args.config)
        soft_model.load_state_dict(_load_internal_state_dict(soft_state_path))
        soft_model.eval()
        soft_checkpoint = save_checkpoint(
            output / "checkpoint_soft.pt",
            soft_model,
            phase="soft",
            training_config=experiment.training,
            history=result.history,
            metadata=metadata,
        )
        soft_artifacts = write_symbolic_report(
            soft_model,
            output / "soft_report",
            variables=_variables(soft_model),
            history=result.history,
            metadata=metadata,
        )

    hardened_checkpoint = save_checkpoint(
        output / "checkpoint_hardened.pt",
        result.model,
        phase="hardened",
        training_config=experiment.training,
        history=result.history,
        metadata=metadata,
    )
    hardened_artifacts = write_symbolic_report(
        result.model,
        output / "hardened_report",
        variables=_variables(result.model),
        history=result.history,
        metadata=metadata,
    )
    print(
        json.dumps(
            {
                "best_soft_validation_loss": result.best_soft_validation_loss,
                "best_hardened_validation_loss": result.best_hardened_validation_loss,
                "soft_checkpoint": None
                if soft_checkpoint is None
                else str(soft_checkpoint),
                "hardened_checkpoint": str(hardened_checkpoint),
                "soft_artifacts": None
                if soft_artifacts is None
                else _stringify_paths(soft_artifacts),
                "hardened_artifacts": _stringify_paths(hardened_artifacts),
            },
            indent=2,
        )
    )
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    summary = checkpoint_summary(args.checkpoint)
    if args.output is not None:
        loaded = load_checkpoint(args.checkpoint, device=args.device)
        artifacts = write_symbolic_report(
            loaded.model,
            args.output,
            variables=_variables(loaded.model),
            history=loaded.history,
            metadata=loaded.metadata,
            top_k=args.top_k,
            complexity_weight=args.complexity_weight,
        )
        summary["artifacts"] = _stringify_paths(artifacts)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def command_prune(args: argparse.Namespace) -> int:
    loaded = load_checkpoint(args.checkpoint, device=args.device)
    edge_report = prune_edges(
        loaded.model,
        min_confidence=args.edge_min_confidence,
        unit_threshold=args.unit_threshold,
        freeze=True,
    )
    unit_report = edge_report["unit_pruning"]
    output_checkpoint = save_checkpoint(
        args.output,
        loaded.model,
        phase="pruned",
        training_config=loaded.checkpoint.get("training_config"),
        history=loaded.history,
        metadata={
            **loaded.metadata,
            "pruning": {"units": unit_report, "edges": edge_report},
            "source_checkpoint": str(args.checkpoint),
        },
    )
    output_path = Path(args.output)
    report_directory = (
        Path(args.report_dir)
        if args.report_dir is not None
        else output_path.with_suffix("").with_name(output_path.stem + "_report")
    )
    artifacts = write_symbolic_report(
        loaded.model,
        report_directory,
        variables=_variables(loaded.model),
        history=loaded.history,
        metadata={
            **loaded.metadata,
            "pruning": {"units": unit_report, "edges": edge_report},
        },
    )
    print(
        json.dumps(
            {
                "checkpoint": str(output_checkpoint),
                "unit_pruning": unit_report,
                "edge_pruning": edge_report,
                "artifacts": _stringify_paths(artifacts),
            },
            indent=2,
        )
    )
    return 0


def command_plot(args: argparse.Namespace) -> int:
    loaded = load_checkpoint(args.checkpoint, device=args.device)
    output = write_structure_svg(loaded.model, args.output, title=args.title)
    print(json.dumps({"structure_svg": str(output)}, indent=2))
    return 0


def command_export(args: argparse.Namespace) -> int:
    loaded = load_checkpoint(args.checkpoint, device=args.device)
    artifacts = write_symbolic_report(
        loaded.model,
        args.output,
        variables=_variables(loaded.model),
        history=loaded.history,
        metadata=loaded.metadata,
        top_k=args.top_k,
        complexity_weight=args.complexity_weight,
    )
    print(json.dumps({"artifacts": _stringify_paths(artifacts)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symkan",
        description="Unofficial attributed Symbolic-KAN research package",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="show version and upstream attribution")
    info.set_defaults(function=command_info)

    smoke = subparsers.add_parser(
        "smoke", help="validate model construction and hardening"
    )
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--points", type=int, default=16)
    smoke.set_defaults(function=command_smoke)

    fit_demo = subparsers.add_parser(
        "fit-demo", help="run a small supervised x-squared demo"
    )
    fit_demo.add_argument("--config", required=True)
    fit_demo.add_argument("--output", required=True)
    fit_demo.add_argument("--train-points", type=int, default=64)
    fit_demo.add_argument("--validation-points", type=int, default=128)
    fit_demo.set_defaults(function=command_fit_demo)

    inspect = subparsers.add_parser(
        "inspect", help="inspect a trusted versioned checkpoint"
    )
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--device", default="cpu")
    inspect.add_argument("--output")
    inspect.add_argument("--top-k", type=int, default=3)
    inspect.add_argument("--complexity-weight", type=float, default=0.02)
    inspect.set_defaults(function=command_inspect)

    prune = subparsers.add_parser(
        "prune", help="deterministically prune and harden a trusted checkpoint"
    )
    prune.add_argument("--checkpoint", required=True)
    prune.add_argument("--output", required=True)
    prune.add_argument("--report-dir")
    prune.add_argument("--device", default="cpu")
    prune.add_argument("--unit-threshold", type=float, default=0.5)
    prune.add_argument("--edge-min-confidence", type=float, default=0.0)
    prune.set_defaults(function=command_prune)

    plot = subparsers.add_parser(
        "plot", help="render selected structure from a checkpoint"
    )
    plot.add_argument("--checkpoint", required=True)
    plot.add_argument("--output", required=True)
    plot.add_argument("--device", default="cpu")
    plot.add_argument("--title", default="Symbolic-KAN selected structure")
    plot.set_defaults(function=command_plot)

    export = subparsers.add_parser(
        "export", help="export JSON, expression, SVG, and HTML audit artifacts"
    )
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--device", default="cpu")
    export.add_argument("--top-k", type=int, default=3)
    export.add_argument("--complexity-weight", type=float, default=0.02)
    export.set_defaults(function=command_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())

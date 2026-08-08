# SPDX-License-Identifier: MIT
# Derived research software; see NOTICE.md for upstream attribution.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import __version__
from .config import load_experiment_config
from .export import export_expression, export_structure
from .model import SymbolicKAN
from .reproducibility import resolve_dtype
from .training import fit_supervised

UPSTREAM = "https://github.com/sfaroughi3/Pub_Symbolic_KANs"
UPSTREAM_COMMIT = "9481a822e73e5a7520c6c0a425a8a402f2878c03"


def _model_from_config(path: str) -> tuple[SymbolicKAN, object]:
    experiment = load_experiment_config(path)
    return SymbolicKAN(experiment.model, initial_temperature=experiment.training.tau_start), experiment


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
    x = torch.linspace(float(domain[0]), float(domain[1]), args.points, dtype=dtype).reshape(-1, 1)
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
        "expression": export_expression(model, variables=["x"] * experiment.model.input_dim),
    }
    print(json.dumps(report, indent=2))
    return 0 if deterministic_delta == 0.0 and report["hardened_output_finite"] else 1


def command_fit_demo(args: argparse.Namespace) -> int:
    model, experiment = _model_from_config(args.config)
    dtype = resolve_dtype(experiment.model.dtype)
    generator = torch.Generator().manual_seed(experiment.training.seed)
    x_train = 2.0 * torch.rand(args.train_points, 1, generator=generator, dtype=dtype) - 1.0
    y_train = x_train.square()
    x_val = torch.linspace(-1.0, 1.0, args.validation_points, dtype=dtype).reshape(-1, 1)
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
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "expression.txt").write_text(
        export_expression(result.model, variables=["x"]), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_soft_validation_loss": result.best_soft_validation_loss,
                "best_hardened_validation_loss": result.best_hardened_validation_loss,
                "output": str(result.output_directory),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symkan",
        description="Unofficial attributed Symbolic-KAN research package",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="show version and upstream attribution")
    info.set_defaults(function=command_info)

    smoke = subparsers.add_parser("smoke", help="validate model construction and hardening")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--points", type=int, default=16)
    smoke.set_defaults(function=command_smoke)

    fit_demo = subparsers.add_parser("fit-demo", help="run a small supervised x-squared demo")
    fit_demo.add_argument("--config", required=True)
    fit_demo.add_argument("--output", required=True)
    fit_demo.add_argument("--train-points", type=int, default=64)
    fit_demo.add_argument("--validation-points", type=int, default=128)
    fit_demo.set_defaults(function=command_fit_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())

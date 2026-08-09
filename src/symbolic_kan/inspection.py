# SPDX-License-Identifier: MIT
# Derived research software; see NOTICE.md for upstream attribution.

"""Structure inspection, deterministic pruning, and audit-report helpers."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import torch

from .export import export_expression, export_structure
from .model import SymbolicKAN

UPSTREAM_REPOSITORY = "https://github.com/sfaroughi3/Pub_Symbolic_KANs"
UPSTREAM_COMMIT = "9481a822e73e5a7520c6c0a425a8a402f2878c03"

_PRIMITIVE_COMPLEXITY = {
    "zero": 0.0,
    "const": 0.5,
    "id": 1.0,
    "x": 1.0,
    "abs": 1.5,
    "x2": 1.5,
    "sqrtx": 2.0,
    "sin": 2.0,
    "cos": 2.0,
    "exp": 2.0,
    "log": 2.0,
    "inv": 2.0,
    "x3": 2.0,
    "x4": 2.5,
    "x5": 3.0,
    "tanh": 2.5,
    "sigmoid": 2.5,
    "gauss": 2.5,
    "lorentz": 2.5,
}

_PRIMITIVE_COLORS = {
    "x": "#2783DE",
    "id": "#2783DE",
    "x2": "#5E9FE8",
    "x3": "#4FB9C9",
    "sin": "#46A171",
    "cos": "#72BC8F",
    "exp": "#D5803B",
    "log": "#DE9255",
    "inv": "#E56458",
}


@dataclass(frozen=True, slots=True)
class PrimitiveCandidate:
    """One native primitive candidate ranked from deterministic gate evidence."""

    block: int
    unit: int
    edge: int
    rank: int
    primitive: str
    probability: float
    complexity: float
    score: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def _package_version() -> str:
    try:
        return version("symbolic-kan-reproducible")
    except PackageNotFoundError:
        return "development"


def primitive_complexity(name: str) -> float:
    """Return a transparent, documented complexity prior for one primitive."""

    return float(_PRIMITIVE_COMPLEXITY.get(name, 3.0))


@torch.no_grad()
def rank_primitive_candidates(
    model: SymbolicKAN,
    *,
    top_k: int = 3,
    complexity_weight: float = 0.02,
) -> list[PrimitiveCandidate]:
    """Rank every edge's native primitives without claiming post-hoc fit quality."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if complexity_weight < 0:
        raise ValueError("complexity_weight cannot be negative")

    ranked: list[PrimitiveCandidate] = []
    for block_index, block in enumerate(model.blocks):
        for unit_index, unit_edges in enumerate(block.edges):
            for edge_index, edge in enumerate(unit_edges):
                probabilities = edge.deterministic_probabilities(hard=False)
                scores = [
                    float(probabilities[index].item())
                    - complexity_weight * primitive_complexity(name)
                    for index, name in enumerate(edge.primitive_names)
                ]
                ordering = sorted(
                    range(len(edge.primitive_names)),
                    key=lambda index: (
                        -scores[index],
                        -float(probabilities[index].item()),
                        edge.primitive_names[index],
                    ),
                )[:top_k]
                for rank, primitive_index in enumerate(ordering, start=1):
                    name = edge.primitive_names[primitive_index]
                    ranked.append(
                        PrimitiveCandidate(
                            block=block_index,
                            unit=unit_index,
                            edge=edge_index,
                            rank=rank,
                            primitive=name,
                            probability=float(probabilities[primitive_index].item()),
                            complexity=primitive_complexity(name),
                            score=float(scores[primitive_index]),
                        )
                    )
    return ranked


@torch.no_grad()
def structure_diagnostics(
    model: SymbolicKAN,
    *,
    top_k: int = 3,
    complexity_weight: float = 0.02,
) -> dict[str, Any]:
    """Return JSON-ready deterministic gate, edge, and unit diagnostics."""

    candidate_lookup: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for candidate in rank_primitive_candidates(
        model, top_k=top_k, complexity_weight=complexity_weight
    ):
        candidate_lookup.setdefault(
            (candidate.block, candidate.unit, candidate.edge), []
        ).append(candidate.to_dict())

    blocks: list[dict[str, Any]] = []
    for block_index, block in enumerate(model.blocks):
        units: list[dict[str, Any]] = []
        unit_probabilities = (
            torch.ones(block.hidden_units, device=block.projection_weight.device)
            if block.unit_logits is None
            else torch.sigmoid(block.unit_logits)
        )
        for unit_index, unit_edges in enumerate(block.edges):
            confidences = torch.stack(
                [edge.deterministic_probabilities(hard=False).max() for edge in unit_edges]
            )
            if bool(block.edge_mask_active.item()):
                edge_probabilities = block.edge_mask[unit_index]
            else:
                edge_probabilities = torch.softmax(
                    confidences / block.edge_temperature, dim=-1
                )
            edges: list[dict[str, Any]] = []
            for edge_index, _edge in enumerate(unit_edges):
                edges.append(
                    {
                        "edge": edge_index,
                        "confidence": float(confidences[edge_index].item()),
                        "selection_probability": float(
                            edge_probabilities[edge_index].item()
                        ),
                        "candidates": candidate_lookup[
                            (block_index, unit_index, edge_index)
                        ],
                    }
                )
            units.append(
                {
                    "unit": unit_index,
                    "unit_probability": float(unit_probabilities[unit_index].item()),
                    "selected_edge": int(edge_probabilities.argmax().item()),
                    "edges": edges,
                }
            )
        blocks.append({"block": block_index, "units": units})
    return {
        "top_k": top_k,
        "complexity_weight": complexity_weight,
        "blocks": blocks,
    }


@torch.no_grad()
def prune_units(
    model: SymbolicKAN,
    *,
    threshold: float = 0.5,
    snap_strength: float = 8.0,
    freeze: bool = True,
) -> dict[str, Any]:
    """Deterministically remove low-probability units while retaining one per block."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if snap_strength <= 0:
        raise ValueError("snap_strength must be positive")

    blocks: list[dict[str, Any]] = []
    for block_index, block in enumerate(model.blocks):
        if block.unit_logits is None:
            blocks.append(
                {
                    "block": block_index,
                    "unit_gates": False,
                    "alive_units": list(range(block.hidden_units)),
                    "pruned_units": [],
                }
            )
            continue
        probabilities = torch.sigmoid(block.unit_logits)
        alive = probabilities >= threshold
        if not bool(alive.any().item()):
            alive[probabilities.argmax()] = True
        block.unit_logits.copy_(
            torch.where(
                alive,
                torch.full_like(block.unit_logits, snap_strength),
                torch.full_like(block.unit_logits, -snap_strength),
            )
        )
        block.unit_gates_hardened.fill_(True)
        if freeze:
            block.unit_logits.requires_grad_(False)
        blocks.append(
            {
                "block": block_index,
                "unit_gates": True,
                "probabilities_before": probabilities.detach().cpu().tolist(),
                "alive_units": alive.nonzero(as_tuple=False).reshape(-1).cpu().tolist(),
                "pruned_units": (~alive)
                .nonzero(as_tuple=False)
                .reshape(-1)
                .cpu()
                .tolist(),
            }
        )
    return {"threshold": threshold, "blocks": blocks}


@torch.no_grad()
def prune_edges(
    model: SymbolicKAN,
    *,
    min_confidence: float = 0.0,
    unit_threshold: float | None = 0.5,
    snap_strength: float = 8.0,
    freeze: bool = True,
) -> dict[str, Any]:
    """Select one edge per unit, harden native primitives, and record uncertainty."""

    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    if snap_strength <= 0:
        raise ValueError("snap_strength must be positive")

    blocks: list[dict[str, Any]] = []
    for block_index, block in enumerate(model.blocks):
        mask = torch.zeros_like(block.edge_mask)
        units: list[dict[str, Any]] = []
        for unit_index, unit_edges in enumerate(block.edges):
            confidences = torch.stack(
                [edge.deterministic_probabilities(hard=False).max() for edge in unit_edges]
            )
            winner = int(confidences.argmax().item())
            selected_confidence = float(confidences[winner].item())
            mask[unit_index, winner] = 1.0
            for edge in unit_edges:
                edge.harden(snap_strength=snap_strength, freeze=freeze)
            units.append(
                {
                    "unit": unit_index,
                    "selected_edge": winner,
                    "confidence": selected_confidence,
                    "below_min_confidence": selected_confidence < min_confidence,
                }
            )
        block.edge_mask.copy_(mask)
        block.edge_mask_active.fill_(True)
        blocks.append({"block": block_index, "units": units})

    unit_report = (
        None
        if unit_threshold is None
        else prune_units(
            model,
            threshold=unit_threshold,
            snap_strength=snap_strength,
            freeze=freeze,
        )
    )
    return {
        "min_confidence": min_confidence,
        "blocks": blocks,
        "unit_pruning": unit_report,
    }


def _vertical_positions(count: int, height: int) -> list[float]:
    if count <= 1:
        return [height / 2]
    top = 110.0
    bottom = height - 72.0
    return [top + index * (bottom - top) / (count - 1) for index in range(count)]


def write_structure_svg(
    model: SymbolicKAN,
    path: str | Path,
    *,
    title: str = "Symbolic-KAN selected structure",
) -> Path:
    """Write a dependency-free SVG for the deterministic selected structure."""

    structure = export_structure(model)
    level_count = len(structure["blocks"]) + 2
    max_nodes = max(
        [model.config.input_dim]
        + [len(block["units"]) for block in structure["blocks"]]
        + [1]
    )
    width = max(960, 210 * level_count)
    height = max(420, 88 * max_nodes + 150)
    x_positions = [
        80 + index * (width - 160) / (level_count - 1)
        for index in range(level_count)
    ]
    input_positions = _vertical_positions(model.config.input_dim, height)
    block_positions = [
        _vertical_positions(len(block["units"]), height)
        for block in structure["blocks"]
    ]
    output_y = height / 2

    lines: list[str] = []
    nodes: list[str] = []
    previous_y = input_positions
    for block_index, block_data in enumerate(structure["blocks"]):
        current_y = block_positions[block_index]
        source_x = x_positions[block_index]
        target_x = x_positions[block_index + 1]
        for unit_index, unit in enumerate(block_data["units"]):
            color = _PRIMITIVE_COLORS.get(unit["primitive"], "#7D7A75")
            weights = unit["projection_weight"]
            scale = max([abs(float(weight)) for weight in weights] + [1e-12])
            for source_index, weight in enumerate(weights[: len(previous_y)]):
                magnitude = abs(float(weight)) / scale
                if magnitude < 1e-6:
                    continue
                lines.append(
                    f'<path d="M {source_x + 22:.1f} {previous_y[source_index]:.1f} '
                    f'C {source_x + 84:.1f} {previous_y[source_index]:.1f}, '
                    f'{target_x - 84:.1f} {current_y[unit_index]:.1f}, '
                    f'{target_x - 34:.1f} {current_y[unit_index]:.1f}" '
                    f'stroke="{color}" stroke-width="{1.0 + 2.8 * magnitude:.2f}" '
                    f'opacity="{0.22 + 0.68 * magnitude:.2f}" fill="none"/>'
                )
        previous_y = current_y

    source_x = x_positions[-2]
    target_x = x_positions[-1]
    for y_value in previous_y:
        lines.append(
            f'<path d="M {source_x + 34:.1f} {y_value:.1f} '
            f'C {source_x + 90:.1f} {y_value:.1f}, {target_x - 90:.1f} '
            f'{output_y:.1f}, {target_x - 30:.1f} {output_y:.1f}" '
            'stroke="#B9D7F1" stroke-width="1.8" fill="none"/>'
        )

    for input_index, y_value in enumerate(input_positions):
        nodes.append(
            f'<circle cx="{x_positions[0]:.1f}" cy="{y_value:.1f}" r="22" '
            'fill="#E5F2FC" stroke="#2783DE" stroke-width="2"/>'
            f'<text x="{x_positions[0]:.1f}" y="{y_value + 5:.1f}" '
            f'text-anchor="middle" class="node-label">x{input_index}</text>'
        )
    for block_index, block_data in enumerate(structure["blocks"]):
        x_value = x_positions[block_index + 1]
        for unit_index, unit in enumerate(block_data["units"]):
            y_value = block_positions[block_index][unit_index]
            color = _PRIMITIVE_COLORS.get(unit["primitive"], "#7D7A75")
            opacity = "1" if unit["alive"] else "0.35"
            primitive = html.escape(str(unit["primitive"]))
            nodes.append(
                f'<g opacity="{opacity}"><rect x="{x_value - 42:.1f}" '
                f'y="{y_value - 27:.1f}" width="84" height="54" rx="11" '
                f'fill="#FFFFFF" stroke="{color}" stroke-width="2"/>'
                f'<text x="{x_value:.1f}" y="{y_value - 2:.1f}" '
                f'text-anchor="middle" class="primitive">{primitive}</text>'
                f'<text x="{x_value:.1f}" y="{y_value + 16:.1f}" '
                f'text-anchor="middle" class="small">b{block_index} · u{unit_index}'
                '</text></g>'
            )
    nodes.append(
        f'<rect x="{x_positions[-1] - 30:.1f}" y="{output_y - 30:.1f}" '
        'width="60" height="60" rx="14" fill="#2783DE"/>'
        f'<text x="{x_positions[-1]:.1f}" y="{output_y + 6:.1f}" '
        'text-anchor="middle" class="output">f(x)</text>'
    )

    subtitle = (
        f"{model.config.num_blocks} blocks · {model.config.hidden_units} units/block · "
        f"{'hardened' if model.is_hardened else 'soft selection'}"
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title>
<desc id="desc">Deterministic Symbolic-KAN structure with selected primitives and weighted projections.</desc>
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; fill: #2C2C2B; }}
.node-label {{ font-size: 14px; font-weight: 700; fill: #1D6FB8; }}
.primitive {{ font-size: 15px; font-weight: 700; }}
.small {{ font-size: 11px; fill: #7D7A75; }}
.output {{ font-size: 16px; font-weight: 700; fill: #FFFFFF; }}
</style>
<rect width="{width}" height="{height}" rx="18" fill="#FFFFFF"/>
<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="17" fill="none" stroke="#E6E5E3"/>
<text x="36" y="42" font-size="24" font-weight="750">{html.escape(title)}</text>
<text x="36" y="68" font-size="13" fill="#7D7A75">{html.escape(subtitle)}</text>
{''.join(lines)}
{''.join(nodes)}
<text x="36" y="{height - 24}" font-size="11" fill="#7D7A75">Unofficial derivative · upstream 9481a82 · deterministic selected structure</text>
</svg>
'''
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


def build_symbolic_report(
    model: SymbolicKAN,
    *,
    variables: Sequence[str] | None = None,
    history: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    top_k: int = 3,
    complexity_weight: float = 0.02,
) -> dict[str, Any]:
    """Build a machine-readable audit report for one selected structure."""

    structure = export_structure(model)
    active_units = sum(
        int(unit["alive"])
        for block in structure["blocks"]
        for unit in block["units"]
    )
    return {
        "schema_version": 1,
        "package": "symbolic-kan-reproducible",
        "package_version": _package_version(),
        "status": "unofficial derivative",
        "provenance": {
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
        },
        "summary": {
            "hardened": model.is_hardened,
            "blocks": len(model.blocks),
            "active_units": active_units,
        },
        "structure": structure,
        "diagnostics": structure_diagnostics(
            model, top_k=top_k, complexity_weight=complexity_weight
        ),
        "expression": export_expression(
            model, variables=None if variables is None else list(variables)
        ),
        "history": history or [],
        "metadata": metadata or {},
    }


def _report_html(report: dict[str, Any]) -> str:
    rows: list[str] = []
    for block in report["structure"]["blocks"]:
        for unit in block["units"]:
            rows.append(
                "<tr>"
                f"<td>{block['block']}</td><td>{unit['unit']}</td>"
                f"<td>{html.escape(str(unit['primitive']))}</td>"
                f"<td>{unit['edge']}</td>"
                f"<td>{'active' if unit['alive'] else 'pruned'}</td>"
                "</tr>"
            )
    expression = html.escape(str(report["expression"]))
    metadata = html.escape(json.dumps(report["metadata"], indent=2, default=str))
    return f'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Symbolic-KAN audit report</title>
<style>
:root {{ --text:#2C2C2B; --muted:#7D7A75; --border:#E6E5E3; --soft:#F9F8F7; --blue:#2783DE; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#fff; color:var(--text); font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:48px 32px 64px; }}
.eyebrow {{ color:#1D6FB8; font-size:13px; font-weight:750; letter-spacing:.1em; }}
h1 {{ margin:8px 0 4px; font-size:40px; line-height:1.15; }} .muted {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:32px 0; }}
.card {{ border:1px solid var(--border); border-radius:12px; padding:18px; background:var(--soft); }}
.value {{ display:block; color:var(--blue); font-size:28px; font-weight:750; }}
section {{ margin-top:36px; }} img {{ width:100%; border:1px solid var(--border); border-radius:12px; }}
pre {{ overflow:auto; padding:18px; border:1px solid var(--border); border-radius:10px; background:var(--soft); }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; }} th {{ color:var(--muted); font-size:13px; }}
.notice {{ margin-top:32px; padding:16px; border-left:4px solid #D5803B; background:#FBEBDE; }}
@media (max-width:700px) {{ main {{ padding:28px 18px; }} .grid {{ grid-template-columns:1fr; }} h1 {{ font-size:32px; }} }}
</style></head><body><main>
<div class="eyebrow">UNOFFICIAL · ATTRIBUTED · AUDITABLE</div>
<h1>Symbolic-KAN audit report</h1>
<p class="muted">Deterministic structure, native candidate evidence, and export metadata.</p>
<div class="grid">
<div class="card"><span class="value">{report['summary']['blocks']}</span>blocks</div>
<div class="card"><span class="value">{report['summary']['active_units']}</span>active units</div>
<div class="card"><span class="value">{'yes' if report['summary']['hardened'] else 'no'}</span>hardened</div>
</div>
<section><h2>Selected structure</h2><img src="structure.svg" alt="Selected Symbolic-KAN structure"></section>
<section><h2>Expression</h2><pre>{expression}</pre></section>
<section><h2>Selected units</h2><table><thead><tr><th>Block</th><th>Unit</th><th>Primitive</th><th>Edge</th><th>Status</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section><h2>Metadata</h2><pre>{metadata}</pre></section>
<div class="notice"><strong>Scientific-integrity boundary.</strong> This report was produced by an unofficial derivative package. It is not an upstream paper result and does not imply endorsement by the original authors.</div>
</main></body></html>
'''


def write_symbolic_report(
    model: SymbolicKAN,
    directory: str | Path,
    *,
    variables: Sequence[str] | None = None,
    history: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    top_k: int = 3,
    complexity_weight: float = 0.02,
) -> dict[str, Path]:
    """Write JSON, text, SVG, and HTML audit artifacts."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    report = build_symbolic_report(
        model,
        variables=variables,
        history=history,
        metadata=metadata,
        top_k=top_k,
        complexity_weight=complexity_weight,
    )
    paths = {
        "report": output / "symbolic_report.json",
        "expression": output / "expression.txt",
        "structure": output / "structure.svg",
        "html": output / "report.html",
    }
    paths["report"].write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    paths["expression"].write_text(str(report["expression"]), encoding="utf-8")
    write_structure_svg(model, paths["structure"])
    paths["html"].write_text(_report_html(report), encoding="utf-8")
    return paths

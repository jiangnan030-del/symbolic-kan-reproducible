# SPDX-License-Identifier: MIT
# Derived research software; see NOTICE.md for upstream attribution.

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .model import FixedSumReadout, GatedPrimitiveEdge, SymbolicKAN, SymbolicKANBlock
from .primitives import primitive_expression


def _number(value: torch.Tensor | float, precision: int = 6) -> str:
    scalar = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
    if abs(scalar) < 10 ** (-(precision + 1)):
        scalar = 0.0
    return f"{scalar:.{precision}g}"


def _selected_edge(block: SymbolicKANBlock, unit_index: int) -> int:
    if bool(block.edge_mask_active.item()):
        return int(block.edge_mask[unit_index].argmax().item())
    confidence = torch.stack(
        [edge.deterministic_probabilities().max() for edge in block.edges[unit_index]]
    )
    return int(confidence.argmax().item())


def _selected_primitive(edge: GatedPrimitiveEdge) -> int:
    return int(edge.logits.argmax().item())


def export_structure(model: SymbolicKAN) -> dict[str, Any]:
    """Return the selected hierarchy and continuous parameters as JSON-ready data."""

    blocks: list[dict[str, Any]] = []
    for block_index, block in enumerate(model.blocks):
        units: list[dict[str, Any]] = []
        for unit_index in range(block.hidden_units):
            edge_index = _selected_edge(block, unit_index)
            edge = block.edges[unit_index][edge_index]
            primitive_index = _selected_primitive(edge)
            primitive = edge.primitives[primitive_index]
            alive = True
            if block.unit_logits is not None:
                alive = bool((torch.sigmoid(block.unit_logits[unit_index]) > 0.5).item())
            units.append(
                {
                    "unit": unit_index,
                    "alive": alive,
                    "edge": edge_index,
                    "primitive": edge.primitive_names[primitive_index],
                    "projection_weight": block.projection_weight[
                        unit_index, edge_index
                    ].detach().cpu().tolist(),
                    "projection_bias": float(
                        block.projection_bias[unit_index, edge_index].detach().cpu().item()
                    ),
                    "gamma": float(primitive.gamma.detach().cpu().item()),
                    "beta": float(primitive.beta.detach().cpu().item()),
                    "amplitude": float(primitive.amplitude.detach().cpu().item()),
                    "bias": float(primitive.bias.detach().cpu().item()),
                }
            )
        blocks.append({"block": block_index, "residual": block.residual, "units": units})

    readout: dict[str, Any]
    if isinstance(model.readout, FixedSumReadout):
        readout = {"type": "fixed_sum"}
    elif isinstance(model.readout, nn.Linear):
        readout = {
            "type": "trainable_linear",
            "weight": model.readout.weight.detach().cpu().reshape(-1).tolist(),
            "bias": None
            if model.readout.bias is None
            else float(model.readout.bias.detach().cpu().item()),
        }
    else:
        raise TypeError(f"unsupported readout type: {type(model.readout)!r}")

    return {
        "package": "symbolic-kan-reproducible",
        "upstream_commit": "9481a822e73e5a7520c6c0a425a8a402f2878c03",
        "hardened": model.is_hardened,
        "blocks": blocks,
        "readout": readout,
        "inverse_parameters": [float(value.detach().cpu().item()) for value in model.inverse_parameters()],
    }


def _linear_expression(weights: torch.Tensor, variables: list[str], bias: torch.Tensor) -> str:
    terms = [f"{_number(weight)}*({variable})" for weight, variable in zip(weights, variables)]
    if float(bias.detach().cpu().item()) != 0.0:
        terms.append(_number(bias))
    return " + ".join(terms) if terms else "0"


def export_expression(
    model: SymbolicKAN,
    *,
    variables: list[str] | None = None,
    latex: bool = False,
) -> str:
    """Export a hierarchical expression for the currently selected structure.

    Export is intended for auditability. Deep models naturally produce long expressions;
    callers may prefer :func:`export_structure` for machine-readable analysis.
    """

    if variables is None:
        variables = [f"x_{index}" for index in range(model.config.input_dim)]
    if len(variables) != model.config.input_dim:
        raise ValueError("variable count must equal model input_dim")

    previous = list(variables)
    for block_index, block in enumerate(model.blocks):
        current: list[str] = []
        for unit_index in range(block.hidden_units):
            if block.unit_logits is not None and not bool(
                (torch.sigmoid(block.unit_logits[unit_index]) > 0.5).item()
            ):
                expression = "0"
            else:
                edge_index = _selected_edge(block, unit_index)
                edge = block.edges[unit_index][edge_index]
                primitive_index = _selected_primitive(edge)
                primitive = edge.primitives[primitive_index]
                projection = _linear_expression(
                    block.projection_weight[unit_index, edge_index],
                    previous,
                    block.projection_bias[unit_index, edge_index],
                )
                argument = (
                    f"{_number(primitive.gamma)}*({projection})"
                    f" + {_number(primitive.beta)}"
                )
                base = primitive_expression(edge.primitive_names[primitive_index], argument, latex=latex)
                expression = (
                    f"{_number(primitive.amplitude)}*({base})"
                    f" + {_number(primitive.bias)}"
                )
            if block.residual and len(previous) == block.hidden_units:
                expression = f"({previous[unit_index]}) + ({expression})"
            current.append(f"({expression})")
            if latex:
                current[-1] = f"({expression})"
        previous = current

    if isinstance(model.readout, FixedSumReadout):
        output = " + ".join(previous)
    elif isinstance(model.readout, nn.Linear):
        output = _linear_expression(
            model.readout.weight.detach().reshape(-1),
            previous,
            model.readout.bias.detach()
            if model.readout.bias is not None
            else model.readout.weight.new_zeros(()),
        )
    else:
        raise TypeError(f"unsupported readout type: {type(model.readout)!r}")
    return output

# SPDX-License-Identifier: MIT
# Primitive set derived from sfaroughi3/Pub_Symbolic_KANs at 9481a82.
# See NOTICE.md for full authorship, license, and non-endorsement information.

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-6


def _safe_signed_denominator(z: torch.Tensor, eps: float) -> torch.Tensor:
    # torch.sign(0) is zero; defining zero as positive avoids the upstream 1/0 case.
    direction = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    return z + eps * direction


def apply_primitive(kind: str, z: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Evaluate one supported scalar primitive with finite-domain safeguards."""

    functions: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "id": lambda value: value,
        "x": lambda value: value,
        "zero": lambda value: torch.zeros_like(value),
        "const": lambda value: torch.ones_like(value),
        "x2": lambda value: value.square(),
        "x3": lambda value: value.pow(3),
        "x4": lambda value: value.pow(4),
        "x5": lambda value: value.pow(5),
        "sqrtx": lambda value: torch.sqrt(value.abs() + eps),
        "abs": torch.abs,
        "sign": torch.sign,
        "relu": torch.relu,
        "softplus": F.softplus,
        "sigmoid": torch.sigmoid,
        "swish": lambda value: value * torch.sigmoid(value),
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": lambda value: torch.tan(value.clamp(-1.4, 1.4)),
        "tanh": torch.tanh,
        "sinh": lambda value: torch.sinh(value.clamp(-4.0, 4.0)),
        "cosh": lambda value: torch.cosh(value.clamp(-4.0, 4.0)),
        "sech": lambda value: torch.cosh(value.clamp(-4.0, 4.0)).reciprocal(),
        "exp": lambda value: torch.exp(value.clamp(-6.0, 6.0)),
        "exp_m1": lambda value: torch.expm1(value.clamp(-6.0, 6.0)),
        "log": lambda value: torch.log(value.abs() + eps),
        "log1p": lambda value: torch.log1p(torch.sqrt(value.square() + eps)),
        "inv": lambda value: _safe_signed_denominator(value, eps).reciprocal(),
        "rsqrt": lambda value: torch.rsqrt(value.square() + eps),
        "gauss": lambda value: torch.exp(-value.square()),
        "lorentz": lambda value: (1.0 + value.square()).reciprocal(),
        "elu": F.elu,
        "softsign": lambda value: value / (1.0 + value.abs()),
    }
    try:
        return functions[kind](z)
    except KeyError as exc:
        raise ValueError(f"unknown primitive: {kind}") from exc


class SymbolicPrimitive(nn.Module):
    """Bounded affine-in/affine-out wrapper around one analytic primitive."""

    def __init__(self, kind: str, *, use_beta: bool = False, use_bias: bool = False) -> None:
        super().__init__()
        self.kind = kind
        self.gamma_raw = nn.Parameter(torch.tensor(0.7))
        self.amplitude_raw = nn.Parameter(torch.tensor(0.7))
        if use_beta:
            self.beta_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("beta_raw", torch.tensor(0.0))
        if use_bias:
            self.bias_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("bias_raw", torch.tensor(0.0))

    @property
    def gamma(self) -> torch.Tensor:
        return 3.0 * torch.tanh(self.gamma_raw)

    @property
    def amplitude(self) -> torch.Tensor:
        return 3.0 * torch.tanh(self.amplitude_raw)

    @property
    def beta(self) -> torch.Tensor:
        return torch.tanh(self.beta_raw)

    @property
    def bias(self) -> torch.Tensor:
        return torch.tanh(self.bias_raw)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        z = self.gamma * value + self.beta
        return self.amplitude * apply_primitive(self.kind, z) + self.bias


def primitive_expression(kind: str, argument: str, *, latex: bool = False) -> str:
    """Format a primitive around an already formatted argument."""

    if kind in {"id", "x"}:
        return argument
    if kind == "zero":
        return "0"
    if kind == "const":
        return "1"
    if kind in {"x2", "x3", "x4", "x5"}:
        exponent = kind[1:]
        return rf"\left({argument}\right)^{{{exponent}}}" if latex else f"({argument})^{exponent}"
    if kind == "sqrtx":
        return rf"\sqrt{{\left|{argument}\right|}}" if latex else f"sqrt(abs({argument}))"
    if kind == "abs":
        return rf"\left|{argument}\right|" if latex else f"abs({argument})"
    if kind == "sign":
        return rf"\operatorname{{sign}}\left({argument}\right)" if latex else f"sign({argument})"
    if kind == "relu":
        return rf"\max\left(0,{argument}\right)" if latex else f"max(0, {argument})"
    if kind == "softplus":
        return rf"\log\left(1+e^{{{argument}}}\right)" if latex else f"softplus({argument})"
    if kind == "sigmoid":
        return rf"\sigma\left({argument}\right)" if latex else f"sigmoid({argument})"
    if kind == "swish":
        return rf"{argument}\,\sigma\left({argument}\right)" if latex else f"({argument})*sigmoid({argument})"
    if kind in {"sin", "cos", "tan", "tanh", "sinh", "cosh"}:
        prefix = rf"\{kind}" if latex else kind
        return rf"{prefix}\left({argument}\right)" if latex else f"{prefix}({argument})"
    if kind == "sech":
        return rf"\operatorname{{sech}}\left({argument}\right)" if latex else f"sech({argument})"
    if kind == "exp":
        return rf"\exp\left({argument}\right)" if latex else f"exp({argument})"
    if kind == "exp_m1":
        return rf"\exp\left({argument}\right)-1" if latex else f"expm1({argument})"
    if kind == "log":
        return rf"\log\left(\left|{argument}\right|+\varepsilon\right)" if latex else f"log(abs({argument}) + eps)"
    if kind == "log1p":
        return rf"\log\left(1+\sqrt{{({argument})^2+\varepsilon}}\right)" if latex else f"log1p(sqrt(({argument})^2 + eps))"
    if kind == "inv":
        if latex:
            return "\\frac{1}{" + argument + "+\\varepsilon\\,\\operatorname{sign}_0(" + argument + ")}"
        return f"safe_inv({argument})"
    if kind == "rsqrt":
        if latex:
            return "\\frac{1}{\\sqrt{(" + argument + ")^2+\\varepsilon}}"
        return f"rsqrt(({argument})^2 + eps)"
    if kind == "gauss":
        return rf"\exp\left(-({argument})^2\right)" if latex else f"exp(-({argument})^2)"
    if kind == "lorentz":
        if latex:
            return "\\frac{1}{1+(" + argument + ")^2}"
        return f"1/(1 + ({argument})^2)"
    if kind in {"elu", "softsign"}:
        return rf"\operatorname{{{kind}}}\left({argument}\right)" if latex else f"{kind}({argument})"
    raise ValueError(f"unknown primitive: {kind}")

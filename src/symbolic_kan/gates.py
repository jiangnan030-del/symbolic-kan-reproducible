# SPDX-License-Identifier: MIT
# Derived research software; see NOTICE.md for upstream attribution.

from __future__ import annotations

import torch
import torch.nn.functional as F


def one_hot_argmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return a deterministic one-hot tensor at the maximum score."""

    indices = scores.argmax(dim=dim, keepdim=True)
    return torch.zeros_like(scores).scatter_(dim, indices, 1.0)


def straight_through_argmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """One-hot forward value with gradients from the supplied soft scores."""

    hard = one_hot_argmax(scores, dim=dim)
    return (hard - scores).detach() + scores


def primitive_probabilities(
    logits: torch.Tensor,
    tau: float,
    *,
    training: bool,
    hard_sample: bool,
    hardened: bool,
) -> torch.Tensor:
    """Select stochastic training or deterministic evaluation probabilities.

    Unlike the audited upstream code, evaluation never samples Gumbel noise.
    """

    if hardened:
        return one_hot_argmax(logits)
    if training:
        return F.gumbel_softmax(logits, tau=tau, hard=hard_sample, dim=-1)
    return torch.softmax(logits / tau, dim=-1)

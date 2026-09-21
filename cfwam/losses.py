"""Losses for cause attribution, localization and counterfactual explanation."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def dice_loss(logits: Tensor, target: Tensor, valid: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    probability = logits.sigmoid()
    if valid is not None:
        probability, target = probability * valid, target * valid
    numerator = 2 * (probability * target).sum(dim=-1) + eps
    denominator = probability.sum(dim=-1) + target.sum(dim=-1) + eps
    return (1 - numerator / denominator).mean()


def counterfactual_loss(
    cause_logits: Tensor,
    mask_logits: Tensor,
    hypothesis_residuals: Tensor,
    residual_target: Tensor,
    cause_target: Tensor,
    mask_target: Tensor,
    class_weights: Tensor | None = None,
    node_valid: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return the preregistered 1 : 1 : 0.5 objective components."""
    cause = F.cross_entropy(cause_logits, cause_target, weight=class_weights)
    pointwise = F.binary_cross_entropy_with_logits(mask_logits, mask_target, reduction="none")
    if node_valid is None:
        bce = pointwise.mean()
    else:
        bce = (pointwise * node_valid).sum() / node_valid.sum().clamp_min(1.0)
    mask = bce + dice_loss(mask_logits, mask_target, node_valid)
    selected = hypothesis_residuals[torch.arange(cause_target.shape[0], device=cause_target.device), cause_target]
    explanation = F.huber_loss(selected, residual_target)
    total = cause + mask + 0.5 * explanation
    return {"total": total, "cause": cause, "mask": mask, "counterfactual": explanation}

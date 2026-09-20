"""Metrics for attribution, localization and control; all return auditable scalars."""

from collections import Counter

import torch


def macro_f1(predicted: torch.Tensor, target: torch.Tensor, classes: int = 5) -> float:
    scores = []
    for label in range(classes):
        p, t = predicted == label, target == label
        tp, denom = (p & t).sum().item(), (p.sum() + t.sum()).item()
        scores.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return sum(scores) / classes


def mask_iou(predicted: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    p, t = predicted >= threshold, target.bool()
    union = (p | t).sum().item()
    return 1.0 if union == 0 else (p & t).sum().item() / union


def control_summary(records: list[dict]) -> dict[str, float]:
    """Summarise explicitly logged outcomes; never infer them from policy inputs."""
    count, total = max(1, len(records)), Counter()
    for row in records:
        for key in ("terminal_success", "false_stop", "unnecessary_global_refresh", "safe_failure"):
            total[key] += bool(row.get(key, False))
        for key in ("wam_calls", "recovery_steps", "latency_ms"):
            total[key] += float(row.get(key, 0))
    return {key: value / count for key, value in total.items()}

"""Cross-task supervised contrastive objective for mismatch factors."""

from __future__ import annotations

def cross_task_factor_contrastive_loss(
    embeddings,
    factor_targets,
    task_ids,
    *,
    temperature: float = 0.1,
):
    """Pull together equal factor signatures from different tasks.

    Same-task pairs are never positives, so the loss cannot be satisfied by
    memorising a task appearance. Pairs with different supervised factor
    signatures form the negative set. Rows without a cross-task positive are
    omitted instead of receiving fabricated supervision.
    """
    import torch
    import torch.nn.functional as F

    if embeddings.ndim != 2 or factor_targets.ndim != 2 or task_ids.ndim != 1:
        raise ValueError("expected embeddings[N,D], factor_targets[N,F], task_ids[N]")
    if len(embeddings) != len(factor_targets) or len(embeddings) != len(task_ids):
        raise ValueError("contrastive inputs must have the same row count")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")
    z = F.normalize(embeddings, dim=1)
    logits = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    same_signature = (factor_targets[:, None, :] == factor_targets[None, :, :]).all(-1)
    cross_task = task_ids[:, None] != task_ids[None, :]
    positives = same_signature & cross_task & ~eye
    candidates = ~eye
    valid = positives.any(1)
    if not bool(valid.any()):
        return embeddings.sum() * 0.0
    masked_logits = logits.masked_fill(~candidates, float("-inf"))
    log_prob = masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
    positive_log_prob = (log_prob.masked_fill(~positives, 0.0).sum(1)
                         / positives.sum(1).clamp(min=1))
    return -positive_log_prob[valid].mean()

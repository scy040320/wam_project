"""Bridge learned outputs to a transparent recovery decision without simulator GT."""

import torch

from .recovery import RecoveryDecision, RecoveryRouter
from .runtime import AbstainPolicy
from .types import AttributionResult, Cause


CAUSE_ORDER = (Cause.NORMAL, Cause.VISUAL_OCCLUSION, Cause.OBJECT_SHIFT, Cause.ACTION_NOISE, Cause.UNKNOWN)


def decode_attribution(outputs: dict[str, torch.Tensor], residual: torch.Tensor,
                       node_ids: list[str], abstain: AbstainPolicy,
                       mask_threshold: float = 0.5) -> AttributionResult:
    """Decode one batch item. No field in this function may be simulator truth."""
    probabilities = torch.softmax(outputs["cause_logits"][0], dim=-1)
    probability_map = {cause: float(probabilities[index]) for index, cause in enumerate(CAUSE_ORDER)}
    candidate = CAUSE_ORDER[int(probabilities.argmax())]
    energy = torch.nn.functional.huber_loss(
        outputs["hypothesis_residuals"][0, int(probabilities.argmax())], residual[0], reduction="mean"
    ).item()
    result = abstain.apply(probability_map, candidate, energy)
    mask = torch.sigmoid(outputs["mask_logits"][0]) >= mask_threshold
    nodes = frozenset(node for node, selected in zip(node_ids, mask.tolist()) if selected)
    return AttributionResult(result.cause, probability_map, nodes, energy, result.entropy, result.abstained)


def decide_recovery(outputs: dict[str, torch.Tensor], residual: torch.Tensor, node_ids: list[str],
                    abstain: AbstainPolicy, router: RecoveryRouter, timestamp: int) -> RecoveryDecision:
    attribution = decode_attribution(outputs, residual, node_ids, abstain)
    return router.route(attribution, timestamp)

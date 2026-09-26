"""Calibration contracts and D17 tri-state output adaptation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .contracts import (
    AttributionOutput, CoarseCause, ConsistencyFactor, EvidenceQuality, TriValue,
)


HEAD_TO_FACTOR = {
    "observation_sufficient": ConsistencyFactor.OBSERVATION_RELIABLE,
    "world_state_consistent": ConsistencyFactor.WORLD_STATE_CONSISTENT,
    "execution_contact_consistent": ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT,
    "task_progress_consistent": ConsistencyFactor.TASK_STAGE_CONSISTENT,
    "cause_resolved": ConsistencyFactor.CAUSE_RESOLVED,
}


@dataclass(frozen=True)
class HeadThreshold:
    threshold: float
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0 or self.temperature <= 0.0:
            raise ValueError("invalid calibration threshold or temperature")


def _tri_state(distribution: Mapping[str, float], threshold: float) -> tuple[TriValue, float]:
    if set(distribution) != {"no", "yes", "uncertain"}:
        raise ValueError("tri-state distribution must contain no/yes/uncertain")
    confidence = max(float(value) for value in distribution.values())
    label = max(distribution, key=distribution.__getitem__)
    if confidence < threshold:
        return TriValue.UNKNOWN, confidence
    return {"no": TriValue.FALSE, "yes": TriValue.TRUE, "uncertain": TriValue.UNKNOWN}[label], confidence


def build_attribution_output(
    *, coarse_probs: Mapping[str, float], factor_distributions: Mapping[str, Mapping[str, float]],
    thresholds: Mapping[str, HeadThreshold], source_block_id: str,
) -> AttributionOutput:
    """Convert calibrated D17 heads into the simulator-independent D19 contract."""
    expected_classes = {item.value for item in CoarseCause}
    if set(coarse_probs) != expected_classes or not math.isclose(sum(coarse_probs.values()), 1.0, abs_tol=1e-5):
        raise ValueError("invalid coarse probability distribution")
    required_heads = set(HEAD_TO_FACTOR) | {
        "primary_view_reliable", "wrist_view_reliable", "action_record_reliable"
    }
    if set(factor_distributions) != required_heads or not required_heads.issubset(thresholds):
        raise ValueError("missing factor distribution or threshold")

    states: dict[str, TriValue] = {}
    confidences: dict[str, float] = {}
    scalar_probs: dict[str, float] = {}
    for head, factor in HEAD_TO_FACTOR.items():
        state, confidence = _tri_state(factor_distributions[head], thresholds[head].threshold)
        states[factor.value] = state
        confidences[factor.value] = confidence
        scalar_probs[factor.value] = float(factor_distributions[head]["yes"])

    evidence_states = {}
    for head in ("primary_view_reliable", "wrist_view_reliable", "action_record_reliable"):
        evidence_states[head] = _tri_state(factor_distributions[head], thresholds[head].threshold)
    primary = evidence_states["primary_view_reliable"]
    wrist = evidence_states["wrist_view_reliable"]
    execution = evidence_states["action_record_reliable"]
    conflict = (
        primary[1] >= thresholds["primary_view_reliable"].threshold
        and wrist[1] >= thresholds["wrist_view_reliable"].threshold
        and primary[0] is not TriValue.UNKNOWN and wrist[0] is not TriValue.UNKNOWN
        and primary[0] is not wrist[0]
    )

    coarse_name = max(coarse_probs, key=coarse_probs.__getitem__)
    coarse_confidence = float(coarse_probs[coarse_name])
    coarse_threshold = thresholds["coarse"].threshold
    cause_resolved = states[ConsistencyFactor.CAUSE_RESOLVED.value] is TriValue.TRUE
    projected = CoarseCause(coarse_name)
    if coarse_confidence < coarse_threshold or conflict or not cause_resolved:
        projected = CoarseCause.UNKNOWN
    entropy = -sum(float(p) * math.log(max(float(p), 1e-12)) for p in coarse_probs.values())
    return AttributionOutput(
        scalar_probs, coarse_probs, projected, coarse_confidence, entropy,
        EvidenceQuality(primary[0] is TriValue.TRUE, wrist[0] is TriValue.TRUE,
                        execution[0] is TriValue.TRUE, conflict),
        source_block_id, states, confidences,
    )

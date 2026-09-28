"""Frozen D19--D21 selector policies for paired downstream evaluation.

The first downstream pilot deliberately excludes the uncalibrated D20 soft
weights. Oracle and learned attribution use the same auditable policy: update
belief, hard-gate infeasible candidates, then use the official Cosmos value as
the tie-breaker among feasible candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from .belief import update_belief
from .candidate_effects import parse_candidate_effect
from .contracts import (
    AttributionOutput, BeliefFact, BeliefState, CandidateDecision, CandidateVisualEvidence,
    ConsistencyFactor, ScoreWeights, TriValue,
)
from .reranker import evaluate_candidate, select_candidate


class SelectorMode(str, Enum):
    VALUE_ONLY = "value_only"
    ORACLE_HARD_GATE = "oracle_hard_gate"
    LEARNED_HARD_GATE = "learned_hard_gate"


@dataclass(frozen=True)
class SelectionResult:
    mode: SelectorMode
    selected_candidate_id: int | None
    fallback: str | None
    decisions: tuple[CandidateDecision, ...]
    belief_snapshot: Mapping[str, object] | None


_VALUE_ONLY_WEIGHTS = ScoreWeights(0.0, 0.0, 0.0, 0.0, calibrated=True, requery_cost=0.0)


def refresh_from_current_observation(
    belief: BeliefState, attribution: AttributionOutput, block_index: int
) -> None:
    """Refresh only facts directly established by the post-block observation.

    A reliable current image can make the *new* target pose current after an
    object shift. It cannot, by itself, re-establish reachability, grasp, lift,
    or placement facts invalidated by the previous block.
    """
    if attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is not TriValue.TRUE:
        return
    confidence = min(
        attribution.confidence,
        attribution.factor_confidence(ConsistencyFactor.OBSERVATION_RELIABLE),
    )
    evidence_id = f"{attribution.source_block_id}:current_observation"
    for name in ("target_visible", "target_pose_current"):
        belief.facts[name] = BeliefFact(
            TriValue.TRUE, confidence, "observation", block_index, (evidence_id,)
        )


def select_value_only(values: Sequence[float]) -> SelectionResult:
    """Reproduce the official best-of-N argmax without additional gating."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite one-dimensional sequence")
    selected = int(np.argmax(array))
    decisions = tuple(
        CandidateDecision(index, True, float(value), float(value), (), {"official_value": float(value)})
        for index, value in enumerate(array)
    )
    return SelectionResult(SelectorMode.VALUE_ONLY, selected, None, decisions, None)


def select_hard_gate_value_tiebreak(
    *,
    mode: SelectorMode,
    belief: BeliefState,
    attribution: AttributionOutput,
    block_index: int,
    candidate_actions: Sequence[np.ndarray],
    official_values: Sequence[float],
    visual_support: Sequence[float] | None = None,
    visual_evidence: Sequence[CandidateVisualEvidence | None] | None = None,
    hard_confidence: float = 0.75,
) -> SelectionResult:
    """Apply attribution-aware belief gating, then official-value tie-break."""
    if mode not in {SelectorMode.ORACLE_HARD_GATE, SelectorMode.LEARNED_HARD_GATE}:
        raise ValueError("hard-gate selection requires oracle_hard_gate or learned_hard_gate mode")
    if len(candidate_actions) != len(official_values) or not candidate_actions:
        raise ValueError("candidate actions and official values must have equal non-zero length")
    supports = [0.5] * len(candidate_actions) if visual_support is None else list(visual_support)
    if len(supports) != len(candidate_actions):
        raise ValueError("visual_support length must match candidates")
    visual_items = [None] * len(candidate_actions) if visual_evidence is None else list(visual_evidence)
    if len(visual_items) != len(candidate_actions):
        raise ValueError("visual_evidence length must match candidates")

    update_belief(belief, attribution, block_index)
    refresh_from_current_observation(belief, attribution, block_index)
    decisions: list[CandidateDecision] = []
    for candidate_id, (actions, value, support, visual_item) in enumerate(
        zip(candidate_actions, official_values, supports, visual_items, strict=True)
    ):
        effect = parse_candidate_effect(
            candidate_id, actions, visual_support=float(support), visual_evidence=visual_item
        )
        decisions.append(
            evaluate_candidate(
                belief,
                attribution,
                effect,
                float(value),
                _VALUE_ONLY_WEIGHTS,
                hard_confidence=hard_confidence,
            )
        )
    chosen, fallback = select_candidate(decisions, attribution)
    return SelectionResult(
        mode,
        None if chosen is None else chosen.candidate_id,
        fallback,
        tuple(decisions),
        belief.snapshot(),
    )

"""Frozen D19--D21 selector policies for paired downstream evaluation.

The first downstream pilot deliberately excludes the uncalibrated D20 soft
weights. Oracle and learned attribution use the same auditable policy: update
belief, hard-gate infeasible candidates, then use the official Cosmos value as
the tie-breaker among feasible candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from .belief import DEFAULT_GRAPH, DependencyGraph, update_belief
from .candidate_effects import parse_candidate_effect
from .candidate_utility import (
    CandidateUtilityModel, candidate_utility_features, select_with_utility,
)
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


def _epistemic_only_rejection(decision: CandidateDecision) -> bool:
    """Return true only for unknown-state rejections without direct danger evidence."""
    return bool(decision.rejection_reasons) and all(
        "=unknown" in reason and not reason.startswith("candidate_evidence:")
        for reason in decision.rejection_reasons
    )


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
    score_weights: ScoreWeights | None = None,
    utility_model: CandidateUtilityModel | None = None,
    dependency_graph: DependencyGraph = DEFAULT_GRAPH,
) -> SelectionResult:
    """Apply belief gating and optionally candidate-specific calibrated scoring.

    ``score_weights=None`` preserves the frozen Method-V5 value tie-break.
    Passing calibrated weights enables the later D20 candidate-effect policy
    without changing candidate generation or the safety gate.
    """
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

    weights = _VALUE_ONLY_WEIGHTS if score_weights is None else score_weights
    if not weights.calibrated:
        raise RuntimeError("candidate-effect score weights must be calibrated")
    update_belief(belief, attribution, block_index, graph=dependency_graph)
    refresh_from_current_observation(belief, attribution, block_index)
    decisions: list[CandidateDecision] = []
    utility_features: dict[int, np.ndarray] = {}
    for candidate_id, (actions, value, support, visual_item) in enumerate(
        zip(candidate_actions, official_values, supports, visual_items, strict=True)
    ):
        effect = parse_candidate_effect(
            candidate_id, actions, visual_support=float(support), visual_evidence=visual_item
        )
        utility_features[candidate_id] = candidate_utility_features(
            effect, float(value), attribution
        )
        decisions.append(
            evaluate_candidate(
                belief,
                attribution,
                effect,
                float(value),
                weights,
                hard_confidence=hard_confidence,
            )
        )
    if utility_model is None:
        chosen, fallback = select_candidate(decisions, attribution)
    else:
        # Safe residual contract: when the official-value anchor is rejected
        # solely because a prerequisite is unknown, there is no positive
        # evidence that it is unsafe.  Preserve the baseline in that narrowly
        # defined epistemic case.  Explicit false predicates and candidate
        # hard violations are never overridden.
        value_anchor = max(
            decisions, key=lambda item: (item.official_value, -item.candidate_id)
        )
        if not value_anchor.accepted and _epistemic_only_rejection(value_anchor):
            decisions = [
                replace(
                    item,
                    accepted=True,
                    total_score=item.official_value,
                    rejection_reasons=(),
                    components={
                        **item.components,
                        "baseline_preserving_epistemic_override": 1.0,
                        "overridden_epistemic_rejection_count": float(
                            len(item.rejection_reasons)
                        ),
                    },
                ) if item.candidate_id == value_anchor.candidate_id else item
                for item in decisions
            ]
        chosen, utility_scores = select_with_utility(decisions, utility_features, utility_model)
        decisions = [
            replace(
                decision,
                components={
                    **decision.components,
                    **({"learned_utility": utility_scores[decision.candidate_id]}
                       if decision.candidate_id in utility_scores else {}),
                },
            )
            for decision in decisions
        ]
        # ``chosen`` above references the pre-annotation object; recover the
        # corresponding auditable decision after adding utility components.
        if chosen is not None:
            chosen = next(item for item in decisions if item.candidate_id == chosen.candidate_id)
            fallback = None
        else:
            _, fallback = select_candidate(decisions, attribution)
    return SelectionResult(
        mode,
        None if chosen is None else chosen.candidate_id,
        fallback,
        tuple(decisions),
        belief.snapshot(),
    )

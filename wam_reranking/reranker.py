"""Prerequisite gating, attribution-aware scoring and explicit fallbacks."""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import (
    AttributionOutput, BeliefState, CandidateDecision, CandidateEffect, CoarseCause,
    ConsistencyFactor, ScoreWeights, Stage, TriValue,
)


def _compatibility(attribution: AttributionOutput, stage: Stage) -> float:
    score = 0.0
    if attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is not TriValue.TRUE:
        score += 0.8 if stage is Stage.OBSERVE else -0.6
    if attribution.factor_state(ConsistencyFactor.WORLD_STATE_CONSISTENT) is not TriValue.TRUE:
        score += 0.7 if stage in {Stage.OBSERVE, Stage.APPROACH} else -0.8
    if attribution.factor_state(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT) is not TriValue.TRUE:
        score += 0.5 if stage in {Stage.OBSERVE, Stage.APPROACH, Stage.GRASP} else -0.8
    if attribution.factor_state(ConsistencyFactor.TASK_STAGE_CONSISTENT) is not TriValue.TRUE:
        score += 0.5 if stage in {Stage.OBSERVE, Stage.APPROACH} else -0.5
    return score


def evaluate_candidate(belief: BeliefState, attribution: AttributionOutput, effect: CandidateEffect,
                       official_value: float, weights: ScoreWeights, *, hard_confidence: float = 0.75,
                       query_cost: float = 0.0, allow_uncalibrated: bool = False) -> CandidateDecision:
    if not weights.calibrated and not allow_uncalibrated:
        raise RuntimeError("score weights must be calibrated before deployment")
    if not 0.0 <= official_value <= 1.0:
        raise ValueError("official value must be in [0,1]")
    if query_cost < 0.0:
        raise ValueError("query_cost must be non-negative")
    rejection: list[str] = []
    risk = 0.0
    uncertainty = 1.0 - effect.confidence
    safety_critical = {
        Stage.GRASP: {"target_pose_current", "target_reachable"},
        Stage.LIFT: {"grasped", "execution_consistent"},
        Stage.TRANSPORT: {"grasped", "lifted"},
        Stage.PLACE: {"lifted", "place_ready"},
    }.get(effect.stage, set())
    for requirement, required_confidence in effect.required_facts.items():
        fact = belief.facts[requirement]
        if fact.value is TriValue.FALSE and fact.confidence >= hard_confidence:
            rejection.append(f"{requirement}=false@{fact.confidence:.2f}")
        elif fact.value is TriValue.UNKNOWN and requirement in safety_critical and fact.confidence >= hard_confidence:
            rejection.append(f"{requirement}=unknown@{fact.confidence:.2f}")
        elif fact.value is not TriValue.TRUE:
            risk += required_confidence * max(fact.confidence, 0.25)
        else:
            risk += required_confidence * max(0.0, required_confidence - fact.confidence)
        uncertainty += 1.0 - fact.confidence
    if rejection:
        return CandidateDecision(effect.candidate_id, False, official_value, None, tuple(rejection), {})
    compatibility = _compatibility(attribution, effect.stage)
    progress = {Stage.OBSERVE: 0.0, Stage.APPROACH: 0.2, Stage.GRASP: 0.4, Stage.LIFT: 0.6,
                Stage.TRANSPORT: 0.7, Stage.PLACE: 1.0, Stage.UNCERTAIN: -0.2}[effect.stage]
    total = (official_value + weights.attribution_compatibility * compatibility
             + weights.progress * progress - weights.dependency_risk * risk
             - weights.uncertainty * uncertainty - weights.requery_cost * query_cost)
    return CandidateDecision(effect.candidate_id, True, official_value, float(total), (), {
        "official_value": official_value, "attribution_compatibility": compatibility,
        "progress": progress, "dependency_risk": risk, "uncertainty": uncertainty,
        "query_cost": query_cost,
    })


def select_candidate(decisions: Sequence[CandidateDecision], attribution: AttributionOutput) -> tuple[CandidateDecision | None, str | None]:
    accepted = [decision for decision in decisions if decision.accepted]
    if accepted:
        return max(accepted, key=lambda item: float(item.total_score)), None
    if attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is not TriValue.TRUE:
        return None, "reobserve"
    if attribution.projected_cause is CoarseCause.UNKNOWN:
        quality = attribution.evidence_quality
        if not quality.primary_reliable and not quality.wrist_reliable:
            return None, "reobserve"
        return None, "safe_reject"
    return None, "requery"

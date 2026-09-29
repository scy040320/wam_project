"""Prerequisite gating, attribution-aware scoring and explicit fallbacks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .contracts import (
    AttributionOutput, BeliefState, CandidateDecision, CandidateEffect, CoarseCause,
    ConsistencyFactor, ScoreWeights, Stage, TriValue,
)


PREDICATE_HARD_THRESHOLDS = {
    "target_pose_current": 0.60,
    "target_reachable": 0.65,
    "execution_consistent": 0.60,
    "grasped": 0.55,
    "lifted": 0.55,
    "place_ready": 0.60,
}


def _attribution_invalidates_requirement(
    attribution: AttributionOutput, requirement: str, effect: CandidateEffect
) -> tuple[bool, str | None, float]:
    routes = {
        ConsistencyFactor.WORLD_STATE_CONSISTENT: {
            "target_pose_current", "target_reachable", "grasped", "lifted", "place_ready", "placed",
        },
        ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT: {
            "execution_consistent", "grasped", "lifted", "place_ready", "placed",
        },
        ConsistencyFactor.TASK_STAGE_CONSISTENT: {"grasped", "lifted", "place_ready", "placed"},
    }
    for factor, predicates in routes.items():
        if requirement not in predicates:
            continue
        state = attribution.factor_state(factor)
        confidence = attribution.factor_confidence(factor)
        if (
            factor is ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT
            and requirement in {"grasped", "lifted", "place_ready"}
            and _candidate_preserves_contact_chain(effect)
        ):
            # An execution mismatch directly invalidates command consistency,
            # but it does not prove that the object was dropped.  A complete
            # closed-gripper -> transport -> release chunk with observable
            # contact continuity can retain derived object predicates.  This
            # exception never overrides a world-state invalidation above.
            continue
        if state is not TriValue.TRUE and confidence >= 0.5:
            return True, factor.value, confidence
    return False, None, 0.0


def _candidate_preserves_contact_chain(effect: CandidateEffect) -> bool:
    """Return deployable evidence that a place chunk retains object contact.

    This is deliberately predicate-specific.  It cannot restore
    ``execution_consistent`` and it cannot override an object/world shift.
    """
    if effect.stage is not Stage.PLACE:
        return False
    evidence = effect.evidence
    before_distance = float(evidence.get("target_gripper_distance_before", 1.0))
    after_distance = float(evidence.get("target_gripper_distance_after", 1.0))
    before_affinity = float(evidence.get("target_gripper_affinity_before", 0.0))
    after_affinity = float(evidence.get("target_gripper_affinity_after", 0.0))
    contact_confidence = float(evidence.get("contact_confidence", 0.0))
    before_support = max(before_affinity, 1.0 - min(1.0, 2.0 * before_distance))
    continuity = after_distance <= before_distance + 0.03 or after_affinity >= before_affinity - 0.05
    commanded_sequence = (
        float(evidence.get("close_strength", 0.0)) >= 0.5
        and float(evidence.get("open_strength", 0.0)) >= 0.5
    )
    return contact_confidence >= 0.55 and before_support >= 0.35 and continuity and commanded_sequence


def _compatibility(attribution: AttributionOutput, effect: CandidateEffect) -> float:
    stage = effect.stage
    score = 0.0
    if attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is not TriValue.TRUE:
        score += 0.8 if stage is Stage.OBSERVE else -0.6
    if attribution.factor_state(ConsistencyFactor.WORLD_STATE_CONSISTENT) is not TriValue.TRUE:
        score += 0.7 if stage in {Stage.OBSERVE, Stage.APPROACH} else -0.8
    if attribution.factor_state(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT) is not TriValue.TRUE:
        score += 0.5 if stage in {Stage.OBSERVE, Stage.APPROACH, Stage.GRASP} else -0.8
    if attribution.factor_state(ConsistencyFactor.TASK_STAGE_CONSISTENT) is not TriValue.TRUE:
        score += 0.5 if stage in {Stage.OBSERVE, Stage.APPROACH} else -0.5
    evidence = effect.evidence
    relation_confidence = float(evidence.get("relation_confidence", 0.0))
    contact_confidence = float(evidence.get("contact_confidence", 0.0))
    relation_delta = float(evidence.get("relation_score_delta", 0.0))
    grasp_support = float(evidence.get("predicted_grasp_support", 0.0))
    release_support = float(evidence.get("predicted_release_support", 0.0))
    if attribution.factor_state(ConsistencyFactor.WORLD_STATE_CONSISTENT) is not TriValue.TRUE:
        # After a world-state mismatch, candidate compatibility must depend on
        # its own predicted subject--anchor recovery, not only a coarse stage.
        score += relation_confidence * float(np.clip(2.0 * relation_delta, -1.0, 1.0))
    if attribution.factor_state(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT) is not TriValue.TRUE:
        if stage in {Stage.GRASP, Stage.LIFT, Stage.TRANSPORT}:
            score += contact_confidence * grasp_support
        elif stage is Stage.PLACE:
            score += contact_confidence * release_support
    return float(score)


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
    for reason, confidence in effect.hard_violations.items():
        if confidence >= 0.50:
            rejection.append(f"candidate_evidence:{reason}@{confidence:.2f}")
    for requirement, required_confidence in effect.required_facts.items():
        fact = belief.facts[requirement]
        predicate_threshold = PREDICATE_HARD_THRESHOLDS.get(requirement, hard_confidence)
        invalidated, factor_name, source_confidence = _attribution_invalidates_requirement(
            attribution, requirement, effect
        )
        if fact.value is TriValue.FALSE and fact.confidence >= predicate_threshold:
            rejection.append(f"{requirement}=false@{fact.confidence:.2f}")
        elif fact.value is TriValue.UNKNOWN and requirement in safety_critical:
            if fact.source == "attribution" and invalidated:
                rejection.append(
                    f"{requirement}=unknown_from_{factor_name}@{source_confidence:.2f}"
                )
            elif fact.source != "attribution" and fact.confidence >= predicate_threshold:
                rejection.append(f"{requirement}=unknown@{fact.confidence:.2f}")
            else:
                # Derived uncertainty from attribution is a soft risk when a
                # predicate-specific candidate trace supplies continuity
                # evidence.  It is not made safe merely by confidence decay.
                risk += required_confidence * max(fact.confidence, 0.25)
        elif fact.value is not TriValue.TRUE:
            risk += required_confidence * max(fact.confidence, 0.25)
        else:
            risk += required_confidence * max(0.0, required_confidence - fact.confidence)
        uncertainty += 1.0 - fact.confidence
    if rejection:
        return CandidateDecision(effect.candidate_id, False, official_value, None, tuple(rejection), {})
    risk += float(effect.evidence.get("relation_regression_soft", 0.0))
    risk += 0.5 * float(effect.evidence.get("trajectory_risk", 0.0))
    compatibility = _compatibility(attribution, effect)
    stage_progress = {Stage.OBSERVE: 0.0, Stage.APPROACH: 0.2, Stage.GRASP: 0.4, Stage.LIFT: 0.6,
                      Stage.TRANSPORT: 0.7, Stage.PLACE: 1.0, Stage.UNCERTAIN: -0.2}[effect.stage]
    relation_progress = (
        float(effect.evidence.get("relation_score_delta", 0.0))
        * float(effect.evidence.get("relation_confidence", 0.0))
    )
    if effect.stage in {Stage.GRASP, Stage.LIFT, Stage.TRANSPORT}:
        manipulation_progress = float(effect.evidence.get("predicted_grasp_support", 0.0))
    elif effect.stage is Stage.PLACE:
        manipulation_progress = float(effect.evidence.get("predicted_release_support", 0.0))
    else:
        manipulation_progress = 0.0
    progress = float(stage_progress + relation_progress + 0.25 * manipulation_progress)
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

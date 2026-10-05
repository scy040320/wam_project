"""Evidence-source scoped belief gating on a shared candidate backbone.

Unknown is not evidence of a physical false state.  Confidence in the coarse
unknown class must not become confidence in a world-state violation.  This
policy is separate from the frozen historical selector; it has no task IDs,
scene exceptions, outcome inputs, or newly fitted decision thresholds.
"""
from dataclasses import replace

from .belief import DEFAULT_GRAPH, update_belief
from .candidate_utility import select_with_utility
from .contracts import (
    BeliefFact, CoarseCause, ConsistencyFactor, ScoreWeights, TriValue,
)
from .evidence_residual import candidate_backbone_features, typed_gate_effect
from .policy import _epistemic_only_rejection, refresh_from_current_observation
from .reranker import evaluate_candidate


def scoped_attribution(attribution):
    """Mask an unsupported physical interpretation, not the original output."""
    states = {f.value: attribution.factor_state(f) for f in ConsistencyFactor}
    reliable = (
        states[ConsistencyFactor.OBSERVATION_RELIABLE.value] is TriValue.TRUE
        and not attribution.evidence_quality.cross_view_conflict
    )
    resolved = states[ConsistencyFactor.CAUSE_RESOLVED.value] is TriValue.TRUE
    world = ConsistencyFactor.WORLD_STATE_CONSISTENT.value
    if states[world] is not TriValue.TRUE and not (reliable and resolved):
        states[world] = TriValue.UNKNOWN
    # Measured command-record evidence remains usable despite image occlusion.
    return replace(attribution, factor_states=states)


def prepare_source_scoped_decisions(*, belief, attribution, block_index,
                                    effects, values, relation, backbone,
                                    graph=DEFAULT_GRAPH):
    routed = scoped_attribution(attribution)
    # Do not erase a directly witnessed physical contradiction with a learned
    # uncertain attribution. Fresh target pose is separately refreshed below.
    witnessed = {n: replace(f) for n, f in belief.facts.items()
                 if f.source == "observation" and f.value is TriValue.FALSE
                 and n not in {"target_visible", "target_pose_current"}}
    start = len(belief.history)
    update_belief(belief, routed, block_index, graph=graph)
    # update_belief's historical aggregate confidence is inappropriate for
    # multi-label routing. Cap each new fact/change by ITS factor evidence.
    factors = {f.value: f for f in ConsistencyFactor}
    for index in range(start, len(belief.history)):
        change = belief.history[index]
        if change.reason not in factors:
            continue
        confidence = min(change.new_confidence,
                         attribution.factor_confidence(factors[change.reason]))
        belief.history[index] = replace(change, new_confidence=confidence)
    for name, fact in list(belief.facts.items()):
        changes = [c for c in belief.history[start:] if c.predicate == name]
        if changes:
            belief.facts[name] = replace(fact, confidence=changes[-1].new_confidence)
    belief.facts.update(witnessed)
    refresh_from_current_observation(belief, attribution, block_index)
    weights = ScoreWeights(0., 0., 0., 0., calibrated=True)
    decisions = [evaluate_candidate(belief, routed, typed_gate_effect(e, relation),
                                   float(v), weights)
                 for e, v in zip(effects, values, strict=True)]
    unfiltered = [replace(d, accepted=True, rejection_reasons=(),
                          components={"dependency_risk": 0., "uncertainty": 0.})
                  for d in decisions]
    base, _ = select_with_utility(unfiltered,
        {e.candidate_id: candidate_backbone_features(e, float(v))
         for e, v in zip(effects, values, strict=True)}, backbone)
    anchors = {base.candidate_id,
               max(decisions, key=lambda d: (d.official_value, -d.candidate_id)).candidate_id}
    result = []
    # Distinguish unresolved state-change evidence from an otherwise normal
    # physical state with an unresolved cause. In the latter case there is no
    # supported recovery need: retain the original conservative value route.
    # This is a factor-semantic rule, not a task/state-specific exception.
    unreliable = (attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE)
                  is not TriValue.TRUE or attribution.evidence_quality.cross_view_conflict)
    unresolved = (attribution.projected_cause is CoarseCause.UNKNOWN or
                  attribution.factor_state(ConsistencyFactor.CAUSE_RESOLVED) is not TriValue.TRUE)
    physical_consistent = all(attribution.factor_state(f) is TriValue.TRUE for f in (
        ConsistencyFactor.WORLD_STATE_CONSISTENT,
        ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT,
        ConsistencyFactor.TASK_STAGE_CONSISTENT))
    for decision in decisions:
        components = {**decision.components, "source_scoped_belief": 1.}
        if (decision.candidate_id in anchors and not decision.accepted
                and _epistemic_only_rejection(decision)):
            components["baseline_preserving_epistemic_override"] = 1.
            components["overridden_epistemic_rejection_count"] = float(len(decision.rejection_reasons))
            decision = replace(decision, accepted=True,
                               total_score=decision.official_value, rejection_reasons=())
        if unreliable or (unresolved and physical_consistent):
            components["preserve_official_value"] = 1.
        if unresolved and physical_consistent:
            components["unresolved_without_supported_physical_recovery"] = 1.
        if routed.factor_state(ConsistencyFactor.WORLD_STATE_CONSISTENT) is TriValue.UNKNOWN:
            components["world_cause_not_physically_resolved"] = 1.
        result.append(replace(decision, components=components))
    return tuple(result)

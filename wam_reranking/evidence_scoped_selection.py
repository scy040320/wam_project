"""Additive PRE-only routing repair; historical frozen selectors stay untouched.

Coarse abstention is metadata, not a physical contradiction.  Predicate writes
require their own evidence.  This module intentionally does not manufacture a
current pose certificate from a global reliability or observability score.
"""
from copy import deepcopy
from dataclasses import replace

import numpy as np

from .belief import DEFAULT_GRAPH, DependencyGraph
from .contracts import (BeliefChange, BeliefFact, CoarseCause, ConsistencyFactor,
                        ScoreWeights, Stage, TriValue)
from .candidate_utility import select_with_utility
from .evidence_residual import candidate_backbone_features, cause_residual_features, typed_gate_effect
from .reranker import PREDICATE_HARD_THRESHOLDS, evaluate_candidate, select_candidate
from .whole_chain_attribution import absent_carrier
from . import command_predicate_utility as command_api

SCHEMA = "predicate_scoped_attribution_selection_v2"
MODES = ("full", "ranker_command", "learned_no_dag", "cause_only", "quality_only", "shuffled")


def usable_visual(attribution):
    q = attribution.evidence_quality
    return bool((q.primary_reliable or q.wrist_reliable) and not q.cross_view_conflict
                and attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is TriValue.TRUE)


def _write(belief, predicate, value, confidence, reason, block, evidence_id, path):
    old = belief.facts[predicate]
    # A threshold-qualified FALSE cannot be erased by abstention/propagation.
    # A weak FALSE is not a physical contradiction certificate: retaining it
    # would hide a later, stronger UNKNOWN invalidation from the safety gate.
    # Its original value/confidence/evidence remain in the transition history.
    supported_false = old.confidence >= PREDICATE_HARD_THRESHOLDS.get(predicate, .75)
    if old.value is TriValue.FALSE and supported_false and value is TriValue.UNKNOWN:
        return
    belief.facts[predicate] = BeliefFact(value, float(confidence), "attribution", block, (evidence_id,))
    belief.history.append(BeliefChange(predicate, old.value, old.confidence, value,
        float(confidence), reason, len(path) == 1, path, evidence_id))


def update_scoped_belief(prior, attribution, block_index, *, graph=DEFAULT_GRAPH):
    """Factor confidence, not coarse-class confidence, determines each write.

    UNKNOWN factors add no physical claim. A negative WORLD consistency factor
    invalidates the preceding location estimate; it does not certify a false
    current location. TRUE consistency does not restore a previously invalidated
    grasp. Coarse UNKNOWN adds no physical writes at all.
    """
    belief = deepcopy(prior)
    q = attribution.evidence_quality
    visual = usable_visual(attribution)
    obs = ConsistencyFactor.OBSERVATION_RELIABLE
    if not visual:
        strength = attribution.factor_confidence(obs)
        # Only freshness/visibility are affected; established physical facts
        # such as grasp/lift are retained. No DAG propagation for poor views.
        for predicate in ("target_visible", "target_pose_current"):
            _write(belief, predicate, TriValue.UNKNOWN, strength, obs.value,
                   block_index, attribution.source_block_id, (predicate,))
    sources = (
        (ConsistencyFactor.WORLD_STATE_CONSISTENT, "target_pose_current", visual),
        (ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT, "execution_consistent",
         visual and q.execution_reliable),
    )
    for factor, root, source_usable in sources:
        strength = attribution.factor_confidence(factor)
        if not source_usable or attribution.factor_state(factor) is not TriValue.FALSE or strength < .5:
            continue
        paths = {root: (root,), **graph.descendants_with_paths(root)}
        for predicate, path in paths.items():
            old = belief.facts[predicate]
            # A fresh independent observation does not depend on an old
            # assumed pose/command. No candidate forecast qualifies here.
            if (predicate != root and old.source == "observation" and old.updated_at_block >= block_index
                    and old.confidence > 0 and "initial_observation" not in old.evidence_ids
                    and old.value is not TriValue.UNKNOWN):
                continue
            # WORLD is a comparison with the preceding prediction, not a
            # current-object detector. Its root means "old estimate stale,
            # current estimate unconfirmed", never a physical FALSE witness.
            # Keep its confidence, provenance and DAG paths. The unchanged
            # safety gate still rejects unknown critical grasp prerequisites.
            value = (TriValue.FALSE if predicate == root and
                     factor is ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT
                     else TriValue.UNKNOWN)
            confidence = max(.05, strength * .85 ** (len(path) - 1))
            _write(belief, predicate, value, confidence, factor.value,
                   block_index, attribution.source_block_id, path)
    return belief


def apply_current_facts(belief, facts, block_index):
    """Optional explicit current-observation witnesses, never forecasts.

    Global learned reliability/observability alone cannot satisfy this API.
    The replay runner supplies no synthetic witnesses when records lack them.
    """
    for item in facts:
        required = {"predicate", "value", "confidence", "block_index", "source", "evidence_id", "predecision"}
        if set(item) != required or item["source"] != "current_observation" or item["predecision"] is not True:
            raise ValueError("Explicit PRE current-observation witness required")
        if item["block_index"] != block_index or not item["evidence_id"] or item["predicate"] not in belief.facts:
            raise ValueError("Current witness identity/time mismatch")
        value, confidence = TriValue(item["value"]), float(item["confidence"])
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid current witness confidence")
        old = belief.facts[item["predicate"]]
        if value is TriValue.UNKNOWN and old.value is TriValue.FALSE and old.confidence > 0:
            continue
        belief.facts[item["predicate"]] = BeliefFact(value, confidence, "observation", block_index, (item["evidence_id"],))
        belief.history.append(BeliefChange(item["predicate"], old.value, old.confidence, value, confidence,
            "current_observation", True, (item["predicate"],), item["evidence_id"]))


def gate_attribution(attribution):
    """No reliable visual source means no learned physical hard certificate.

    A reliable measured command remains a separate recipient channel. This
    projection changes source usability, not raw probabilities or thresholds.
    """
    states, strengths = dict(attribution.factor_states), dict(attribution.factor_confidences)
    if not usable_visual(attribution):
        for factor in (ConsistencyFactor.WORLD_STATE_CONSISTENT, ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT):
            states[factor.value], strengths[factor.value] = TriValue.UNKNOWN, 0.
    if not attribution.evidence_quality.execution_reliable:
        factor = ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT
        states[factor.value], strengths[factor.value] = TriValue.UNKNOWN, 0.
    return replace(attribution, factor_states=states, factor_confidences=strengths)


def unresolved_history_rejections(belief, effect):
    """Normal/poor-view evidence cannot certify an old missing prerequisite.

    Inherit the evaluator's critical-use stages and .50 attribution source
    threshold, not a new acceptance threshold. Propagation decay does not erase
    its original invalidation. Only an explicit current-observation TRUE write
    resolves that history; a candidate forecast or consistency TRUE does not.
    BeliefChange lacks a block field, so this is a history-provenance check,
    not a claim of independently certified historical timing or physical truth.
    """
    critical = {
        Stage.GRASP: {'target_pose_current', 'target_reachable'},
        Stage.LIFT: {'grasped', 'execution_consistent'},
        Stage.TRANSPORT: {'grasped', 'lifted'},
        Stage.PLACE: {'lifted'},
    }.get(effect.stage, set())
    roots = {'world_state_consistent': 'target_pose_current',
             'execution_contact_consistent': 'execution_consistent'}
    reasons = []
    for predicate in sorted(critical.intersection(effect.required_facts)):
        if belief.facts[predicate].value is not TriValue.UNKNOWN:
            continue
        for change in reversed(belief.history):
            if change.predicate != predicate:
                continue
            if change.reason == 'current_observation' and change.new_value is TriValue.TRUE:
                break
            root = roots.get(change.reason)
            if (root is None or change.new_value is TriValue.TRUE
                    or not change.propagation_path
                    or change.propagation_path[0] != root
                    or change.propagation_path[-1] != predicate):
                continue
            origins = [c for c in belief.history if c.predicate == root
                and c.reason == change.reason and c.evidence_id == change.evidence_id
                and c.direct and c.propagation_path == (root,)
                and c.new_value is not TriValue.TRUE]
            if origins:
                strength = origins[-1].new_confidence
                if strength >= .5:
                    reasons.append(f'{predicate}=unresolved_history_from_{change.reason}@{strength:.2f}')
            break
    return tuple(reasons)


def evaluate_scoped_candidate(belief, attribution, effect, value, weights):
    decision = evaluate_candidate(belief, attribution, effect, value, weights)
    historical = unresolved_history_rejections(belief, effect)
    if historical and decision.accepted:
        return replace(decision, accepted=False, total_score=None,
                       rejection_reasons=historical,
                       components={**decision.components, 'unresolved_history_hard_gate': 1.})
    return decision


def comparable_ids(decisions, traces, attribution, reference_id):
    """Keep .50/.55 source masks and actual command evidence unchanged.

    UNKNOWN does not introduce a second official-value route or a global veto.
    A frozen correction must still have the relevant usable missing predicate.
    """
    if reference_id is None:
        return set()
    answer = {reference_id}
    reference = traces[reference_id]
    for decision in decisions:
        cid = decision.candidate_id
        if not decision.accepted or cid == reference_id:
            continue
        trace = traces[cid]
        visual_ok = usable_visual(attribution) and any(
            predicate != "execution_consistent" and trace["predicate_usable"][predicate]
            and reference["predicate_usable"][predicate] for predicate in trace["deficits"])
        deficit = trace["deficits"].get("execution_consistent", {})
        execution_ok = (trace["execution_usable"] and reference["execution_usable"]
            and deficit.get("route") == "execution_contact_deviation" and deficit.get("state") == "false"
            and trace["command_efficiency"] > reference["command_efficiency"] + 1e-12
            and trace["command_risk"] <= reference["command_risk"] + 1e-12)
        if visual_ok or execution_ok:
            answer.add(cid)
    return answer


def select_scoped_chain(*, prior, attribution, effects, plans, values, relation,
                        block_index, command, availability, backbone, frozen_residual,
                        conditional_model, mode="full", current_facts=(), gate_effects=None):
    """No terminal outcome arguments; same frozen models and switch margin.

    No epistemic-only acceptance override is added. Candidate hard violations
    and witnessed FALSE prerequisite checks use the unchanged gate evaluator.
    """
    if mode not in MODES or len(effects) != len(plans) or len(plans) != len(values) or not effects:
        raise ValueError("Aligned nonempty PRE pool required")
    if len(availability) != 3 or any(type(value) is not bool for value in availability):
        raise ValueError("Actual recipient availability requires three literal bools")
    if command is None or not command.valid or command.use_block_index != block_index:
        raise ValueError("Verified preceding recipient command required")
    if [effect.candidate_id for effect in effects] != list(range(len(effects))):
        raise ValueError("Contiguous aligned candidate IDs required")
    # A source-audited adapter may provide a gate-only projection. The original
    # candidate prediction features remain unchanged for frozen scoring.
    gate_effects = effects if gate_effects is None else gate_effects
    if len(gate_effects) != len(effects) or any(
            gate.candidate_id != effect.candidate_id or gate.stage != effect.stage
            or dict(gate.required_facts) != dict(effect.required_facts)
            or dict(gate.hard_violations) != dict(effect.hard_violations)
            for gate, effect in zip(gate_effects, effects, strict=True)):
        raise ValueError("Gate projection cannot alter membership, prerequisites or hard violations")
    no_learned, no_dag = mode == "ranker_command", mode == "learned_no_dag"
    learned_reasons = {factor.value for factor in ConsistencyFactor} | {'consistent_block'}
    if no_learned and any(change.reason in learned_reasons for change in prior.history):
        raise ValueError('No-learned control requires its own attribution-free history, not a shared full-method prior')
    if no_dag and any(change.reason in learned_reasons and len(change.propagation_path) > 1
                      for change in prior.history):
        raise ValueError('No-DAG control requires its own unpropagated history, not a shared full-method prior')
    graph = DependencyGraph(()) if no_dag else DEFAULT_GRAPH
    if no_learned:
        attribution = absent_carrier(command.source_block_id, availability)
        belief = deepcopy(prior)
    else:
        belief = update_scoped_belief(prior, attribution, block_index, graph=graph)
    apply_current_facts(belief, current_facts, block_index)
    history_start = len(prior.history)
    weights = ScoreWeights(0., 0., 0., 0., calibrated=True)
    gate_attr = gate_attribution(attribution)
    decisions = tuple(evaluate_scoped_candidate(belief, gate_attr, typed_gate_effect(effect, relation), float(value), weights)
                      for effect, value in zip(gate_effects, values, strict=True))
    xs = {effect.candidate_id: candidate_backbone_features(effect, float(value))
          for effect, value in zip(effects, values, strict=True)}
    # Preserve the frozen selector's parameter-free Pareto switch guard.
    # Learned score penalties are not equivalent to that deployment constraint:
    # erasing these components had silently disabled an existing safeguard.
    # All attribution controls use this same guard and unchanged switch margin.
    reference, base_scores = select_with_utility(decisions, xs, backbone)
    zs = {d.candidate_id: cause_residual_features(typed_gate_effect(e, relation), attribution, d)
          for e, d in zip(effects, decisions, strict=True)}
    base_scores = {cid: score + (0. if no_learned else frozen_residual.score(zs[cid]))
                   for cid, score in base_scores.items()}
    # Existing frozen evidence model currently has gain0. Guard against silently
    # misusing a future nonzero version whose reference selection also differs.
    # A nonzero stored cap is harmless when gain is exactly zero: the frozen
    # model's score() multiplies by gain before clipping. Do not invent a new
    # requirement that the historical cap itself must also be zero.
    if frozen_residual.gain != 0:
        raise ValueError("This repair is pinned to the frozen zero-gain evidence residual")
    feature_attr = attribution
    if no_learned:
        feature_attr = replace(attribution, factor_states={**attribution.factor_states,
            ConsistencyFactor.OBSERVATION_RELIABLE.value: TriValue.TRUE if any(availability[:2]) else TriValue.UNKNOWN})
    vectors, traces = [], []
    for effect, actions in zip(effects, plans, strict=True):
        vector, trace = command_api.conditioned_features(prior=belief, attribution=feature_attr,
            effect=effect, actions=actions, relation=relation, no_dag=no_dag, block_index=block_index,
            belief_already_updated=True, history_start=history_start, command=command)
        vectors.append(vector); traces.append(trace)
    deltas = np.asarray(conditional_model.deltas(np.asarray(vectors)), dtype=float)
    if deltas.shape != (len(effects),) or not np.isfinite(deltas).all() or np.ptp(deltas) > command_api.CAP + 1e-12:
        raise ValueError("Frozen conditional model alignment/cap failure")
    reference_id = None if reference is None else reference.candidate_id
    comparable = comparable_ids(decisions, traces, feature_attr, reference_id)
    scores = {d.candidate_id: base_scores[d.candidate_id] + deltas[d.candidate_id] for d in decisions if d.accepted}
    selected = reference
    margin = max(.01, float(backbone.switch_margin))
    if reference_id is not None:
        eligible = [d for d in decisions if d.accepted and d.candidate_id != reference_id
                    and d.candidate_id in comparable and deltas[d.candidate_id] > deltas[reference_id] + 1e-12
                    and scores[d.candidate_id] > scores[reference_id] + margin]
        if eligible:
            top = max(scores[d.candidate_id] for d in eligible)
            selected = max((d for d in eligible if top - scores[d.candidate_id] <= margin),
                           key=lambda d: (d.official_value, -d.candidate_id))
    fallback = None
    if selected is None:
        _, fallback = select_candidate(decisions, gate_attr)
    return dict(schema=SCHEMA, mode=mode, prior=prior.snapshot(), live_belief=belief.snapshot(),
        attribution=attribution, gate_attribution=gate_attr, decisions=decisions,
        features=np.asarray(vectors), traces=traces, base_scores=base_scores, deltas=deltas,
        scores=scores, comparable_ids=sorted(comparable), reference_id=reference_id,
        selected_id=None if selected is None else selected.candidate_id, fallback=fallback,
        graph_edges=list(graph.edges), source_scoped_gate_confidences=True,
        stale_world_estimate_is_unknown_not_physical_false=True,
        frozen_reference_pareto_guard_preserved=True,
        unresolved_history_safety_checks_preserved=True,
        independent_history_required_for_whole_chain_controls=True,
        coarse_unknown_physical_writes=False, official_value_route_forced=False,
        epistemic_rejection_overrides=0, current_witness_count=len(current_facts),
        future_predictions_written_to_belief=False, newly_trained=False)

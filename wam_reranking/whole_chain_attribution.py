"""PRE-only full-chain attribution controls, without a retained full-V8 anchor.

The no-learned arm does not inject a fake NORMAL prediction. It skips learned
belief updates, learned observation refresh, learned gate invalidation and
learned route arbitration. Its carrier has zero evidence confidence. Actual
command measurements remain common; candidate forecasts are never observations.
Each rollout must maintain its OWN prior. A post-attribution historical V8
snapshot is not a valid no-learned starting point.
"""
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np

from .belief import DEFAULT_GRAPH, DependencyGraph
from .contracts import (AttributionOutput, CoarseCause, ConsistencyFactor,
                        EvidenceQuality, ScoreWeights, TriValue)
from .candidate_utility import select_with_utility
from .evidence_arbitration import prepare_arbitrated_decisions, select_evidence_arbitration
from .evidence_residual import candidate_backbone_features, cause_residual_features, typed_gate_effect
from .policy import _epistemic_only_rejection
from .reranker import evaluate_candidate, select_candidate
from . import command_predicate_utility as command_api

SCHEMA = "predecision_whole_chain_attribution_controls_v1"
MODES = ("ranker_command", "learned_no_dag", "full", "shuffled")


def absent_carrier(source_block_id, availability):
    """Compatibility carrier only: no physical consistency or normal certificate."""
    return AttributionOutput(
        {f.value: .5 for f in ConsistencyFactor},
        {c.value: 1 / len(CoarseCause) for c in CoarseCause},
        CoarseCause.UNKNOWN, 0., math.log(len(CoarseCause)),
        EvidenceQuality(*availability, cross_view_conflict=False), source_block_id,
        {f.value: TriValue.UNKNOWN for f in ConsistencyFactor},
        {f.value: 0. for f in ConsistencyFactor})


def recipient_attribution(prediction, *, source_block_id, availability):
    """Replace learned output coherently; retain recipient sensor availability.

    Learned reliability/conflict comes from the donor in shuffled mode, not
    secretly from the original recipient prediction. Physical file availability
    and measured command validity cannot be shuffled away.
    """
    q = prediction.evidence_quality
    return replace(prediction, source_block_id=source_block_id,
        evidence_quality=EvidenceQuality(q.primary_reliable and availability[0],
            q.wrist_reliable and availability[1], availability[2], q.cross_view_conflict))


def _no_learned_decisions(belief, carrier, effects, values, relation, backbone):
    weights = ScoreWeights(0., 0., 0., 0., calibrated=True)
    decisions = [evaluate_candidate(belief, carrier, typed_gate_effect(e, relation),
                                    float(v), weights)
                 for e, v in zip(effects, values, strict=True)]
    unfiltered = [replace(d, accepted=True, rejection_reasons=(),
                          components={"dependency_risk": 0., "uncertainty": 0.})
                  for d in decisions]
    xs = {e.candidate_id: candidate_backbone_features(e, float(v))
          for e, v in zip(effects, values, strict=True)}
    reference, _ = select_with_utility(unfiltered, xs, backbone)
    anchors = {reference.candidate_id,
               max(decisions, key=lambda d: (d.official_value, -d.candidate_id)).candidate_id}
    answer = []
    for d in decisions:
        components = {**d.components, "attribution_compatibility": 0.,
                      "learned_attribution_removed": 1.}
        if d.candidate_id in anchors and not d.accepted and _epistemic_only_rejection(d):
            components["baseline_preserving_epistemic_override"] = 1.
            d = replace(d, accepted=True, total_score=d.official_value, rejection_reasons=())
        answer.append(replace(d, components=components))
    return tuple(answer), xs


def select_chain(*, prior, learned_attribution, effects, plans, values, relation,
                 block_index, command, availability, backbone, frozen_residual,
                 conditional_model, mode, donor_attribution=None):
    """Independent whole current-block chain. Does not accept outcomes.

    Command evidence uses the existing private soft-score rule in ALL arms.
    It does not assert a contact failure or weaken an existing hard threshold.
    Dependency removal occurs before learned AND command deficit propagation.
    """
    if mode not in MODES or len(effects) != len(plans) or len(plans) != len(values):
        raise ValueError("Explicit aligned whole-chain mode required")
    if command is None or not command.valid or command.use_block_index != block_index:
        raise ValueError("Verified common preceding command evidence required")
    if len(availability) != 3 or any(type(x) is not bool for x in availability):
        raise ValueError("Actual recipient availability required")
    if mode == "shuffled" and donor_attribution is None:
        raise ValueError("Whole learned-output donor required, never original fallback")
    belief = deepcopy(prior)
    history_start = len(belief.history)
    no_learned = mode == "ranker_command"
    no_dag = mode == "learned_no_dag"
    graph = DependencyGraph(()) if no_dag else DEFAULT_GRAPH
    carrier = absent_carrier(command.source_block_id, availability)
    attribution = carrier if no_learned else recipient_attribution(
        donor_attribution if mode == "shuffled" else learned_attribution,
        source_block_id=command.source_block_id, availability=availability)
    if no_learned:
        decisions, xs = _no_learned_decisions(belief, carrier, effects, values, relation, backbone)
        zs = {e.candidate_id: np.zeros(10) for e in effects}
        reference, original_scores = select_with_utility(decisions, xs, backbone)
    else:
        decisions = prepare_arbitrated_decisions(belief=belief, attribution=attribution,
            block_index=block_index, effects=effects, values=values, relation=relation,
            backbone=backbone, graph=graph)
        xs = {e.candidate_id: candidate_backbone_features(e, float(v))
              for e, v in zip(effects, values, strict=True)}
        zs = {e.candidate_id: cause_residual_features(typed_gate_effect(e, relation),
              attribution, d) for e, d in zip(effects, decisions, strict=True)}
        reference, original_scores = select_evidence_arbitration(decisions=decisions,
            backbone_features=xs, cause_features=zs, backbone=backbone,
            residual=frozen_residual, attribution=attribution)
    vectors, traces = [], []
    # Without the learned module, the ranker still owns its ordinary parser
    # reliability channels. Do NOT make that baseline blind to visual forecasts
    # just because the zero-confidence carrier has no learned OBS factor.
    # This adapter only opens source availability; .50/.55 predicate-specific
    # parser thresholds still apply. It never updates/refreshed current belief.
    feature_attribution = attribution
    if no_learned:
        feature_attribution = replace(carrier, factor_states={**carrier.factor_states,
            ConsistencyFactor.OBSERVATION_RELIABLE.value:
                TriValue.TRUE if any(availability[:2]) else TriValue.UNKNOWN})
    for e, a in zip(effects, plans, strict=True):
        x, t = command_api.conditioned_features(prior=belief, attribution=feature_attribution,
            effect=e, actions=a, relation=relation, no_dag=no_dag,
            block_index=block_index, belief_already_updated=True,
            history_start=history_start, command=command)
        vectors.append(x); traces.append(t)
    # This model is frozen; these new controls receive no training/calibration.
    deltas = conditional_model.deltas(np.asarray(vectors))
    if np.asarray(deltas).shape != (len(effects),) or not np.isfinite(deltas).all():
        raise ValueError('Invalid frozen conditional corrections')
    if len(deltas) and np.ptp(deltas) > command_api.CAP + 1e-12:
        raise ValueError('Frozen pairwise correction cap exceeded')
    comparable = command_api.comparable_ids(decisions=decisions, backbone_features=xs,
        attribution=feature_attribution, traces=traces,
        reference_id=None if reference is None else reference.candidate_id)
    scores = {d.candidate_id: original_scores[d.candidate_id] + float(deltas[i])
              for i, d in enumerate(decisions) if d.accepted}
    selected = reference
    margin = max(.01, float(backbone.switch_margin))
    if reference is not None:
        rid = reference.candidate_id
        correction = {d.candidate_id: float(x) for d, x in zip(decisions, deltas, strict=True)}
        eligible = [d for d in decisions if d.accepted and d.candidate_id != rid
            and d.candidate_id in comparable and correction[d.candidate_id] > correction[rid] + 1e-12
            and scores[d.candidate_id] > scores[rid] + margin]
        if eligible:
            top = max(scores[d.candidate_id] for d in eligible)
            selected = max((d for d in eligible if top - scores[d.candidate_id] <= margin),
                           key=lambda d: (d.official_value, -d.candidate_id))
    fallback = None
    if selected is None:
        if no_learned:
            fallback = "reobserve" if not any(availability[:2]) else "requery"
        else:
            _, fallback = select_candidate(decisions, attribution)
    soft_belief = command_api.effective_belief(prior=belief, attribution=attribution,
        block_index=block_index, no_dag=no_dag, belief_already_updated=True, command=command)
    return dict(mode=mode, prior=prior.snapshot(), live_belief=belief.snapshot(),
        soft_belief=soft_belief.snapshot(), graph_edges=list(graph.edges),
        learned_input_removed=no_learned, learned_update_skipped=no_learned,
        learned_observation_refresh_skipped=no_learned, inherited_full_v8_anchor=False,
        baseline_parser_quality_retained=True,
        command=command.record(), attribution=attribution, decisions=decisions,
        features=np.asarray(vectors), traces=traces, base_scores=original_scores,
        deltas=deltas, scores=scores, comparable_ids=sorted(comparable),
        reference_id=None if reference is None else reference.candidate_id,
        selected_id=None if selected is None else selected.candidate_id, fallback=fallback,
        future_predictions_written_to_belief=False, newly_trained=False)

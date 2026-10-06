"""Recovery-conditioned ranking contracts, separate from frozen deployments.

Attribution -> source-aware belief -> DAG provenance -> ordered candidate
requirements -> ranking features is one function chain. Predicted effects are
forecasts, never observations or automatically restored physical predicates.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .belief import DEFAULT_GRAPH, DependencyGraph, update_belief
from .candidate_utility import select_with_utility
from .contracts import (
    AttributionOutput, BeliefFact, BeliefState, CandidateDecision, CandidateEffect, CoarseCause,
    ConsistencyFactor, PREDICATES, Stage, TriValue,
)
from .reranker import PREDICATE_HARD_THRESHOLDS
from .evidence_residual import candidate_backbone_features, typed_gate_effect
from .source_scoped_policy import scoped_attribution

SCHEMA = 'candidate_recovery_supervision_v1'
FEATURE_SCHEMA = 'cause_dependency_recovery_features_v3_libero_command'
ROUTES = tuple(c.value for c in CoarseCause)
ROUTE_ROOTS = {
    'normal': (),
    'visual_occlusion': ('target_visible', 'target_pose_current', 'receptacle_visible'),
    'object_shift': ('target_pose_current',),
    'execution_contact_deviation': ('execution_consistent',),
    'unknown': (),  # Candidate-specific epistemic risk, not invented physical falsehood.
}
FACT_METRICS = (
    'need', 'direct_need', 'propagated_need', 'required',
    'entry_observation_support', 'terminal_proposal',
    'ordered_forecast_support', 'unmet_requirement',
)
FACT_FEATURE_NAMES = tuple(f'{route}.{fact}.{metric}' for route in ROUTES
                           for fact in PREDICATES for metric in FACT_METRICS)
COMMAND_FEATURE_NAMES = ('has_close','has_upward','has_release','first_close_time',
    'first_upward_time','first_release_time','close_before_upward','release_after_close',
    'release_after_upward','close_fraction','open_upward_fraction','pre_close_upward_sum',
    'net_vertical_command','vertical_direction_switches')
VISUAL_EVIDENCE_KEYS = ('relation_score_after','relation_score_delta','relation_confidence',
    'cross_view_agreement','grasp_support_after','grasp_support_delta','contact_confidence',
    'target_gripper_distance_after','target_gripper_distance_before','target_displacement',
    'predicted_grasp_support','predicted_release_support','trajectory_path_efficiency',
    'trajectory_risk','terminal_progress_proxy_risk','articulated_operation')
FEATURE_NAMES = (FACT_FEATURE_NAMES
    + tuple('command.'+name for name in COMMAND_FEATURE_NAMES)
    + tuple(f'{route}.command.{name}' for route in ROUTES for name in COMMAND_FEATURE_NAMES)
    + tuple(f'{route}.visual.{key}.{part}' for route in ROUTES
            for key in VISUAL_EVIDENCE_KEYS for part in ('value','available')))


@dataclass(frozen=True)
class CurrentFactEvidence:
    predicate: str
    value: TriValue
    confidence: float
    evidence_id: str
    source: str = 'current_observation'
    view: str = 'fused'

    def __post_init__(self):
        if self.predicate not in PREDICATES or not isinstance(self.value, TriValue):
            raise ValueError('Invalid current predicate evidence')
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError('Invalid evidence confidence')
        if self.source not in ('current_observation', 'measured_execution') or not self.evidence_id:
            raise ValueError('Predictions, GT and unreferenced evidence cannot refresh current belief')
        if self.view not in ('primary','wrist','fused','record'):
            raise ValueError('Evidence must identify its sensor source')


@dataclass(frozen=True)
class ForecastFactEvent:
    predicate: str
    value: TriValue
    confidence: float
    available_from_step: int
    evidence_id: str
    source: str = 'predicted_endpoint'

    def __post_init__(self):
        if self.predicate not in PREDICATES or not isinstance(self.value, TriValue):
            raise ValueError('Invalid forecast predicate')
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError('Invalid forecast confidence')
        if not isinstance(self.available_from_step, int) or not 1 <= self.available_from_step <= 16:
            raise ValueError('Forecast event must follow an executed prefix')
        if self.source not in ('predicted_endpoint', 'predicted_intermediate') or not self.evidence_id:
            raise ValueError('Forecast source must be a referenced prediction')
        if self.source == 'predicted_endpoint' and self.available_from_step != 16:
            raise ValueError('Terminal prediction cannot claim pre-release or entry evidence')


@dataclass(frozen=True)
class RequirementUse:
    predicate: str
    at_step: int
    confidence: float

    def __post_init__(self):
        if self.predicate not in PREDICATES or not isinstance(self.at_step,int) or not 0 <= self.at_step < 16:
            raise ValueError('Invalid prerequisite identity or action time')
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError('Invalid prerequisite confidence')


@dataclass(frozen=True)
class RecoveryContext:
    attribution: AttributionOutput
    belief: BeliefState
    graph: DependencyGraph
    changes: tuple
    block_index: int


@dataclass(frozen=True)
class CandidateRecoveryTrace:
    candidate_id: int
    feature_names: tuple[str, ...]
    features: np.ndarray
    facts: Mapping[str, Mapping[str, object]]
    hard_reasons: tuple[str, ...]
    unresolved_requirements: tuple[str, ...]
    command_order: Mapping[str, object]


def prepare_recovery_context(*, prior: BeliefState, attribution: AttributionOutput,
                             observations: Sequence[CurrentFactEvidence] = (),
                             block_index: int, graph=DEFAULT_GRAPH) -> RecoveryContext:
    """Preserve caller state; never turn generic visual reliability into grasp."""
    belief = deepcopy(prior)
    for name, fact in list(belief.facts.items()):
        if fact.source == 'candidate_prediction' or 'initial_observation' in fact.evidence_ids:
            belief.facts[name] = BeliefFact(TriValue.UNKNOWN, 0., 'observation',
                                          block_index, ('unverified_assumption_masked',))
    # A directly observed false physical state cannot be erased by a prediction.
    witnessed = {n: deepcopy(f) for n, f in belief.facts.items()
                 if f.source == 'observation' and f.value is TriValue.FALSE}
    supported = {n: deepcopy(f) for n, f in belief.facts.items()
                 if f.source == 'observation' and f.value is not TriValue.UNKNOWN}
    start = len(belief.history)
    routed = scoped_attribution(attribution)
    if not attribution.evidence_quality.execution_reliable:
        states={f.value:routed.factor_state(f) for f in ConsistencyFactor}
        if states[ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value] is not TriValue.TRUE:
            states[ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value]=TriValue.UNKNOWN
            routed=replace(routed,factor_states=states)
    update_belief(belief, routed, block_index, graph=graph)
    factors = {f.value: f for f in ConsistencyFactor}
    for i in range(start, len(belief.history)):
        change = belief.history[i]
        if change.reason in factors:
            belief.history[i] = replace(change, new_confidence=min(
                change.new_confidence, attribution.factor_confidence(factors[change.reason])))
    for name, fact in list(belief.facts.items()):
        changes = [c for c in belief.history[start:] if c.predicate == name]
        if changes:
            belief.facts[name] = replace(fact, confidence=changes[-1].new_confidence)
            # Generic normal consistency must not erase the provenance of a
            # real observation or inflate its physical confidence.
            if name in supported and all(c.reason=='consistent_block' for c in changes):
                old=supported[name]
                belief.facts[name]=old
                for i in range(start,len(belief.history)):
                    if belief.history[i].predicate==name:
                        belief.history[i]=replace(belief.history[i],new_confidence=old.confidence)
    belief.facts.update(witnessed)
    if len({o.predicate for o in observations}) != len(observations):
        raise ValueError('Conflicting current predicate sources require explicit adjudication')
    for item in observations:
        if item.source == 'measured_execution' and item.predicate != 'execution_consistent':
            raise ValueError('Command equality cannot certify grasp, lift or placement')
        if item.source == 'measured_execution' and not attribution.evidence_quality.execution_reliable:
            continue
        if item.source == 'current_observation':
            q = attribution.evidence_quality
            usable={'primary':q.primary_reliable,'wrist':q.wrist_reliable,
                    'fused':q.primary_reliable and q.wrist_reliable,'record':False}[item.view]
            if q.cross_view_conflict or not usable:
                continue
        belief.facts[item.predicate] = BeliefFact(item.value, item.confidence,
            'observation', block_index, (item.evidence_id,))
    return RecoveryContext(routed, belief, graph, tuple(belief.history[start:]), block_index)


def command_requirements(effect: CandidateEffect, actions: np.ndarray):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (16, 7) or not np.isfinite(actions).all():
        raise ValueError('Recovery plan requires a complete finite 16x7 chunk')
    # LIBERO Panda measured action convention: -1 opens, +1 closes.
    # This new contract does not rewrite the historical parser/backbone.
    close = actions[:, 6] > .25
    opening = actions[:, 6] < -.25
    close_indices = np.flatnonzero(close)
    upward_indices = np.flatnonzero(actions[:, 2] > 0.)
    first_close = int(close_indices[0]) if close_indices.size else None
    first_up = int(upward_indices[0]) if upward_indices.size else None
    release = np.flatnonzero(opening & np.maximum.accumulate(close))
    first_release = int(release[0]) if release.size else None
    when = 0
    if effect.stage is Stage.GRASP and first_close is not None:
        when = first_close
    elif effect.stage is Stage.LIFT and first_up is not None:
        when = first_up
    uses = tuple(RequirementUse(name, first_release if name == 'place_ready'
        and first_release is not None else when, float(confidence))
        for name, confidence in effect.required_facts.items())
    return uses, dict(first_close=first_close, first_upward=first_up,
        first_release=first_release,
        closes_before_upward=first_close is not None and first_up is not None and first_close < first_up,
        commanded_close_is_not_grasp_evidence=True)


def planned_command_timing(actions: np.ndarray) -> np.ndarray:
    """LIBERO +close/-open plan timing; never measured grasp/contact state."""
    a=np.asarray(actions,dtype=np.float64)
    if a.shape!=(16,7) or not np.isfinite(a).all():
        raise ValueError('Invalid complete plan for command timing')
    close=a[:,6]>.25;opening=a[:,6]<-.25;up=a[:,2]>0
    ci=np.flatnonzero(close);ui=np.flatnonzero(up)
    ri=np.flatnonzero(opening&np.maximum.accumulate(close))
    c=int(ci[0]) if ci.size else None;u=int(ui[0]) if ui.size else None
    r=int(ri[0]) if ri.size else None
    signs=np.sign(a[:,2]);switches=np.sum(signs[1:]*signs[:-1]<0)
    return np.asarray([c is not None,u is not None,r is not None,
        (c/16 if c is not None else 0),(u/16 if u is not None else 0),
        (r/16 if r is not None else 0),c is not None and u is not None and c<u,
        c is not None and r is not None and c<r,
        u is not None and r is not None and u<r,float(close.mean()),
        float((opening&up).mean()),float(np.clip(a[:c if c is not None else 16,2].clip(0).sum()/16,0,1)),
        float(np.clip(a[:,2].sum()/16,-1,1)),float(switches/15)],dtype=np.float64)


def trace_candidate_recovery(context: RecoveryContext, effect: CandidateEffect,
                             actions: np.ndarray, *, forecasts: Sequence[ForecastFactEvent] = ()):
    """Outcome-free, temporally matched features for one candidate and one DAG.

Even an ordered forecast only supplies a soft feature. It never writes belief
or overrides a witnessed hard contradiction. Endpoint effects are at step16.
"""
    uses, order = command_requirements(effect, actions)
    events = tuple(ForecastFactEvent(name, value, float(confidence), 16,
        f'candidate:{effect.candidate_id}:endpoint:{name}')
        for name, (value, confidence) in effect.proposed_effects.items()) + tuple(forecasts)
    evidence_ids = [e.evidence_id for e in events]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError('Duplicate forecast evidence identity')
    required = {u.predicate: u for u in uses}
    details, hard, unresolved = {}, [], []
    for name in PREDICATES:
        fact = context.belief.facts[name]
        history = [c for c in context.changes if c.predicate == name]
        use = required.get(name)
        entry_support = fact.confidence if fact.value is TriValue.TRUE and fact.source == 'observation' else 0.
        proposed = max((e.confidence for e in events if e.predicate == name and e.value is TriValue.TRUE), default=0.)
        ordered = max((e.confidence for e in events if use and e.predicate == name
            and e.value is TriValue.TRUE and e.available_from_step <= use.at_step), default=0.)
        unmet = 0. if use is None else max(0., use.confidence-entry_support)
        threshold = PREDICATE_HARD_THRESHOLDS.get(name, .75)
        if use and fact.value is TriValue.FALSE and fact.confidence >= threshold:
            hard.append(f'{name}=false@{fact.confidence:.2f}')
        if use and entry_support < use.confidence:
            unresolved.append(name)
        details[name] = dict(value=fact.value.value, confidence=fact.confidence,
            source=fact.source, evidence_ids=list(fact.evidence_ids),
            required_at_step=None if use is None else use.at_step,
            paths=[list(c.propagation_path) for c in history],
            metrics=dict(need=float(fact.value is not TriValue.TRUE),
                direct_need=float(any(c.direct and c.new_value is not TriValue.TRUE for c in history)),
                propagated_need=float(any(not c.direct and c.new_value is not TriValue.TRUE for c in history)),
                required=0. if use is None else use.confidence,
                entry_observation_support=entry_support, terminal_proposal=proposed,
                ordered_forecast_support=ordered, unmet_requirement=unmet))
    a = context.attribution
    # Multi-factor interactions survive coarse projection; world/execution may
    # coexist. Unknown preserves epistemic risk without inventing known causes.
    visible = a.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is TriValue.TRUE
    resolved = a.factor_state(ConsistencyFactor.CAUSE_RESOLVED) is TriValue.TRUE
    routes = dict(a.class_probs)
    if not visible:
        routes['visual_occlusion'] = max(routes['visual_occlusion'], 1-a.factor(ConsistencyFactor.OBSERVATION_RELIABLE))
    routes['object_shift'] = (max(routes['object_shift'], 1-a.factor(ConsistencyFactor.WORLD_STATE_CONSISTENT))
        if visible and resolved and a.factor_state(ConsistencyFactor.WORLD_STATE_CONSISTENT) is TriValue.FALSE else 0.)
    routes['execution_contact_deviation'] = (max(routes['execution_contact_deviation'], 1-a.factor(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT))
        if a.evidence_quality.execution_reliable and a.factor_state(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT) is TriValue.FALSE else 0.)
    routes['unknown'] = max(routes['unknown'], float(not resolved))
    vector = []
    for route in ROUTES:
        roots = ROUTE_ROOTS[route]
        scoped = set(roots)
        # Missing visibility does not invalidate established physical holding.
        if route in ('object_shift', 'execution_contact_deviation'):
            for root in roots:
                scoped.update(context.graph.descendants_with_paths(root))
        if route in ('normal', 'unknown'):
            scoped = set(required)
        for name in PREDICATES:
            weight = routes[route] if name in scoped else 0.
            vector.extend(weight*details[name]['metrics'][m] for m in FACT_METRICS)
    timing=planned_command_timing(actions)
    vector.extend(timing)
    for route in ROUTES:vector.extend(routes[route]*timing)
    for route in ROUTES:
        for key in VISUAL_EVIDENCE_KEYS:
            available=key in effect.evidence
            value=float(effect.evidence.get(key,0.))
            if not np.isfinite(value):raise ValueError('Nonfinite candidate visual proxy')
            vector.extend((routes[route]*float(np.clip(value,-1,1)),routes[route]*float(available)))
    hard.extend(f'candidate_evidence:{reason}' for reason, confidence in effect.hard_violations.items()
                if confidence >= .5)
    return CandidateRecoveryTrace(effect.candidate_id, FEATURE_NAMES,
        np.asarray(vector, dtype=np.float64), details, tuple(hard), tuple(unresolved), order)


def require_recovery_training_ready(audit: Mapping):
    if audit.get('schema') != SCHEMA or not audit.get('recovery_training_ready', False):
        raise RuntimeError('Recovery V9 training blocked: actual block evidence or supervised coverage is missing')
    if audit.get('group_leakage', True) or audit.get('deployment_gt_leakage', True):
        raise RuntimeError('Recovery V9 training blocked: provenance/leakage audit failed')


@dataclass(frozen=True)
class RecoverySelection:
    selected_candidate_id: int | None
    fallback: str | None
    belief_snapshot: Mapping[str, object]
    traces: tuple[CandidateRecoveryTrace, ...]
    scores: Mapping[int, float]


def select_recovery_candidate(*, prior: BeliefState, attribution: AttributionOutput,
                              observations: Sequence[CurrentFactEvidence], block_index: int,
                              effects: Sequence[CandidateEffect], actions: Sequence[np.ndarray],
                              values: Sequence[float], relation: str, backbone,
                              recovery_model=None, training_audit=None,
                              graph=DEFAULT_GRAPH,
                              forecasts: Sequence[Sequence[ForecastFactEvent]] | None = None) -> RecoverySelection:
    """Single outcome-free inference entry; no step can bypass shared context.

    A learned model can be supplied only after its supervision audit passes.
    In cold-start/no-model mode the outcome is an explicitly marked baseline
    control, not evidence that a recovery model has been trained.
    """
    if not effects or len(effects)!=len(actions) or len(effects)!=len(values):
        raise ValueError('Recovery candidates must be aligned and nonempty')
    if len({e.candidate_id for e in effects})!=len(effects):
        raise ValueError('Duplicate candidate identity')
    if not np.isfinite(values).all() or any(not 0 <= v <= 1 for v in values):
        raise ValueError('Official candidate values must be finite in [0,1]')
    if recovery_model is not None:
        require_recovery_training_ready(training_audit or {})
    if forecasts is None:forecasts=[() for _ in effects]
    if len(forecasts)!=len(effects):raise ValueError('Candidate forecast identity mismatch')
    context=prepare_recovery_context(prior=prior,attribution=attribution,
        observations=observations,block_index=block_index,graph=graph)
    traces=tuple(trace_candidate_recovery(context,typed_gate_effect(e,relation),a,forecasts=f)
                 for e,a,f in zip(effects,actions,forecasts,strict=True))
    ds=[];selection_features={};audited_traces=[]
    for e,t,value in zip(effects,traces,values,strict=True):
        x=candidate_backbone_features(e,value)
        # Unknown safety prerequisites do not become admissible merely because
        # attribution propagation decayed their confidence. Terminal forecasts
        # cannot certify them. No outcome can override this gate.
        critical={Stage.GRASP:{'target_pose_current','target_reachable'},
                  Stage.LIFT:{'grasped'},Stage.TRANSPORT:{'grasped','lifted'},
                  Stage.PLACE:{'lifted'}}.get(e.stage,set())
        reasons=list(t.hard_reasons)
        reasons.extend(f'{name}=unknown_requires_current_evidence' for name in t.unresolved_requirements
                       if name in critical and t.facts[name]['value']=='unknown')
        audited_traces.append(replace(t, hard_reasons=tuple(reasons)))
        accepted=not reasons
        total=float(backbone.score(x))
        if recovery_model is not None:
            correction=float(recovery_model.score(t.features))
            if not np.isfinite(correction):raise ValueError('Nonfinite recovery model score')
            total+=correction
        if not np.isfinite(total):raise ValueError('Nonfinite composite candidate score')
        ds.append(CandidateDecision(e.candidate_id,accepted,float(value),total if accepted else None,
            tuple(reasons),{'no_recovery_model_control':float(recovery_model is None)}))
        selection_features[e.candidate_id]=np.asarray([total],dtype=np.float64)
    class SharedSelectionScore:
        # Both arms use the frozen wrapper's exact margin, value tie break,
        # Pareto guard and rejection handling. Only the precomputed score's
        # additive recovery term differs; a zero head is the identical control.
        switch_margin=backbone.switch_margin
        @staticmethod
        def score(precomputed):
            return float(precomputed[0])
    chosen,scores=select_with_utility(ds,selection_features,SharedSelectionScore())
    fallback=None
    if chosen is None:
        fallback='reobserve' if context.attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is not TriValue.TRUE else 'requery'
    return RecoverySelection(None if chosen is None else chosen.candidate_id,fallback,
        context.belief.snapshot(),tuple(audited_traces),scores)

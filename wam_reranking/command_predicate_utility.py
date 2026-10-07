"""PRE-only command provenance and predicate-scoped terminal utility channel.

This is NOT a physical recovery certificate or a retrained attribution head.
Frozen V8 decisions/gates remain the reference. Terminal outcomes are accepted
only by the separate supervision audit, never by these feature/admission APIs.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import numpy as np

from . import persistent_conditioned_utility as old
from .belief import DEFAULT_GRAPH, DependencyGraph, update_belief
from .contracts import BeliefChange, BeliefFact, ConsistencyFactor, PREDICATES, TriValue
from .evidence_arbitration import affirmative_improvement, evidence_route, select_evidence_arbitration
from .source_scoped_policy import scoped_attribution

SCHEMA = "command_provenance_predicate_scoped_terminal_utility_v1"
FEATURE_NAMES = old.FEATURE_NAMES
COMMAND_THRESHOLD = 1e-6  # Existing frozen applied-requested hard rule.
CONTACT_THRESHOLD = .55  # Existing frozen contact evidence contract.
VISUAL_THRESHOLD = .5    # Existing residual visual feature threshold.
CAP = old.CAP
COMMAND_REASON = "measured_command_deviation"
REASON_ROUTE = {**old.REASON_ROUTE, COMMAND_REASON: "execution_contact_deviation"}
CONTACT_PREDICATES = frozenset(("target_reachable", "grasped", "lifted"))


@dataclass(frozen=True)
class CommandEvidence:
    source_block_id: str
    source_block_index: int
    use_block_index: int
    endpoint_policy_step: int
    valid: bool
    max_abs: float | None
    changed_steps: tuple[int, ...]
    evidence_ids: tuple[str, ...]
    invalid_reasons: tuple[str, ...] = ()

    def __post_init__(self):
        if self.valid and (self.max_abs is None or not np.isfinite(self.max_abs) or self.max_abs < 0
                           or self.use_block_index != self.source_block_index + 1 or self.invalid_reasons):
            raise ValueError("Invalid typed command evidence")

    @property
    def deviated(self):
        return self.valid and self.max_abs is not None and self.max_abs > COMMAND_THRESHOLD

    def record(self):
        return {**asdict(self), "threshold": COMMAND_THRESHOLD,
                "deviated": self.deviated, "physical_contact_failure_certified": False}


def command_evidence(*, requested, applied, record_available, source_block_id,
                     source_block_index, use_block_index, policy_step_t,
                     endpoint_policy_step, observation_policy_step, executed_steps,
                     alignment_valid, evidence_ids=()):
    """Whitelist only preceding complete-block deployment records/counters."""
    a, b = np.asarray(requested), np.asarray(applied)
    reasons = []
    if not record_available: reasons.append("record_unavailable")
    arrays_ok = (a.shape == b.shape == (16, 7) and
                 np.isfinite(a).all() and np.isfinite(b).all())
    if not arrays_ok: reasons.append("incomplete_or_nonfinite_commands")
    if not (alignment_valid and executed_steps == 16 and
            endpoint_policy_step == observation_policy_step == policy_step_t + 16 and
            use_block_index == source_block_index + 1):
        reasons.append("source_use_time_mismatch")
    valid = not reasons
    diff = np.abs(b.astype(np.float64) - a.astype(np.float64)) if arrays_ok else None
    return CommandEvidence(source_block_id, source_block_index, use_block_index,
        endpoint_policy_step, valid, float(diff.max()) if valid else None,
        tuple(np.flatnonzero(np.max(diff, axis=1) > COMMAND_THRESHOLD).tolist()) if valid else (),
        tuple(evidence_ids), tuple(reasons))


def effective_belief(*, prior, attribution, block_index, no_dag=False,
                     belief_already_updated=False, command=None):
    """Private soft-score snapshot; never mutate the frozen live reference.

    A measured mismatch invalidates command consistency, not contact truth.
    Its dependent facts become UNKNOWN, not FALSE. Fresh independently observed
    physical facts are retained. Future candidate predictions establish none.
    """
    b = deepcopy(prior)
    for p, f in list(b.facts.items()):
        if f.source == "candidate_prediction" or "initial_observation" in f.evidence_ids:
            b.facts[p] = BeliefFact(TriValue.UNKNOWN, 0., "observation", block_index,
                                   ("unverified_assumption_masked",))
    witnessed = {p: deepcopy(f) for p, f in b.facts.items()
                 if f.source == "observation" and f.value is TriValue.FALSE
                 and p not in {"target_visible", "target_pose_current"}}
    fresh_physical = {p: deepcopy(f) for p, f in b.facts.items()
        if p not in {"execution_consistent", "target_visible", "target_pose_current"}
        and f.source == "observation" and f.updated_at_block >= block_index
        and f.confidence > 0 and f.value is not TriValue.UNKNOWN}
    graph = DependencyGraph(()) if no_dag else DEFAULT_GRAPH
    if not belief_already_updated:
        update_belief(b, scoped_attribution(attribution), block_index, graph=graph)
    b.facts.update(witnessed)
    b.facts.update(fresh_physical)
    if command is not None and command.valid and command.use_block_index != block_index:
        raise ValueError("Command evidence used at the wrong decision block")
    if command is not None and command.deviated and attribution.evidence_quality.execution_reliable:
        paths = {"execution_consistent": ("execution_consistent",),
                 **graph.descendants_with_paths("execution_consistent")}
        for p, path in paths.items():
            f = b.facts[p]
            # A new witnessed grasp/lift does not depend on the old command
            # assumption. Keep the observation, including witnessed FALSE.
            witnessed_physical = (p != "execution_consistent" and f.source == "observation"
                and f.confidence > 0 and (f.value is TriValue.FALSE or
                    (f.value is TriValue.TRUE and f.updated_at_block >= block_index)))
            if witnessed_physical: continue
            value = TriValue.FALSE if p == "execution_consistent" else TriValue.UNKNOWN
            strength = max(.05, .85 ** (len(path) - 1))
            evidence_id = command.source_block_id + ":applied_minus_requested"
            b.facts[p] = BeliefFact(value, strength, "observation" if len(path) == 1 else "attribution",
                                    block_index, (evidence_id,))
            b.history.append(BeliefChange(p, f.value, f.confidence, value, strength,
                COMMAND_REASON, len(path) == 1, path, evidence_id))
    return b


def predicate_masks(attribution, effect):
    q, e = attribution.evidence_quality, effect.evidence
    visual_source = ((q.primary_reliable or q.wrist_reliable) and not q.cross_view_conflict
        and attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is TriValue.TRUE)
    clip = lambda key: float(np.clip(e.get(key, 0.), 0., 1.))
    relation_quality = min(effect.confidence, clip("relation_confidence"), clip("cross_view_agreement"))
    # This reliability is from the frozen parser's subject/gripper channel;
    # DO NOT replace it by the subject/container cross-view agreement.
    contact_quality = min(effect.confidence, clip("current_contact_reliability"))
    visibility_quality = min(effect.confidence, clip("visual_support"))
    quality, usable, sources = {}, {}, {}
    for p in PREDICATES:
        if p == "execution_consistent":
            quality[p], usable[p], sources[p] = 1., bool(q.execution_reliable), "execution_record"
        elif p in CONTACT_PREDICATES:
            quality[p], usable[p], sources[p] = contact_quality, bool(visual_source and contact_quality >= CONTACT_THRESHOLD), "subject_gripper"
        elif p == "target_visible":
            quality[p], usable[p], sources[p] = visibility_quality, bool(visual_source and visibility_quality >= VISUAL_THRESHOLD), "visibility"
        else:
            quality[p], usable[p], sources[p] = relation_quality, bool(visual_source and relation_quality >= VISUAL_THRESHOLD), "subject_anchor_relation"
    return quality, usable, sources


def conditioned_features(*, prior, attribution, effect, actions, relation="inside", no_dag=False,
                         block_index=3, belief_already_updated=False, history_start=None, command=None):
    a = old._finite_plan(actions)
    start = len(prior.history) if history_start is None else history_start
    b = effective_belief(prior=prior, attribution=attribution, block_index=block_index,
        no_dag=no_dag, belief_already_updated=belief_already_updated, command=command)
    quality, usable, source = predicate_masks(attribution, effect)
    paths, deficits = {}, {}
    for p, f in b.facts.items():
        if p not in PREDICATES or f.value is TriValue.TRUE or f.confidence <= 0: continue
        origins = [(i, c) for i, c in enumerate(b.history) if c.predicate == p
                   and c.new_value is not TriValue.TRUE and c.reason in REASON_ROUTE]
        if not origins: continue
        i, c = origins[-1]
        if no_dag and len(c.propagation_path) > 1: continue
        r, strength = REASON_ROUTE[c.reason], min(f.confidence, c.new_confidence)
        if i >= start and c.reason in {x.value for x in ConsistencyFactor}:
            strength = min(strength, attribution.factor_confidence(ConsistencyFactor(c.reason)))
        deficits[(r, p, f.value.value)] = float(strength)
        paths[p] = {"route": r, "state": f.value.value, "path": list(c.propagation_path),
                    "historical": i < start, "strength": float(strength), "evidence_id": c.evidence_id,
                    "origin": c.reason}
    e = effect.evidence
    clip = lambda key, default=0.: float(np.clip(e.get(key, default), 0., 1.))
    risk, rq, contact, grasp = clip("trajectory_risk"), clip("relation_confidence"), clip("contact_confidence"), clip("grasp_support_after")
    # Retain the previous score proxy unchanged in this two-interface repair.
    # This plan-shape proxy is not a witnessed grasp/lift certificate.
    close_up = float(np.any((a[:, 6] > .25) & (a[:, 2] > 0.)))
    length = float(np.linalg.norm(a[:, :3], axis=1).sum())
    efficiency = float(np.linalg.norm(a[:, :3].sum(axis=0)) / max(length, 1e-12))
    jerk = float(np.mean(np.linalg.norm(np.diff(a[:, :6], axis=0), axis=1)))
    command_risk = float(jerk / (1. + jerk))
    proxy = {"target_visible": quality["target_visible"],
        "target_pose_current": rq * clip("target_displacement"),
        "target_reachable": contact * float(np.clip(1. - e.get("target_gripper_distance_after", 1.), 0., 1.)),
        "grasped": contact * grasp, "lifted": contact * grasp * close_up,
        "receptacle_visible": rq * quality["receptacle_visible"],
        "place_ready": rq * float(np.clip(e.get("relation_score_after", 0.), -1., 1.)),
        "placed": rq * float(np.clip(e.get("relation_score_delta", 0.), -1., 1.)),
        "execution_consistent": efficiency - command_risk}
    required, proposals = dict(effect.required_facts), dict(effect.proposed_effects)
    if relation in {"open", "closed", "articulated"}:
        required, proposals = {"target_visible": .5, "target_pose_current": .6}, {}
    vector = []
    for r in old.ROUTES:
        for p in PREDICATES:
            command_channel = r == "execution_contact_deviation" and p == "execution_consistent"
            allowed = usable[p] and (p != "execution_consistent" or command_channel)
            uncertainty = command_risk if command_channel else float(np.clip(1. - quality[p] + risk, 0., 1.))
            for s in old.STATES:
                w = deficits.get((r, p, s), 0.) if allowed else 0.
                req = max(float(required.get(p, 0.)), 1. if command_channel else 0.)
                proposed = proposals.get(p, (TriValue.UNKNOWN, 0.))
                support = float(proposed[1]) if proposed[0] is TriValue.TRUE else 0.
                if r == "unknown": vector.extend((w * req * uncertainty, 0., -w * req * uncertainty))
                else: vector.extend((w * req * uncertainty,
                    0. if command_channel else w * support * quality[p], w * proxy[p] * quality[p]))
    x = np.asarray(vector)
    if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all(): raise ValueError("Invalid PRE features")
    return x, {"deficits": paths, "predicate_usable": usable, "predicate_quality": quality,
        "predicate_source": source, "visual_usable": any(usable[p] for p in PREDICATES if p != "execution_consistent"),
        "execution_usable": usable["execution_consistent"], "command_efficiency": efficiency,
        "command_risk": command_risk, "command_evidence": None if command is None else command.record(),
        "endpoint_is_forecast_only": True, "live_belief_mutated": False, "no_dag": bool(no_dag)}


def comparable_ids(*, decisions, backbone_features, attribution, traces, reference_id):
    if reference_id is None: return set()
    route = evidence_route(attribution)
    feasible = {d.candidate_id: d for d in decisions if d.accepted}
    value_route = any(d.components.get("preserve_official_value", 0.) for d in feasible.values())
    answer = {reference_id}
    for cid in feasible:
        if cid == reference_id: continue
        t, ref = traces[cid], traces[reference_id]
        # Compare only an actually missing predicate's two usable channels.
        visual_ok = route.visual_reliable and any(
            p != "execution_consistent" and t["predicate_usable"][p] and ref["predicate_usable"][p]
            for p in t["deficits"])
        if visual_ok and value_route:
            visual_ok = route.comparison_allowed and affirmative_improvement(backbone_features[cid], backbone_features[reference_id])
        deficit = t["deficits"].get("execution_consistent", {})
        execution_ok = (t["execution_usable"] and ref["execution_usable"]
            and deficit.get("route") == "execution_contact_deviation" and deficit.get("state") == "false"
            and t["command_efficiency"] > ref["command_efficiency"] + 1e-12
            and t["command_risk"] <= ref["command_risk"] + 1e-12)
        if visual_ok or execution_ok: answer.add(cid)
    return answer


def select_conditioned(*, decisions, backbone_features, cause_features, backbone,
                       frozen_residual, attribution, deltas, traces):
    reference, original = select_evidence_arbitration(decisions=decisions, backbone_features=backbone_features,
        cause_features=cause_features, backbone=backbone, residual=frozen_residual, attribution=attribution)
    ids, delta = [d.candidate_id for d in decisions], np.asarray(deltas, dtype=float)
    if delta.shape != (len(ids),) or not np.isfinite(delta).all() or len(set(ids)) != len(ids): raise ValueError("Invalid deltas")
    if delta.size and np.ptp(delta) > CAP + 1e-12:
        raise ValueError("Corrections exceed the frozen pairwise cap")
    audit = {"hard_gates_unchanged": True, "conditional_switch": False}
    if reference is None or not np.any(delta): return reference, original, {**audit, "reason": "exact_zero_or_fallback"}
    correction = dict(zip(ids, delta)); rid = reference.candidate_id
    comparable = comparable_ids(decisions=decisions, backbone_features=backbone_features,
        attribution=attribution, traces=traces, reference_id=rid)
    scores = {d.candidate_id: original[d.candidate_id] + correction[d.candidate_id] for d in decisions if d.accepted}
    margin = max(.01, float(backbone.switch_margin))
    eligible = [d for d in decisions if d.accepted and d.candidate_id != rid and d.candidate_id in comparable
        and correction[d.candidate_id] > correction[rid] + 1e-12 and scores[d.candidate_id] > scores[rid] + margin]
    if not eligible: return reference, scores, {**audit, "reason": "no_admissible_margin_clearance"}
    top = max(scores[d.candidate_id] for d in eligible)
    chosen = max((d for d in eligible if top - scores[d.candidate_id] <= margin), key=lambda d: (d.official_value, -d.candidate_id))
    return replace(chosen, components={**chosen.components, "conditional_utility": correction[chosen.candidate_id]}), scores, {
        **audit, "reason": "source_admitted_positive_correction", "conditional_switch": True}

"""Bounded utility channel, with persistent deficits and source-specific masks.

This predicts terminal choice utility, NOT physical recovery probabilities.
Forecasts never write current belief or override a frozen hard rejection.
"""
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from .belief import DEFAULT_GRAPH, DependencyGraph, update_belief
from .contracts import BeliefFact, ConsistencyFactor, PREDICATES, TriValue
from .evidence_arbitration import affirmative_improvement, evidence_route, select_evidence_arbitration
from .source_scoped_policy import scoped_attribution
from .terminal_conditioned_residual import CAP, GAIN_GRID, ROOTS, ROUTES, _finite_plan

SCHEMA = "persistent_source_conditioned_terminal_utility_v10_v2"
STATES = ("false", "unknown")
METRICS = ("required_risk", "endpoint_forecast", "predicted_effect")
FEATURE_NAMES = tuple(f"{r}.{p}.{s}.{m}" for r in ROUTES for p in PREDICATES
                      for s in STATES for m in METRICS)
REASON_ROUTE = {"observation_reliable": "visual_occlusion",
                "world_state_consistent": "object_shift",
                "execution_contact_consistent": "execution_contact_deviation",
                "cause_resolved": "unknown", "task_stage_consistent": "unknown"}
SCORE_UNIT = .01  # Original complete V8 switching unit, not a changed threshold.


def conditioned_features(*, prior, attribution, effect, actions, relation="inside",
                         no_dag=False, block_index=3, belief_already_updated=False,
                         history_start=None):
    a = _finite_plan(actions)
    b = deepcopy(prior)
    for p, f in list(b.facts.items()):
        if f.source == "candidate_prediction" or "initial_observation" in f.evidence_ids:
            b.facts[p] = BeliefFact(TriValue.UNKNOWN, 0., "observation", block_index,
                                   ("unverified_assumption_masked",))
    graph = DependencyGraph(()) if no_dag else DEFAULT_GRAPH
    start = len(b.history) if history_start is None else history_start
    routed = scoped_attribution(attribution)
    # A directly witnessed physical FALSE must not be overwritten by learned
    # uncertainty. This is the same provenance distinction as the V8 gate.
    witnessed = {p: deepcopy(f) for p, f in b.facts.items()
                 if f.source == "observation" and f.value is TriValue.FALSE
                 and p not in {"target_visible", "target_pose_current"}}
    if not belief_already_updated:
        update_belief(b, routed, block_index, graph=graph)
    b.facts.update(witnessed)
    q = attribution.evidence_quality
    visual = ((q.primary_reliable or q.wrist_reliable) and not q.cross_view_conflict
              and attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is TriValue.TRUE)
    e = effect.evidence
    cq = min(float(effect.confidence), float(np.clip(e.get("cross_view_agreement", 0.), 0., 1.)))
    visual_candidate = visual and cq >= .5
    execution = bool(q.execution_reliable)
    paths, deficits = {}, {}
    for p, f in b.facts.items():
        if p not in PREDICATES or f.value is TriValue.TRUE or f.confidence <= 0:
            continue
        # A normal update may preserve a FALSE but change its reason to
        # consistent_block. Retain its last actual invalidation, not that reason.
        origins = [(i, c) for i, c in enumerate(b.history) if c.predicate == p
                   and c.new_value is not TriValue.TRUE and c.reason in REASON_ROUTE]
        if not origins:
            continue  # Unknown initial fact is not a caused physical deficit.
        i, c = origins[-1]
        if no_dag and len(c.propagation_path) > 1:
            continue
        r = REASON_ROUTE[c.reason]
        strength = min(f.confidence, c.new_confidence)
        if i >= start and c.reason in {x.value for x in ConsistencyFactor}:
            strength = min(strength, attribution.factor_confidence(ConsistencyFactor(c.reason)))
        deficits[(r, p, f.value.value)] = float(strength)
        paths[p] = {"route": r, "state": f.value.value,
                    "path": list(c.propagation_path), "historical": i < start,
                    "strength": float(strength), "evidence_id": c.evidence_id}
    close = a[:, 6] > .25
    close_up = float(np.any(close & (a[:, 2] > 0.)))
    risk = float(np.clip(e.get("trajectory_risk", 0.), 0., 1.))
    rq = float(np.clip(e.get("relation_confidence", 0.), 0., 1.))
    contact = float(np.clip(e.get("contact_confidence", 0.), 0., 1.))
    grasp = float(np.clip(e.get("grasp_support_after", 0.), 0., 1.))
    # This is an action-plan shape proxy, NOT evidence that execution succeeded.
    steps = np.linalg.norm(a[:, :3], axis=1)
    length = float(steps.sum())
    efficiency = float(np.linalg.norm(a[:, :3].sum(axis=0)) / max(length, 1e-12))
    jerk = float(np.mean(np.linalg.norm(np.diff(a[:, :6], axis=0), axis=1)))
    command_risk = float(np.clip(jerk / (1. + jerk), 0., 1.))
    proxy = {"target_visible": cq, "target_pose_current": rq * float(np.clip(e.get("target_displacement", 0.), 0., 1.)),
             "target_reachable": contact * float(np.clip(1. - e.get("target_gripper_distance_after", 1.), 0., 1.)),
             "grasped": contact * grasp, "lifted": contact * grasp * close_up,
             "receptacle_visible": rq * cq,
             "place_ready": rq * float(np.clip(e.get("relation_score_after", 0.), -1., 1.)),
             "placed": rq * float(np.clip(e.get("relation_score_delta", 0.), -1., 1.)),
             "execution_consistent": efficiency - command_risk}
    required, proposals = dict(effect.required_facts), dict(effect.proposed_effects)
    if relation in {"open", "closed", "articulated"}:
        required, proposals = {"target_visible": .5, "target_pose_current": .6}, {}
    vector = []
    for r in ROUTES:
        for p in PREDICATES:
            # Command-record reliability gates command consistency only. It
            # cannot establish unseen grasp/contact despite a close command.
            command_channel = r == "execution_contact_deviation" and p == "execution_consistent"
            usable = execution if command_channel else visual_candidate
            quality = 1. if command_channel else cq
            uncertainty = command_risk if command_channel else float(np.clip(1. - cq + risk, 0., 1.))
            for s in STATES:
                w = deficits.get((r, p, s), 0.) if usable else 0.
                req = float(required.get(p, 0.))
                if command_channel:
                    req = max(req, 1.)  # Every executable action needs a reliable command path.
                f = proposals.get(p, (TriValue.UNKNOWN, 0.))
                support = float(f[1]) if f[0] is TriValue.TRUE else 0.
                if r == "unknown":
                    vector.extend((w * req * uncertainty, 0., -w * req * uncertainty))
                else:
                    vector.extend((w * req * uncertainty,
                                   0. if command_channel else w * support * quality,
                                   w * proxy[p] * quality))
    x = np.asarray(vector)
    if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
        raise ValueError("Invalid PRE-only source-conditioned features")
    return x, {"deficits": paths, "visual_usable": bool(visual_candidate),
               "execution_usable": execution, "command_efficiency": efficiency,
               "command_risk": command_risk, "endpoint_is_forecast_only": True,
               "no_dag": bool(no_dag), "live_belief_mutated": False,
               "belief_already_updated": bool(belief_already_updated)}


def comparable_ids(*, decisions, backbone_features, attribution, traces, reference_id):
    """Outcome-independent admission shared by fitting and deployment."""
    if reference_id is None:
        return set()
    route = evidence_route(attribution)
    feasible = {d.candidate_id: d for d in decisions if d.accepted}
    value_route = any(d.components.get("preserve_official_value", 0.) for d in feasible.values())
    answer = {reference_id}
    for cid, d in feasible.items():
        if cid == reference_id:
            continue
        visual_ok = route.visual_reliable and traces[cid]["visual_usable"] and traces[reference_id]["visual_usable"]
        if visual_ok and value_route:
            visual_ok = route.comparison_allowed and affirmative_improvement(
                backbone_features[cid], backbone_features[reference_id])
        # Independent execution source may remain informative when vision is
        # closed, but must have a supported execution deficit and positive plan
        # evidence. Never turn a coarse/factor disagreement into physical FALSE.
        t, ref = traces[cid], traces[reference_id]
        deficit = t["deficits"].get("execution_consistent", {})
        execution_ok = (t["execution_usable"] and ref["execution_usable"]
                        and deficit.get("route") == "execution_contact_deviation"
                        and deficit.get("state") == "false"
                        and t["command_efficiency"] > ref["command_efficiency"] + 1e-12
                        and t["command_risk"] <= ref["command_risk"] + 1e-12)
        if visual_ok or execution_ok:
            answer.add(cid)
    return answer


def select_conditioned(*, decisions, backbone_features, cause_features, backbone,
                       frozen_residual, attribution, deltas, traces):
    reference, original = select_evidence_arbitration(decisions=decisions,
        backbone_features=backbone_features, cause_features=cause_features,
        backbone=backbone, residual=frozen_residual, attribution=attribution)
    ids = [d.candidate_id for d in decisions]
    delta = np.asarray(deltas, dtype=float)
    if delta.shape != (len(ids),) or not np.isfinite(delta).all() or len(set(ids)) != len(ids):
        raise ValueError("Invalid complete candidate corrections")
    audit = {"reference": None if reference is None else reference.candidate_id,
             "hard_gates_unchanged": True, "conditional_switch": False}
    if reference is None or not np.any(delta):
        return reference, original, {**audit, "reason": "exact_zero_or_fallback"}
    corrections = dict(zip(ids, delta))
    comparable = comparable_ids(decisions=decisions, backbone_features=backbone_features,
        attribution=attribution, traces=traces, reference_id=reference.candidate_id)
    scores = {d.candidate_id: original[d.candidate_id] + corrections[d.candidate_id]
              for d in decisions if d.accepted}
    rid = reference.candidate_id
    margin = max(.01, float(backbone.switch_margin))
    # Admit only candidates that actually clear the V8 anchor. Do not apply a
    # top tie rule first and veto its tied anchor while ignoring a valid switch.
    eligible = [d for d in decisions if d.candidate_id in comparable and d.candidate_id != rid
                and corrections[d.candidate_id] > corrections[rid] + 1e-12
                and scores[d.candidate_id] > scores[rid] + margin]
    if not eligible:
        return reference, scores, {**audit, "reason": "no_admissible_margin_clearance",
                                    "comparable": sorted(comparable), "margin": margin}
    top = max(scores[d.candidate_id] for d in eligible)
    tied = [d for d in eligible if top - scores[d.candidate_id] <= margin]
    chosen = max(tied, key=lambda d: (d.official_value, -d.candidate_id))
    return replace(chosen, components={**chosen.components, "conditional_utility": corrections[chosen.candidate_id]}), scores, {
        **audit, "reason": "source_admitted_positive_correction", "conditional_switch": True,
        "comparable": sorted(comparable), "margin": margin}


@dataclass(frozen=True)
class ConditionalUtility:
    scale: np.ndarray
    weights: np.ndarray
    support: np.ndarray
    gain: float = 0.

    def normalized(self, matrix):
        x = np.asarray(matrix, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("Invalid input matrix")
        # Train-unseen dimensions contribute zero, never a learnt confidence.
        # Train-supported dimensions clamp to their frozen envelope; one novel
        # coordinate no longer silently disables all independent source channels.
        bounded = np.clip(x, -self.support, self.support)
        return np.clip((bounded - bounded.mean(axis=0)) / self.scale, -4., 4.)

    def deltas(self, matrix):
        z = self.normalized(matrix)
        raw = CAP * np.tanh(np.clip(z @ self.weights, -30., 30.)) / 2
        return self.gain * (raw - raw.mean())

    def record(self):
        return {"schema": SCHEMA, "feature_names": list(FEATURE_NAMES), "scale": self.scale.tolist(),
                "weights": self.weights.tolist(), "support": self.support.tolist(), "gain": self.gain,
                "cap": CAP, "score_unit": SCORE_UNIT, "physical_recovery_head_trained": False,
                "support_policy": "train_envelope_clip; unseen_dimensions_zero"}


def preference_pairs(train_pools):
    if not train_pools or any(p["split"] != "train" for p in train_pools):
        raise ValueError("Explicit TRAIN pools only")
    pairs, audit = [], {"inactive_features": 0, "outside_correction_range": 0,
                        "route_incomparable": 0, "success": 0, "successful_cost": 0}
    for pool, p in enumerate(train_pools):
        rid = p["baseline_id"]
        if rid is None:
            continue
        for cid in range(len(p["y"])):
            if cid == rid:
                continue
            a, b = p["y"][cid], p["y"][rid]
            if a["success"] != b["success"]:
                better, worse = (cid, rid) if a["success"] else (rid, cid)
                kind, weight = "success", 2. if better == rid else 1.
            elif a["success"] and (a["steps"], a["calls"]) != (b["steps"], b["calls"]):
                better, worse = (cid, rid) if (a["steps"], a["calls"]) < (b["steps"], b["calls"]) else (rid, cid)
                kind, weight = "successful_cost", .1
            else:
                continue
            if cid not in p["comparable"]:
                audit["route_incomparable"] += 1
                continue
            if not np.any(np.abs(p["x"][cid] - p["x"][rid]) > 1e-10):
                audit["inactive_features"] += 1
                continue
            # Fixed cap cannot overcome arbitrarily large V8 score gaps. Do not
            # spend the single fit chasing an impossible correction.
            margin = max(.01, float(p["kw"]["backbone"].switch_margin))
            if better == cid and p["base_scores"][rid] - p["base_scores"][cid] + margin >= CAP:
                audit["outside_correction_range"] += 1
                continue
            pairs.append((pool, better, worse, weight, margin))
            audit[kind] += 1
    return pairs, audit


def fit_once(train_pools, *, steps=1200, lr=.01, l2=.1):
    pairs, audit = preference_pairs(train_pools)
    if not pairs or not audit["success"]:
        raise ValueError(f"No deployment-comparable active TRAIN success supervision: {audit}")
    all_x = np.concatenate([p["x"] for p in train_pools])
    scale = all_x.std(axis=0); scale[scale < 1e-8] = 1.
    support = np.abs(all_x).max(axis=0)
    model = ConditionalUtility(scale, np.zeros(len(FEATURE_NAMES)), support)
    z = [model.normalized(p["x"]) for p in train_pools]
    zi = np.array([z[k][i] for k, i, j, weight, margin in pairs])
    zj = np.array([z[k][j] for k, i, j, weight, margin in pairs])
    base = np.array([train_pools[k]["base_scores"][i] - train_pools[k]["base_scores"][j]
                     for k, i, j, weight, margin in pairs])
    unit = np.array([max(SCORE_UNIT, margin) for k, i, j, weight, margin in pairs])
    weight = np.array([weight for k, i, j, weight, margin in pairs])
    w = model.weights.copy(); losses = []
    for step in range(steps):
        ti, tj = np.tanh(np.clip(zi @ w, -30., 30.)), np.tanh(np.clip(zj @ w, -30., 30.))
        logits = (base + CAP / 2 * (ti - tj) - unit) / unit
        inv = 1. / (1. + np.exp(np.clip(logits, -30., 30.)))
        grad = -np.mean((weight * inv * CAP / (2 * unit))[:, None] *
                       ((1 - ti * ti)[:, None] * zi - (1 - tj * tj)[:, None] * zj), axis=0) + l2 * w
        w -= lr * grad
        if step in (0, steps - 1):
            losses.append(float(np.mean(weight * np.logaddexp(0., -logits)) + l2 / 2 * (w @ w)))
    return replace(model, weights=w), {**audit, "fit_count": 1, "steps": steps,
        "lr": lr, "l2": l2, "loss_first_last": losses, "validation_used": False,
        "nonzero_weights": int(np.count_nonzero(np.abs(w) > 1e-10)),
        "objective": "anchor-relative terminal utility in original switching-score units"}

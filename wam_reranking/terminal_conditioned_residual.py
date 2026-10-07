"""Terminal-utility residual; NOT a supervised physical-recovery probability.

The frozen selector remains authoritative. New features connect current cause,
source-aware belief/DAG deficits and each candidate's PRE-execution prediction.
No future outcome, silver intervention or task identity is an inference input.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .belief import DEFAULT_GRAPH, DependencyGraph, update_belief
from .candidate_utility import select_with_utility
from .contracts import BeliefFact, CoarseCause, ConsistencyFactor, PREDICATES, TriValue

SCHEMA = "terminal_conditioned_residual_v10_utility_v1"
ROUTES = tuple(c.value for c in CoarseCause)
ROOTS = {"normal": (), "visual_occlusion": ("target_visible", "target_pose_current", "receptacle_visible"),
         "object_shift": ("target_pose_current",),
         "execution_contact_deviation": ("execution_consistent",), "unknown": ()}
METRICS = ("required_risk", "endpoint_forecast", "predicted_effect")
FEATURE_NAMES = tuple(f"{r}.{p}.{m}" for r in ROUTES for p in PREDICATES for m in METRICS)
CAP = .10  # Inherits the bounded residual contract, not a new safety threshold.
GAIN_GRID = (0., .125, .25, .375, .5, .625, .75, .875, 1.)


def _finite_plan(actions):
    a = np.asarray(actions, dtype=float)
    if a.shape != (16, 7) or not np.isfinite(a).all():
        raise ValueError("Complete finite native 16x7 plan required")
    return a


def conditioned_features(*, prior, attribution, effect, actions, relation="inside", no_dag=False,
                         block_index=3):
    """Return a typed PRE-only feature vector plus provenance, without a gate write.

    Endpoint support remains a forecast at step16. A close command never
    certifies holding, and terminal support never establishes an earlier fact.
    Generic initial assumptions are masked, rather than certified by reliability.
    """
    a = _finite_plan(actions)
    b = deepcopy(prior)
    for name, f in list(b.facts.items()):
        if f.source == "candidate_prediction" or "initial_observation" in f.evidence_ids:
            b.facts[name] = BeliefFact(TriValue.UNKNOWN, 0., "observation", block_index,
                                      ("unverified_assumption_masked",))
    graph = DependencyGraph(()) if no_dag else DEFAULT_GRAPH
    start = len(b.history)
    update_belief(b, attribution, block_index, graph=graph)
    changes = b.history[start:]
    factors = {f.value: f for f in ConsistencyFactor}
    needs = {}
    paths = {}
    for p in PREDICATES:
        h = [c for c in changes if c.predicate == p and c.new_value is not TriValue.TRUE]
        # Do not turn an unobserved initial unknown into a caused physical loss.
        strengths = [min(c.new_confidence, attribution.factor_confidence(factors[c.reason]))
                     if c.reason in factors else c.new_confidence for c in h]
        needs[p] = max(strengths, default=0.)
        paths[p] = [list(c.propagation_path) for c in h]
    q = attribution.evidence_quality
    observed = attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is TriValue.TRUE
    resolved = attribution.factor_state(ConsistencyFactor.CAUSE_RESOLVED) is TriValue.TRUE
    usable = (q.primary_reliable or q.wrist_reliable) and not q.cross_view_conflict
    e = effect.evidence
    view_quality = float(np.clip(e.get("cross_view_agreement", 0.), 0., 1.))
    candidate_quality = min(float(effect.confidence), view_quality)
    # The inherited evidence confidence cut is not changed to manufacture switches.
    sufficient = usable and candidate_quality >= .5
    routes = {r: 0. for r in ROUTES}
    if sufficient:
        routes["visual_occlusion"] = float(attribution.class_probs["visual_occlusion"])
        if observed and resolved:
            routes["object_shift"] = (1. - attribution.factor(ConsistencyFactor.WORLD_STATE_CONSISTENT))
        if q.execution_reliable:
            routes["execution_contact_deviation"] = (1. - attribution.factor(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT))
        routes["unknown"] = float(attribution.class_probs["unknown"]) if not resolved else 0.
    close = a[:, 6] > .25  # LIBERO +1 closes; command is a prediction input only.
    up = a[:, 2] > 0.
    close_up = float(np.any(close & up))
    relation_q = float(np.clip(e.get("relation_confidence", 0.), 0., 1.))
    contact_q = float(np.clip(e.get("contact_confidence", 0.), 0., 1.))
    grasp = float(np.clip(e.get("grasp_support_after", 0.), 0., 1.))
    risk = float(np.clip(e.get("trajectory_risk", 0.), 0., 1.))
    uncertainty = float(np.clip(1. - candidate_quality + risk, 0., 1.))
    proxy = {
        "target_visible": candidate_quality,
        "target_pose_current": relation_q * float(np.clip(e.get("target_displacement", 0.), 0., 1.)),
        "target_reachable": contact_q * float(np.clip(1. - e.get("target_gripper_distance_after", 1.), 0., 1.)),
        "grasped": contact_q * grasp,
        "lifted": contact_q * grasp * close_up,
        "receptacle_visible": relation_q * candidate_quality,
        "place_ready": relation_q * float(np.clip(e.get("relation_score_after", 0.), -1., 1.)),
        "placed": relation_q * float(np.clip(e.get("relation_score_delta", 0.), -1., 1.)),
        "execution_consistent": float(np.clip(e.get("trajectory_path_efficiency", 0.), 0., 1.)) - risk,
    }
    required = dict(effect.required_facts)
    proposals = dict(effect.proposed_effects)
    if relation in {"open", "closed", "articulated"}:
        required = {"target_visible": .5, "target_pose_current": .6}
        proposals = {}  # Do not claim carrying/lifting an articulated drawer.
    vector = []
    for r in ROUTES:
        scope = set(ROOTS[r])
        if r in {"object_shift", "execution_contact_deviation"}:
            for root in ROOTS[r]:
                scope.update(graph.descendants_with_paths(root))
        if r == "unknown":
            scope = set(required)
        for p in PREDICATES:
            w = routes[r] * needs[p] if p in scope else 0.
            req = float(required.get(p, 0.))
            forecast = proposals.get(p, (TriValue.UNKNOWN, 0.))
            support = float(forecast[1]) if forecast[0] is TriValue.TRUE else 0.
            if r == "unknown":
                # Unknown may encode uncertainty/risk, never a guessed physical recovery.
                vector.extend((w * req * uncertainty, 0., -w * req * uncertainty))
            else:
                vector.extend((w * req * uncertainty, w * support * candidate_quality,
                               w * proxy[p] * candidate_quality))
    x = np.asarray(vector, dtype=float)
    if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
        raise ValueError("Invalid terminal-conditioned inference features")
    return x, {"sufficient_evidence": sufficient, "routes": routes, "needs": needs,
               "paths": paths, "endpoint_is_forecast_only": True,
               "no_dag": bool(no_dag), "block_index": int(block_index),
               "recovery_probability_claim": False}


@dataclass(frozen=True)
class TerminalResidual:
    scale: np.ndarray
    weights: np.ndarray
    support: np.ndarray
    gain: float = 0.

    def __post_init__(self):
        for x in (self.scale, self.weights, self.support):
            if np.asarray(x).shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
                raise ValueError("Invalid residual parameters")
        if np.any(self.scale <= 0) or np.any(self.support < 0) or not 0 <= self.gain <= 1:
            raise ValueError("Invalid residual scale/support/gain")

    def deltas(self, matrix):
        x = np.asarray(matrix, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("Invalid complete candidate matrix")
        # Entire-pool backoff avoids a single unsupported candidate receiving an
        # artificial advantage merely because only its residual was set to zero.
        if np.any(np.abs(x) > self.support + 1e-8) or self.gain == 0:
            return np.zeros(len(x))
        z = (x - x.mean(axis=0)) / self.scale
        raw = CAP * np.tanh(np.clip(z @ self.weights, -30., 30.))
        return self.gain * (raw - raw.mean()) / 2.  # centered; each |delta| <= CAP

    def record(self):
        return {"schema": SCHEMA, "feature_names": list(FEATURE_NAMES), "scale": self.scale.tolist(),
                "weights": self.weights.tolist(), "support": self.support.tolist(), "gain": self.gain,
                "cap": CAP, "recovery_probability_head_trained": False}


def select_residual(decisions, base_scores, deltas, margin):
    """Same frozen feasibility, official tie break and Pareto guard as V8."""
    if len(decisions) != len(base_scores) or len(decisions) != len(deltas):
        raise ValueError("Candidate alignment mismatch")
    if not np.isfinite(base_scores).all() or not np.isfinite(deltas).all():
        raise ValueError("Nonfinite score")
    class Scores:
        switch_margin = margin
        @staticmethod
        def score(v):
            return float(v[0])
    x0 = {d.candidate_id: np.asarray([s]) for d, s in zip(decisions, base_scores)}
    baseline, _ = select_with_utility(decisions, x0, Scores())
    x1 = {d.candidate_id: np.asarray([s + c]) for d, s, c in zip(decisions, base_scores, deltas)}
    selected, scores = select_with_utility(decisions, x1, Scores())
    if baseline is not None and selected is not None and selected.candidate_id != baseline.candidate_id:
        delta = {d.candidate_id: c for d, c in zip(decisions, deltas)}
        # Switching requires positive candidate-specific improvement evidence.
        if delta[selected.candidate_id] <= delta[baseline.candidate_id] + 1e-12:
            selected = baseline
    return selected, scores


def fit_terminal_pairs(train_pools: Sequence[dict], *, steps=1200, lr=.05, l2=.1):
    """One deterministic fit. Only train outcomes define pairwise utility Y."""
    if not train_pools or any(p["split"] != "train" for p in train_pools):
        raise ValueError("Fit accepts explicit TRAIN pools only")
    all_x = np.concatenate([p["x"] for p in train_pools])
    scale = all_x.std(axis=0)
    scale[scale < 1e-8] = 1.
    support = np.abs(all_x).max(axis=0)
    pool_x, pairs, kinds = [], [], []
    for p in train_pools:
        x = (p["x"] - p["x"].mean(axis=0)) / scale
        start = len(pool_x)
        pool_x.extend(x)
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if not (p["decisions"][i].accepted and p["decisions"][j].accepted):
                    continue
                yi, yj = p["y"][i], p["y"][j]
                weight, kind = 0., ""
                if yi["success"] != yj["success"]:
                    better, worse = (i, j) if yi["success"] else (j, i)
                    weight, kind = 1., "success"
                    if p["baseline_id"] is not None and p["y"][p["baseline_id"]]["success"] and worse == p["baseline_id"]:
                        raise AssertionError("A successful V8 anchor cannot be a failure target")
                    if better == p["baseline_id"]:
                        weight = 2.  # preserve a successful frozen anchor first
                elif yi["success"] and (yi["steps"], yi["calls"]) != (yj["steps"], yj["calls"]):
                    better, worse = (i, j) if (yi["steps"], yi["calls"]) < (yj["steps"], yj["calls"]) else (j, i)
                    weight, kind = .1, "successful_cost"
                else:
                    continue
                pairs.append((start + better, start + worse,
                              float(p["base_scores"][better] - p["base_scores"][worse]), weight))
                kinds.append(kind)
    if not pairs:
        raise ValueError("No actual TRAIN terminal preference pairs")
    z = np.asarray(pool_x)
    ii = np.array([v[0] for v in pairs]); jj = np.array([v[1] for v in pairs])
    margins = np.array([v[2] for v in pairs]); pair_w = np.array([v[3] for v in pairs])
    w = np.zeros(len(FEATURE_NAMES))
    losses = []
    for step in range(steps):
        ti = np.tanh(np.clip(z[ii] @ w, -30, 30)); tj = np.tanh(np.clip(z[jj] @ w, -30, 30))
        logits = margins + CAP / 2 * (ti - tj)
        inv = 1. / (1. + np.exp(np.clip(logits, -30, 30)))
        grad = -CAP / 2 * np.mean(pair_w[:, None] * inv[:, None] *
                    ((1 - ti * ti)[:, None] * z[ii] - (1 - tj * tj)[:, None] * z[jj]), axis=0) + l2 * w
        w -= lr * grad
        if step in (0, steps - 1):
            losses.append(float(np.mean(pair_w * np.logaddexp(0., -logits)) + l2 / 2 * (w @ w)))
    return TerminalResidual(scale, w, support), {"success_pairs": kinds.count("success"),
        "successful_cost_pairs": kinds.count("successful_cost"), "steps": steps,
        "loss_first_last": losses, "nonzero_weights": int(np.count_nonzero(np.abs(w) > 1e-10)),
        "fit_uses_validation": False}

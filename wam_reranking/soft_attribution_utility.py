"""Continuous, source-scoped PRE terminal utility; not recovery probability.

The frozen selector supplies the reference and hard-feasible set.  This new
channel only predicts the utility of a forecast, never writes belief, changes
an attribution class/threshold, or certifies current physical prerequisites.
Training supervision is accepted only by ``fit_once``; inference reads an
explicit AFTER-only whitelist.  Old models and feature schemas are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np

from .belief import DEFAULT_GRAPH
from .contracts import CoarseCause, ConsistencyFactor
from .reranker import PREDICATE_HARD_THRESHOLDS

SCHEMA = "continuous_after_forecast_terminal_utility_v2_full_competition"
CAP = .1
MIN_MARGIN = .01
FIT_STEPS = 1200
SUCCESS_STEPS = 900
LR = .01
L2 = .1
BACKTRACK_STEPS = 12
REPAIR_MARGIN_MULTIPLIER = 2.
FEATURE_NAMES = (
    "world_after_relation", "world_after_contact", "world_after_grasp",
    "world_after_release", "world_required_quality_risk", "world_required_relation",
    "execution_after_contact", "execution_after_grasp", "execution_after_release",
    "execution_required_quality_risk", "execution_required_grasp",
    "observation_forecast_visibility", "observation_forecast_uncertainty",
    "observation_required_visibility", "cross_view_forecast_uncertainty",
    "coarse_unknown_forecast_uncertainty", "head_disagreement_forecast_uncertainty",
    "all_physical_negative_after_relation", "all_physical_negative_trajectory_risk",
    "stage_unresolved_reset_compatibility",
)
AFTER_FIELDS = (
    "relation_score_after", "relation_confidence", "cross_view_agreement",
    "contact_confidence", "grasp_support_after", "predicted_grasp_support",
    "predicted_release_support", "target_gripper_distance_after", "visual_support",
    "trajectory_risk",
)


class NoFitError(ValueError):
    """Audited training conditions do not permit the one fixed fit."""

    def __init__(self, message, *, audit=None):
        super().__init__(message)
        self.audit = {} if audit is None else audit


def _unit(value, name):
    value = float(value)
    if not np.isfinite(value) or not 0. <= value <= 1.:
        raise ValueError(f"{name} must be a finite probability/quality in [0,1]")
    return value


def _read(evidence, name, default=0.):
    return _unit(evidence.get(name, default), name)


def _distance(evidence, name, default=1.):
    # Normalized image-space distances are Euclidean and can reach sqrt(2).
    # Their bounded affinity proxy is not a probability/certificate threshold.
    value = float(evidence.get(name, default))
    if not np.isfinite(value) or value < 0.:
        raise ValueError(f"{name} must be a finite nonnegative distance")
    return float(np.clip(value, 0., 1.))


def _missing_requirements(snapshot, block_index, no_dag):
    """Only an upstream-admitted, same-block actual TRUE closes soft exposure.

    A source string alone is not a certificate: require its matching explicit
    current-observation transition as well.  The caller must supply its own
    variant's audited history; a no-DAG arm cannot borrow propagated history.
    """
    if snapshot is None:
        return {}, []
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("facts"), Mapping):
        raise ValueError("An audited belief snapshot mapping is required")
    history = snapshot.get("history", ())
    if no_dag and any(len(h.get("propagation_path", ())) > 1 and h.get("reason") in {
            "world_state_consistent", "execution_contact_consistent", "observation_reliable",
            "measured_command_deviation"} for h in history):
        raise ValueError("No-DAG features require their own unpropagated history")
    missing, certified = {}, []
    for predicate, fact in snapshot["facts"].items():
        if not isinstance(fact, Mapping):
            raise ValueError("Belief snapshot facts must be mappings")
        value = getattr(fact.get("value"), "value", fact.get("value"))
        ids = tuple(fact.get("evidence_ids", ()))
        confidence = _unit(fact.get("confidence", 0.), "fact confidence")
        current_transition = any(h.get("predicate") == predicate
            and h.get("reason") == "current_observation"
            and getattr(h.get("new_value"), "value", h.get("new_value")) == "true"
            and h.get("evidence_id") in ids and bool(h.get("direct"))
            and tuple(h.get("propagation_path", ())) == (predicate,) for h in history)
        verified = (value == "true" and fact.get("source") == "observation"
            and fact.get("updated_at_block") == block_index and ids
            and "initial_observation" not in ids and current_transition
            and confidence >= PREDICATE_HARD_THRESHOLDS.get(predicate, .75))
        missing[predicate] = 0. if verified else 1.
        if verified:
            certified.append(predicate)
    return missing, sorted(certified)


def _required_relevance(effect, root, no_dag, missing):
    """Expected dependency exposure, not an assertion that a fact is false."""
    requirements = dict(effect.required_facts)
    if any(not np.isfinite(float(v)) or not 0. <= float(v) <= 1.
           for v in requirements.values()):
        raise ValueError("Invalid required-fact relevance")
    denominator = sum(float(v) for v in requirements.values())
    if denominator <= 0.:
        return 0.
    paths = {root: (root,)}
    if not no_dag:
        paths.update(DEFAULT_GRAPH.descendants_with_paths(root))
    return sum(float(v) * .85 ** (len(paths[p]) - 1) * missing.get(p, 1.)
               for p, v in requirements.items() if p in paths) / denominator


def soft_features(attribution, effects, relation, *, no_dag=False, masked=False, ablate_all=False,
                  belief_snapshot=None, block_index=3):
    """Candidate-specific scores from continuous evidence and AFTER forecasts.

    Missing/unreliable current vision closes only this visual channel.  Command
    evidence and its frozen reference remain outside this module.  Nonzero
    factor probabilities do not invalidate live belief, even below 0.50.
    BEFORE/delta/target displacement fields are deliberately never read.
    ``masked`` removes learned cause content while retaining ordinary forecast
    main effects; it must not mistake a new candidate-only head for attribution
    benefit.  ``ablate_all`` is a separate exact-zero engineering control.
    """
    effects = tuple(effects)
    if not effects or [int(e.candidate_id) for e in effects] != list(range(len(effects))):
        raise ValueError("Complete ordered candidate IDs required")
    q = attribution.evidence_quality
    usable = bool((q.primary_reliable or q.wrist_reliable) and not q.cross_view_conflict)
    world = 1. - _unit(attribution.factor(ConsistencyFactor.WORLD_STATE_CONSISTENT), "world")
    execution = 1. - _unit(attribution.factor(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT), "execution")
    observation = 1. - _unit(attribution.factor(ConsistencyFactor.OBSERVATION_RELIABLE), "observation")
    stage = 1. - _unit(attribution.factor(ConsistencyFactor.TASK_STAGE_CONSISTENT), "stage")
    coarse = {str(k): _unit(v, "class probability") for k, v in attribution.class_probs.items()}
    if not np.isclose(sum(coarse.values()), 1., atol=1e-5):
        raise ValueError("Coarse probabilities do not sum to one")
    unknown = coarse.get(CoarseCause.UNKNOWN.value, 0.)
    disagreement = float(max(coarse, key=coarse.get) != attribution.projected_cause.value)
    if masked:
        world = execution = observation = stage = unknown = disagreement = 0.
    relation_name = str(relation).lower()
    directional = relation_name != "articulated"
    missing, certified = _missing_requirements(belief_snapshot, block_index, no_dag)
    rows, sources = [], []
    for effect in effects:
        evidence = effect.evidence
        certainty = _unit(effect.confidence, "effect confidence")
        agreement = _read(evidence, "cross_view_agreement")
        relation_quality = min(certainty, _read(evidence, "relation_confidence"), agreement)
        contact_quality = min(certainty, _read(evidence, "contact_confidence"), agreement)
        visibility = min(certainty, _read(evidence, "visual_support", certainty))
        after = _read(evidence, "relation_score_after")
        # The frozen image-space articulated score is frame separation.  Closed
        # means less separation; an unknown articulated direction claims none.
        endpoint = (1. - after if relation_name == "closed" else after) if directional else 0.
        contact = 1. - _distance(evidence, "target_gripper_distance_after", 1.)
        grasp = _read(evidence, "grasp_support_after")
        release = _read(evidence, "predicted_release_support")
        risk = _read(evidence, "trajectory_risk")
        uncertainty = float(np.clip(1. - min(relation_quality, contact_quality) + risk, 0., 1.))
        wr = _required_relevance(effect, "target_pose_current", no_dag, missing)
        er = _required_relevance(effect, "execution_consistent", no_dag, missing)
        vr = _required_relevance(effect, "target_visible", no_dag, missing)
        negative = (1. - world) * (1. - execution)
        reset = float(effect.stage.value in {"observe", "approach", "grasp"})
        row = np.asarray([
            world * relation_quality * endpoint, world * contact_quality * contact,
            world * contact_quality * grasp, world * contact_quality * release,
            world * wr * (1. - relation_quality), world * wr * relation_quality * endpoint,
            execution * contact_quality * contact, execution * contact_quality * grasp,
            execution * contact_quality * release, execution * er * (1. - contact_quality),
            execution * er * contact_quality * grasp,
            observation * visibility, observation * uncertainty, observation * vr * visibility,
            float(q.cross_view_conflict) * uncertainty,
            unknown * uncertainty, disagreement * uncertainty,
            negative * relation_quality * endpoint, negative * risk, stage * reset * certainty,
        ], dtype=np.float64)
        if ablate_all or not usable:
            row[:] = 0.
        rows.append(row)
        sources.append(dict(candidate_id=int(effect.candidate_id), relation_quality=relation_quality,
            contact_quality=contact_quality, world_required_relevance=wr,
            execution_required_relevance=er, observation_required_relevance=vr))
    matrix = np.stack(rows)
    if matrix.shape != (len(effects), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid soft forecast matrix")
    return matrix, dict(schema=SCHEMA, visual_source_usable=usable, masked=bool(masked),
        all_channel_ablation=bool(ablate_all), ordinary_forecast_main_effects_retained_when_masked=True,
        no_dag=bool(no_dag), relation=relation_name, articulated_direction_known=directional,
        continuous_world_anomaly=world, continuous_execution_anomaly=execution,
        coarse_unknown_probability=unknown, direct_final_head_disagreement=bool(disagreement),
        certified_current_required_facts=certified, missing_requirement_weights=missing,
        belief_use_block_index=block_index, absent_belief_means_unconfirmed=belief_snapshot is None,
        source_quality=sources, input_field_whitelist=list(AFTER_FIELDS),
        before_fields_used=False, prediction_writes=0, live_belief_mutated=False,
        physical_recovery_probability=False, command_channel_changed=False)


@dataclass(frozen=True)
class SoftUtilityModel:
    scale: np.ndarray
    weights: np.ndarray
    support: np.ndarray
    gain: float = 1.

    def __post_init__(self):
        for name in ("scale", "weights", "support"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != (len(FEATURE_NAMES),) or not np.isfinite(value).all():
                raise ValueError("Soft model schema/vector mismatch")
            if (name == "scale" and np.any(value <= 0.)) or (name == "support" and np.any(value < 0.)):
                raise ValueError("Invalid scale/support")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.gain not in (0., 1.):
            raise ValueError("Only exact zero identity or fixed unit gain is permitted")

    def normalized(self, matrix):
        x = np.asarray(matrix, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("Invalid soft inference matrix")
        bounded = np.clip(x, -self.support, self.support)
        return np.clip((bounded - bounded.mean(axis=0)) / self.scale, -4., 4.)

    def deltas(self, matrix):
        z = self.normalized(matrix)
        raw = CAP / 2 * np.tanh(np.clip(z @ self.weights, -30., 30.))
        return self.gain * (raw - raw.mean())

    def to_record(self):
        return dict(schema=SCHEMA, feature_names=list(FEATURE_NAMES), scale=self.scale.tolist(),
            weights=self.weights.tolist(), support=self.support.tolist(), gain=self.gain,
            cap=CAP, minimum_switch_margin=MIN_MARGIN, terminal_utility=True,
            physical_recovery_probability=False, validation_used_for_gain=False)

    @classmethod
    def from_record(cls, record):
        if record.get("schema") != SCHEMA or tuple(record.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Soft model record schema mismatch")
        if record.get("cap") != CAP or record.get("minimum_switch_margin") != MIN_MARGIN:
            raise ValueError("Frozen cap/margin mismatch")
        return cls(record["scale"], record["weights"], record["support"], float(record["gain"]))


def fast_selection(base, delta, references, accepted, official, margins):
    """Same outcome-free choice rule as deploy selection, including near ties."""
    base, delta, official = (np.asarray(v, dtype=np.float64) for v in (base, delta, official))
    refs, accepted = np.asarray(references, dtype=int), np.asarray(accepted, dtype=bool)
    margins = np.asarray(margins, dtype=np.float64)
    if base.ndim != 2 or base.shape != delta.shape or base.shape != official.shape or base.shape != accepted.shape:
        raise ValueError("Candidate alignment mismatch")
    n, k = base.shape
    if refs.shape != (n,) or margins.shape != (n,) or not k:
        raise ValueError("Reference/margin alignment mismatch")
    if not all(np.isfinite(v).all() for v in (base, delta, official, margins)):
        raise ValueError("Nonfinite selection input")
    if np.any(margins < MIN_MARGIN) or np.any(np.ptp(delta, axis=1) > CAP + 1e-12):
        raise ValueError("Frozen margin/cap violated")
    if np.any((refs < -1) | (refs >= k)):
        raise ValueError("Invalid reference index")
    safe = np.maximum(refs, 0)
    if np.any((refs >= 0) & ~accepted[np.arange(n), safe]):
        raise ValueError("Reference must be hard-feasible")
    scores = base + delta
    eligible = accepted.copy()
    eligible[np.arange(n), safe] = False
    eligible &= delta > delta[np.arange(n), safe, None] + 1e-12
    eligible &= scores > scores[np.arange(n), safe, None] + margins[:, None]
    eligible[refs < 0, :] = False
    top = np.max(np.where(eligible, scores, -np.inf), axis=1)
    tied = eligible & (scores >= top[:, None] - margins[:, None])
    choice = np.argmax(np.where(tied, official, -np.inf), axis=1)
    return np.where(np.any(eligible, axis=1), choice, refs)


def select_soft_utility(decisions, base_scores, official_values, reference_id, features,
                        model, *, margin=MIN_MARGIN):
    decisions = tuple(decisions)
    if [d.candidate_id for d in decisions] != list(range(len(decisions))):
        raise ValueError("Complete ordered decisions required")
    values = np.asarray(official_values, dtype=np.float64)
    base = np.asarray([base_scores[i] for i in range(len(decisions))], dtype=np.float64) if isinstance(base_scores, Mapping) else np.asarray(base_scores, dtype=np.float64)
    delta = model.deltas(features)
    accepted = np.asarray([d.accepted for d in decisions], dtype=bool)
    ref = -1 if reference_id is None else int(reference_id)
    chosen = int(fast_selection(base[None, :], delta[None, :], [ref], accepted[None, :],
                               values[None, :], [float(margin)])[0])
    return dict(selected_id=None if chosen < 0 else chosen,
        reference_id=reference_id, scores={i: float(base[i] + delta[i]) for i in range(len(base)) if accepted[i]},
        deltas=delta.tolist(), reason="reference_fallback_retained" if ref < 0 else
        "exact_zero_or_reference_retained" if chosen == ref else "positive_soft_margin_clearance",
        hard_feasible_only=True, physical_recovery_probability=False)


def _training_arrays(pools):
    if not pools or any(p["split"] != "train" for p in pools):
        raise NoFitError("NO FIT: explicit TRAIN pools only")
    keys = [tuple(p["key"]) for p in pools]
    if len(set(keys)) != len(keys):
        raise NoFitError("NO FIT: duplicate pool identity")
    x = np.asarray([p["x"] for p in pools], dtype=np.float64)
    base = np.asarray([p["base_scores"] for p in pools], dtype=np.float64)
    official = np.asarray([p["official_values"] for p in pools], dtype=np.float64)
    accepted = np.asarray([p["accepted"] for p in pools], dtype=bool)
    refs = np.array([-1 if p["reference_id"] is None else p["reference_id"] for p in pools], dtype=int)
    margins = np.array([max(MIN_MARGIN, float(p.get("margin", MIN_MARGIN))) for p in pools])
    if x.ndim != 3 or x.shape[:2] != base.shape or x.shape[2] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise NoFitError("NO FIT: invalid training feature schema")
    fast_selection(base, np.zeros_like(base), refs, accepted, official, margins)
    outcomes = [list(p["outcomes"]) for p in pools]
    if any(len(v) != base.shape[1] for v in outcomes):
        raise NoFitError("NO FIT: outcome alignment")
    for row in outcomes:
        for y in row:
            if y is None:
                continue
            if not isinstance(y.get("success"), (bool, np.bool_)) or any(
                not np.isfinite(float(y[name])) or float(y[name]) < 0. for name in ("steps", "calls")):
                raise NoFitError("NO FIT: invalid terminal label")
    return x, base, official, accepted, refs, margins, outcomes


def fit_once(train_pools, *, steps=FIT_STEPS, lr=LR, l2=L2):
    """One fixed success-first fit; no validation/gain search/hidden fallback Y."""
    if (steps, lr, l2) != (FIT_STEPS, LR, L2):
        raise NoFitError("NO FIT: predeclared configuration required")
    x, base, official, accepted, refs, margins, outcomes = _training_arrays(train_pools)
    scale = x.reshape(-1, x.shape[-1]).std(axis=0)
    scale[scale < 1e-8] = 1.
    support = np.max(np.abs(x), axis=(0, 1))
    zero = SoftUtilityModel(scale, np.zeros(x.shape[-1]), support)
    z = np.stack([zero.normalized(v) for v in x])
    groups = {"corrective": [], "protective": [], "cost": []}
    skipped = dict(inactive=0, outside_correction_range=0, unknown_reference=0, unknown_candidate=0,
                   rejected_corrective_candidates=0, unknown_competitor_candidates=0)
    competition_audit = []
    protected = []
    all_failed = 0
    for k, p in enumerate(train_pools):
        yy = outcomes[k]
        all_failed += bool(all(y is not None and not y["success"] for y in yy))
        rid = refs[k]
        if rid < 0:
            # An unexecuted fallback is not a failed label, nor a candidate
            # reference this accepted-only residual can safely replace.
            skipped["unknown_reference"] += 1
            continue
        ref_y = yy[rid]
        if ref_y is None:
            skipped["unknown_reference"] += 1
            continue
        vid = int(p.get("value_id", int(np.argmax(official[k]))))
        if not 0 <= vid < len(yy):
            raise NoFitError("NO FIT: invalid value reference")
        preserve = bool(ref_y["success"] or (accepted[k, vid] and yy[vid] is not None and yy[vid]["success"]))
        if preserve:
            protected.append(k)
        for cid, y in enumerate(yy):
            if cid == rid or not accepted[k, cid]:
                continue
            if y is None:
                skipped["unknown_candidate"] += 1
                continue
            if y["success"] != ref_y["success"]:
                better, worse = (cid, rid) if y["success"] else (rid, cid)
                name = "corrective" if better != rid else "protective"
            elif y["success"] and (y["steps"], y["calls"]) != (ref_y["steps"], ref_y["calls"]):
                better, worse = (cid, rid) if (y["steps"], y["calls"]) < (ref_y["steps"], ref_y["calls"]) else (rid, cid)
                name = "cost"
            else:
                continue
            if not np.any(np.abs(z[k, cid] - z[k, rid]) > 1e-10):
                skipped["inactive"] += 1
                if name == "corrective":
                    skipped["rejected_corrective_candidates"] += 1
                    competition_audit.append(dict(key=list(p["key"]), successful_candidate=cid,
                        admitted=False, reason="inactive_reference_difference", competitors=[]))
                continue
            if name == "corrective":
                # A successful candidate beating just the frozen reference is
                # insufficient: another admitted failure can still be top-1.
                # Admit a complete competition target or exclude that entire
                # target, never silently train only its easy reference pair.
                unknown_ids = [j for j, rival in enumerate(yy) if accepted[k, j] and rival is None]
                failures = [j for j, rival in enumerate(yy)
                            if accepted[k, j] and rival is not None and not rival["success"]]
                required = REPAIR_MARGIN_MULTIPLIER * margins[k]
                rivals, reasons = [], []
                for j in failures:
                    gap = float(base[k, cid] - base[k, j])
                    active = bool(np.any(np.abs(z[k, cid] - z[k, j]) > 1e-10))
                    achievable = gap + CAP > required
                    reason = "outside_fixed_cap" if not achievable else (
                        "inactive_unsatisfied_competitor" if not active and gap < required else None)
                    rivals.append(dict(candidate_id=j, reference_competitor=bool(j == rid),
                        base_score_gap=gap, required_score_gap=float(required),
                        active_feature_difference=active, achievable_under_fixed_cap=bool(achievable),
                        exclusion_reason=reason))
                    if reason:
                        reasons.append(reason)
                if unknown_ids:
                    reasons.append("unknown_accepted_competitor_labels")
                    skipped["unknown_competitor_candidates"] += 1
                admitted = not reasons
                competition_audit.append(dict(key=list(p["key"]), successful_candidate=cid,
                    admitted=admitted, reasons=reasons, competitors=rivals,
                    unknown_accepted_competitors=unknown_ids, whole_candidate_target_excluded=not admitted))
                if not admitted:
                    skipped["rejected_corrective_candidates"] += 1
                    skipped["outside_correction_range"] += sum(r == "outside_fixed_cap" for r in reasons)
                    skipped["inactive"] += sum(r == "inactive_unsatisfied_competitor" for r in reasons)
                    continue
                groups["corrective"].extend((k, cid, j, margins[k]) for j in failures)
                continue
            groups[name].append((k, better, worse, margins[k]))
    if not groups["corrective"]:
        raise NoFitError(f"NO FIT: no admitted corrective TRAIN pair with complete competitor coverage; {skipped}",
                         audit=dict(skipped=skipped, corrective_competition=competition_audit))

    def deltas(w):
        raw = CAP / 2 * np.tanh(np.clip(z @ w, -30., 30.))
        return raw - raw.mean(axis=1, keepdims=True)

    def choices(w):
        return fast_selection(base, deltas(w), refs, accepted, official, margins)

    def safe(w):
        ids = choices(w)
        return all(ids[k] >= 0 and outcomes[k][ids[k]] is not None and outcomes[k][ids[k]]["success"]
                   for k in protected)

    if not safe(zero.weights):
        raise NoFitError("NO FIT: zero reference already harms a hard-feasible train value success")
    arrays = {}
    for name, rows in groups.items():
        if rows:
            kk, ii, jj, mm = map(np.asarray, zip(*rows))
            kk, ii, jj = (v.astype(int) for v in (kk, ii, jj))
            arrays[name] = (z[kk, ii], z[kk, jj], base[kk, ii] - base[kk, jj], mm)
        else:
            arrays[name] = None

    def terms(w, name):
        if arrays[name] is None:
            return 0., np.zeros_like(w), np.array([])
        zi, zj, b, m = arrays[name]
        ti, tj = np.tanh(np.clip(zi @ w, -30., 30.)), np.tanh(np.clip(zj @ w, -30., 30.))
        gap = b + CAP / 2 * (ti - tj)
        deriv = CAP / (2 * m[:, None]) * ((1. - ti * ti)[:, None] * zi - (1. - tj * tj)[:, None] * zj)
        if name == "protective":
            violation = np.maximum(0., (-m - gap) / m)
            return float(np.mean(violation ** 2)), np.mean(-2. * violation[:, None] * deriv, axis=0), gap
        target = (REPAIR_MARGIN_MULTIPLIER if name == "corrective" else 1.) * m
        logits = (gap - target) / m
        inv = 1. / (1. + np.exp(np.clip(logits, -30., 30.)))
        return float(np.mean(np.logaddexp(0., -logits))), np.mean(-inv[:, None] * deriv, axis=0), gap

    def repaired(w):
        gaps = terms(w, "corrective")[2]
        if not np.all(gaps >= REPAIR_MARGIN_MULTIPLIER * arrays["corrective"][3]):
            return False
        ids = choices(w)
        return all(ids[k] >= 0 and outcomes[k][ids[k]] is not None and outcomes[k][ids[k]]["success"]
                   for k, _, _, _ in groups["corrective"])

    weights = zero.weights.copy()
    blocked, cost_steps, checkpoints = 0, 0, []
    for step in range(steps):
        loss, gradient, _ = terms(weights, "corrective")
        guard_loss, guard_grad, _ = terms(weights, "protective")
        phase = "success"
        if step >= SUCCESS_STEPS and repaired(weights):
            loss, gradient, _ = terms(weights, "cost")
            phase = "cost"
            cost_steps += 1
        else:
            loss, gradient = loss + guard_loss, gradient + guard_grad
        gradient = gradient + l2 * weights
        accepted_update = False
        for backtrack in range(BACKTRACK_STEPS):
            proposal = weights - lr / 2 ** backtrack * gradient
            if not safe(proposal) or (phase == "cost" and not repaired(proposal)):
                continue
            weights, accepted_update = proposal, True
            break
        blocked += not accepted_update
        if step in (0, SUCCESS_STEPS - 1, steps - 1):
            checkpoints.append(dict(step=step + 1, phase=phase, loss=float(loss + l2 / 2 * (weights @ weights)),
                corrective_gaps=terms(weights, "corrective")[2].tolist(), protected_successes=safe(weights)))
    model = SoftUtilityModel(scale, weights, support)
    actual = np.stack([model.deltas(p["x"]) for p in train_pools])
    if not np.allclose(actual, deltas(weights), atol=1e-12, rtol=0.):
        raise AssertionError("Training/deployment delta mismatch")
    actual_choices = fast_selection(base, actual, refs, accepted, official, margins)
    if not np.array_equal(actual_choices, choices(weights)):
        raise AssertionError("Training/deployment top-1 mismatch")
    if not safe(weights):
        raise AssertionError("Final train protection failed")
    return model, dict(schema=SCHEMA, fit_count=1, steps=steps, lr=lr, l2=l2, gain=1.,
        cap=CAP, minimum_switch_margin=MIN_MARGIN, success_steps=SUCCESS_STEPS, cost_steps=cost_steps,
        pairs={k: len(v) for k, v in groups.items()}, skipped=skipped, train_pools=len(train_pools),
        corrective_competition=competition_audit,
        corrective_success_candidates=sum(bool(row["admitted"]) for row in competition_audit),
        corrective_training_pools=len({k for k, _, _, _ in groups["corrective"]}),
        complete_hard_accepted_failed_competitor_supervision=True,
        all_failed_pools_retained=all_failed, protected_train_pools=len(protected), blocked_updates=int(blocked),
        repair_training_reserve=REPAIR_MARGIN_MULTIPLIER, checkpoints=checkpoints,
        corrective_training_target_reached=repaired(weights), training_success_protection=True,
        training_deployment_delta_equivalence=True, training_deployment_top1_equivalence=True,
        validation_used=False, gain_search_performed=False,
        executed_future_in_inputs=False, physical_recovery_probability=False, unknown_fallback_is_failure=False)

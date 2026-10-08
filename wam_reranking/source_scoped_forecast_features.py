"""Additive shared-CURRENT forecast features; no model fit or action selection.

CURRENT and own-forecast geometry are uncalibrated proxies.  The caller must
independently authenticate RGB/maps/localizer bytes before using this API.
Schema and source checks are not observation certificates or physical truth.
Historical cached features and frozen models are never reinterpreted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import json
import numpy as np

from .belief import DEFAULT_GRAPH
from .candidate_effects import parse_candidate_effect
from .contracts import CoarseCause, ConsistencyFactor, PREDICATES
from .reranker import PREDICATE_HARD_THRESHOLDS
from .shared_current_visual_baseline import SCHEMA as FUSION_SCHEMA, audit_shared_current_pool

SCHEMA = "source_scoped_shared_current_forecast_features_v2"
IDENTITY_FIELDS = {"dataset", "suite", "task", "state", "moment", "condition", "block_id", "block_index"}
CHANNEL_NAMES = ("world_anomaly", "execution_anomaly", "observation_corruption",
                 "cross_view_conflict", "coarse_unknown", "direct_route_disagreement", "stage_unresolved")
MAIN_NAMES = ("relation_after", "relation_change_shared_current", "contact_after",
    "contact_change_shared_current", "grasp_after", "grasp_change_shared_current",
    "ordered_grasp_forecast", "ordered_release_forecast", "forecast_visibility",
    "relation_quality", "contact_quality", "cross_view_agreement", "forecast_uncertainty",
    "trajectory_risk", "path_efficiency", "net_upward", "net_lateral", "closed_at_entry",
    "first_close_step_fraction", "release_step_fraction", "ordered_release_present",
    "closed_before_release", "close_strength", "open_strength", "reset_stage", "parser_confidence")
EXPOSURE_SCOPES = ("invalidation_evidence", "unverified_evidence")
INTERACTION_NAMES = tuple(f"{scope}__{c}__{m}" for scope in EXPOSURE_SCOPES
                          for c in CHANNEL_NAMES for m in MAIN_NAMES)
LEARNED_REASONS = {"world_state_consistent", "execution_contact_consistent", "observation_reliable"}
HISTORY_ROOTS = {"world_state_consistent": "target_pose_current",
    "execution_contact_consistent": "execution_consistent", "observation_reliable": "target_visible",
    "measured_command_deviation": "execution_consistent"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _identity(identity):
    if not isinstance(identity, Mapping) or set(identity) != IDENTITY_FIELDS:
        raise ValueError("Complete source identity required; no bare task/sample ID")
    value = dict(identity)
    for key in ("task", "state", "moment", "block_index"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError("Literal nonnegative identity index required")
    if any(type(value[k]) is not str or not value[k] for k in IDENTITY_FIELDS - {"task", "state", "moment", "block_index"}):
        raise ValueError("Literal source-scoped identity text required")
    return value


def _sources(source_sha256):
    if not isinstance(source_sha256, Mapping) or not source_sha256:
        raise ValueError("Independently audited source SHA256 references required")
    for key, value in source_sha256.items():
        if (type(key) is not str or not key or type(value) is not str or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("Literal source SHA256 required")
    return dict(source_sha256)


def _unit(value):
    value = float(value)
    if not np.isfinite(value) or not 0. <= value <= 1.:
        raise ValueError("Finite probabilities/qualities in [0,1] required")
    return value


def _channels(attribution):
    cp = {str(k): _unit(v) for k, v in attribution.class_probs.items()}
    if set(cp) != {c.value for c in CoarseCause} or not np.isclose(sum(cp.values()), 1., atol=1e-5):
        raise ValueError("Complete original class probabilities required")
    return np.array([1.-_unit(attribution.factor(ConsistencyFactor.WORLD_STATE_CONSISTENT)),
        1.-_unit(attribution.factor(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT)),
        1.-_unit(attribution.factor(ConsistencyFactor.OBSERVATION_RELIABLE)),
        float(attribution.evidence_quality.cross_view_conflict), cp[CoarseCause.UNKNOWN.value],
        float(max(cp, key=cp.get) != attribution.projected_cause.value),
        1.-_unit(attribution.factor(ConsistencyFactor.TASK_STAGE_CONSISTENT))], dtype=float)


@dataclass(frozen=True)
class TrainChannelCentering:
    mean: tuple[float, ...]
    fingerprint: str
    source_count: int

    def __post_init__(self):
        if (type(self.source_count) is not int or self.source_count < 1
                or type(self.fingerprint) is not str or len(self.fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in self.fingerprint)):
            raise ValueError("Audited TRAIN centering record required")
        mean = tuple(_unit(v) for v in self.mean)
        if len(mean) != len(CHANNEL_NAMES):
            raise ValueError("Audited TRAIN centering record required")
        object.__setattr__(self, "mean", mean)


def calibrate_train_channels(records):
    """Center PRE channel means on TRAIN only; no labels, fitting or val use."""
    rows, seen = [], set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"identity", "split", "source_sha256", "attribution"}:
            raise ValueError("PRE-only calibration whitelist required; no outcomes/labels")
        if record["split"] != "train":
            raise ValueError("TRAIN-only centering; validation cannot calibrate")
        identity, sources = _identity(record["identity"]), _sources(record["source_sha256"])
        if "attribution" not in sources:
            raise ValueError("Frozen attribution bytes must be source-bound")
        key = _digest(identity)
        if key in seen:
            raise ValueError("Duplicate source-scoped calibration identity")
        seen.add(key)
        if record["attribution"].source_block_id != identity["block_id"]:
            raise ValueError("Attribution/source block mismatch")
        channel = _channels(record["attribution"])
        rows.append(dict(identity=identity, sources=sources, channels=channel.tolist()))
    if not rows:
        raise ValueError("Nonempty TRAIN calibration required")
    rows.sort(key=lambda r: _digest(r["identity"]))
    fingerprint = _digest(dict(schema=SCHEMA, channels=list(CHANNEL_NAMES), rows=rows))
    channels = [row["channels"] for row in rows]
    return TrainChannelCentering(tuple(np.mean(channels, axis=0).tolist()), fingerprint, len(rows))


def _historical_invalidation(history, predicate):
    """Untimed source provenance only; never a current physical certificate."""
    records = []
    for change in history:
        root = HISTORY_ROOTS.get(change.get("reason"))
        path = tuple(change.get("propagation_path", ()))
        if (root is None or change.get("predicate") != predicate
                or change.get("new_value") not in {"false", "unknown"}
                or not path or path[0] != root or path[-1] != predicate):
            continue
        origins = [h for h in history if h.get("predicate") == root
            and h.get("reason") == change["reason"] and h.get("evidence_id") == change.get("evidence_id")
            and h.get("direct") is True and tuple(h.get("propagation_path", ())) == (root,)
            and h.get("new_value") in {"false", "unknown"}]
        if origins:
            strength = _unit(origins[-1].get("new_confidence"))
            records.append(dict(reason=change["reason"], evidence_id=change.get("evidence_id"),
                root=root, root_strength=strength, propagation_path=list(path),
                historical_block_time_authenticated=False, physical_state_certified=False))
    return max((r["root_strength"] for r in records), default=0.), records


def _missing(snapshot, block, no_dag):
    """Soft belief context, not a physical-state estimator or admission gate.

    A recorded invalidation/staleness transition differs from an absent witness.
    A declared observation TRUE can remain soft prior context; it never becomes
    a current certificate just because it is confident or recently initialized.
    """
    if not isinstance(snapshot, Mapping) or set(snapshot) - {"facts", "history", "task_id", "target_object", "receptacle", "task_stage"}:
        raise ValueError("Own variant's audited belief snapshot required")
    facts, history = snapshot.get("facts"), snapshot.get("history")
    if not isinstance(facts, Mapping) or not isinstance(history, (tuple, list)):
        raise ValueError("Explicit facts and history required")
    if set(facts) - set(PREDICATES) or any(not isinstance(h, Mapping) for h in history):
        raise ValueError("Predicate-only facts and explicit source history required")
    if no_dag and any(len(h.get("propagation_path", ())) > 1 and h.get("reason") in
                     LEARNED_REASONS | {"measured_command_deviation"} for h in history):
        raise ValueError("No-DAG variant cannot borrow propagated history")
    scopes = {name: {p: 0. if name == "invalidation_evidence" else 1. for p in PREDICATES}
              for name in EXPOSURE_SCOPES}
    certified, context = [], {}
    for predicate in set(PREDICATES) - set(facts):
        historical_strength, records = _historical_invalidation(history, predicate)
        scopes["invalidation_evidence"][predicate] = historical_strength
        scopes["unverified_evidence"][predicate] = 1.-historical_strength
        context[predicate] = dict(value="unknown", source="absent_PRE_fact", confidence=0.,
            updated_at_block=-1, current_verified=False, invalidation_transition=False,
            retained_declared_TRUE_soft_prior=0., prior_is_current_certificate=False,
            source_evidence_ids=[], latest_matching_reason=None, historical_invalidation_evidence=records,
            historical_evidence_resolved_by_current_verified=False)
    for predicate, fact in facts.items():
        if (not isinstance(fact, Mapping) or set(fact) !=
                {"value", "confidence", "source", "updated_at_block", "evidence_ids"}
                or fact["value"] not in {"true", "false", "unknown"}
                or fact["source"] not in {"observation", "attribution", "candidate_prediction"}
                or type(fact["updated_at_block"]) is not int
                or not -1 <= fact["updated_at_block"] <= block
                or not isinstance(fact["evidence_ids"], (tuple, list))
                or any(type(v) is not str or not v for v in fact["evidence_ids"])):
            raise ValueError("Complete PRE belief fact source/time contract required")
        confidence, ids = _unit(fact["confidence"]), tuple(fact["evidence_ids"])
        latest = next((h for h in reversed(history) if h.get("predicate") == predicate
                       and h.get("evidence_id") in ids), None)
        if latest is not None and (latest.get("new_value") != fact["value"]
                or "new_confidence" not in latest
                or not np.isclose(_unit(latest["new_confidence"]), confidence, rtol=0., atol=1e-8)):
            raise ValueError("Latest source history and live fact disagree")
        observed = (latest is not None and latest.get("reason") == "current_observation"
            and latest.get("new_value") == "true" and latest.get("direct") is True
            and tuple(latest.get("propagation_path", ())) == (predicate,))
        verified = (fact.get("value") == "true" and fact.get("source") == "observation"
            and fact.get("updated_at_block") == block and ids and "initial_observation" not in ids
            and observed and confidence >= PREDICATE_HARD_THRESHOLDS.get(predicate, .75))
        invalidation = (not verified and fact["source"] == "attribution" and latest is not None
            and latest.get("reason") in LEARNED_REASONS | {"measured_command_deviation"}
            and latest.get("new_value") == fact["value"] and fact["value"] in {"false", "unknown"})
        # This confidence describes the recorded belief transition, not whether
        # an object physically moved. UNKNOWN never becomes physical FALSE.
        historical_strength, historical_records = _historical_invalidation(history, predicate)
        # A normal/consistent write or a candidate TRUE does not delete old
        # invalidation. Its time is not authenticated by BeliefChange, so it
        # remains soft provenance until an independently sourced CURRENT TRUE.
        inval = 0. if verified else max(confidence if invalidation else 0., historical_strength)
        prior = confidence if not verified and fact["source"] == "observation" and fact["value"] == "true" else 0.
        scopes["invalidation_evidence"][predicate] = inval
        scopes["unverified_evidence"][predicate] = 0. if verified else 1.-max(inval, prior)
        context[predicate] = dict(value=fact["value"], source=fact["source"], confidence=confidence,
            updated_at_block=fact["updated_at_block"], current_verified=bool(verified),
            invalidation_transition=bool(invalidation), retained_declared_TRUE_soft_prior=prior,
            prior_is_current_certificate=False, source_evidence_ids=list(ids),
            latest_matching_reason=None if latest is None else latest.get("reason"))
        context[predicate]["historical_invalidation_evidence"] = historical_records
        context[predicate]["historical_evidence_resolved_by_current_verified"] = bool(verified and historical_records)
        if verified:
            certified.append(predicate)
    return scopes, sorted(certified), context


def _exposure(requirements, root, missing, no_dag):
    denominator = sum(float(v) for v in requirements.values())
    if denominator <= 0.:
        return 0.
    if root is None:
        return sum(float(v)*missing.get(p, 1.) for p,v in requirements.items())/denominator
    paths = {root: (root,)}
    if not no_dag:
        paths.update(DEFAULT_GRAPH.descendants_with_paths(root))
    return sum(float(v)*missing.get(p, 1.)*.85**(len(paths[p])-1)
               for p,v in requirements.items() if p in paths)/denominator


def source_scoped_features(*, attribution, plans, visual_evidences, fusion_diagnostics,
        identity, source_sha256, split, variant, history_context, belief_snapshot,
        centering, relation, mask_learned=False, no_dag=False, close_when_negative=True):
    """Return separate ordinary/main and attribution/interaction matrices.

    Internal ordered parsing prevents legacy effects masquerading as the new
    source contract.  The output is neither a fitted model nor deployable gate.
    Main effects are unchanged by learned-content masking, including contact,
    grasp/release, quality, uncertainty, risk, and command-sequence capacity.
    """
    identity, sources = _identity(identity), _sources(source_sha256)
    if split not in ("train", "val") or not isinstance(variant, str) or not variant:
        raise ValueError("Explicit split and variant identity required")
    if attribution.source_block_id != identity["block_id"]:
        raise ValueError("Attribution/source block mismatch")
    if not isinstance(centering, TrainChannelCentering):
        raise ValueError("TRAIN-only calibrated channel centering required")
    if history_context != dict(identity=identity, variant=variant, no_dag=bool(no_dag),
                               snapshot_sha256=_digest(belief_snapshot)):
        raise ValueError("History must belong to this source and variant, not a borrowed arm")
    if len(plans) == 0 or not (len(plans) == len(visual_evidences) == len(fusion_diagnostics)):
        raise ValueError("Complete aligned own-candidate plan/forecast pool required")
    required_sources = {"current_primary", "current_wrist", "localizer_model", "localizer_code",
                        "shared_fusion_code", "candidate_parser_code", "attribution"}
    required_sources.update(f"candidate_{i}_{field}" for i in range(len(plans))
                            for field in ("plan", "predicted_primary", "predicted_wrist"))
    if set(sources) != required_sources:
        raise ValueError("Exact CURRENT, own-plan/forecast, extractor and attribution source references required")
    if any(d.get("schema") != FUSION_SCHEMA or d.get("physical_certificate") is not False
           or d.get("before_weights_use_forecasts") is not False for d in fusion_diagnostics):
        raise ValueError("New shared-CURRENT fusion schema required; legacy cached BEFORE forbidden")
    fusion_audit = audit_shared_current_pool(visual_evidences, fusion_diagnostics)
    if "task_id" in belief_snapshot and (type(belief_snapshot["task_id"]) is not int
                                         or belief_snapshot["task_id"] != identity["task"]):
        raise ValueError("Belief task/source identity mismatch")
    missing_scopes, certified, belief_context = _missing(belief_snapshot, identity["block_index"], no_dag)
    raw = _channels(attribution)
    centered = raw - np.asarray(centering.mean)
    q = attribution.evidence_quality
    visual_usable = bool((q.primary_reliable or q.wrist_reliable) and not q.cross_view_conflict)
    roots = ("target_pose_current", "execution_consistent", "target_visible", "target_visible", None, None, None)
    main, interactions, exposures, soft_requirements = [], [], [], []
    for cid, (plan, visual) in enumerate(zip(plans, visual_evidences, strict=True)):
        effect = parse_candidate_effect(cid, plan, visual_evidence=visual,
            close_when_negative=close_when_negative, stage_semantics="ordered_release_v1")
        requirements = ({"target_visible": .5, "target_pose_current": .6}
            if str(relation).lower() in {"open", "closed", "articulated"} else dict(effect.required_facts))
        soft_requirements.append(requirements)
        e, a = effect.evidence, np.asarray(plan)
        quality_r = min(effect.confidence, visual.relation_confidence, visual.cross_view_agreement)
        quality_c = min(effect.confidence, visual.contact_confidence, visual.cross_view_agreement)
        contact = lambda distance: 1.-float(np.clip(distance, 0., 1.))
        goal_after = 1.-visual.relation_score_after if relation == "closed" else visual.relation_score_after
        goal_before = 1.-visual.relation_score_before if relation == "closed" else visual.relation_score_before
        if relation == "articulated":
            goal_after = goal_before = 0.  # No invented direction for a joint.
        close = -a[:,6] if close_when_negative else a[:,6]
        first = int(np.flatnonzero(close > .25)[0]) if np.any(close > .25) else 16
        release = int(e["release_index"])
        row = np.array([goal_after, goal_after-goal_before, contact(visual.target_gripper_distance_after),
            contact(visual.target_gripper_distance_after)-contact(visual.target_gripper_distance_before),
            visual.grasp_support_after, visual.grasp_support_after-visual.grasp_support_before,
            e.get("predicted_grasp_support",0.), e.get("predicted_release_support",0.),
            visual.visibility_confidence, quality_r, quality_c, visual.cross_view_agreement,
            np.clip(1.-min(quality_r,quality_c)+e["trajectory_risk"],0.,1.),
            e["trajectory_risk"], e["trajectory_path_efficiency"], e["net_upward"], e["net_lateral"],
            e["closed_at_entry"], first/16., release/16., float(release<16 and first<release),
            e["closed_before_release"], np.clip(e["close_strength"],0.,1.), np.clip(e["open_strength"],0.,1.),
            float(effect.stage.value in {"observe","approach","grasp"}), effect.confidence], dtype=float)
        channel = centered.copy()
        # Missing/unusable visual sources cannot supply learned physical claims.
        # Ordinary own-forecast capacity and the external command channel stay.
        if not visual_usable:
            channel[[0, 2, 3]] = 0.
        if not q.execution_reliable:
            channel[1] = 0.
        exposure = {scope: np.array([_exposure(requirements, root,
                    missing_scopes[scope], no_dag) for root in roots]) for scope in EXPOSURE_SCOPES}
        interaction = np.concatenate([np.outer(channel*exposure[scope], row).reshape(-1)
                                      for scope in EXPOSURE_SCOPES])
        if mask_learned:
            interaction[:] = 0.
        main.append(row); interactions.append(interaction)
        exposures.append({scope: value.tolist() for scope, value in exposure.items()})
    main, interactions = np.stack(main), np.stack(interactions)
    if not np.isfinite(main).all() or not np.isfinite(interactions).all():
        raise ValueError("Finite source-scoped forecast matrices required")
    main.setflags(write=False); interactions.setflags(write=False)
    return main, interactions, dict(schema=SCHEMA, identity=identity, split=split, variant=variant,
        source_sha256=sources, fusion_audit=fusion_audit, parser="ordered_release_v1",
        main_names=list(MAIN_NAMES), interaction_names=list(INTERACTION_NAMES),
        centering_fingerprint=centering.fingerprint, centering_source_count=centering.source_count,
        raw_channels=raw.tolist(), centered_channels=centered.tolist(), dependency_exposure=exposures,
        exposure_scopes=list(EXPOSURE_SCOPES), belief_context=belief_context,
        soft_required_facts=soft_requirements,
        articulated_soft_requirements_match_existing_typed_gate=True,
        absent_fact_predicates_are_unverified=sorted(set(PREDICATES)-set(belief_snapshot["facts"])),
        invalidation_evidence_is_not_physical_FALSE=True,
        unknown_evidence_is_not_physical_invalidation=True,
        retained_prior_cannot_pass_current_hard_gate=True,
        own_variant_history_identity_bound=True, no_dag=bool(no_dag), mask_learned=bool(mask_learned),
        learned_quality_flags_are_channel_metadata=True,
        learned_visual_channel_usable=visual_usable,
        ordinary_forecast_source_contract_audited=True,
        ordinary_main_gated_by_learned_quality=False,
        independent_rgb_and_extractor_byte_authentication_required=True,
        external_command_rule_unchanged=True,
        learned_execution_contact_probability_is_not_certificate=True,
        reliable_command_record_is_not_contact_certificate=True,
        execution_channel_soft_usable=bool(q.execution_reliable),
        visual_quality_cannot_erase_execution_soft_channel=True,
        full_ordinary_main_capacity_preserved=True, certified_prior_facts=certified,
        current_physical_certificates_generated=0, observation_registry_populated=False,
        proxy_geometry_is_physical_truth=False, live_belief_mutated=False, prediction_writes=0,
        actual_after_used=False, terminal_labels_used=False, model_fitted=False, deployed=False)

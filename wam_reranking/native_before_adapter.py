"""Independent, outcome-free native-before adapter; frozen outputs unchanged.

The final *frozen routed* class is the sole class projection here. Auxiliary
unknown probability is retained as a diagnostic, not reapplied as a second
classifier. Selected-class confidence is preserved, including small values.
This adapter supplies no model inference, labels, fitting or empirical gate.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from collections.abc import Mapping

import numpy as np

from .contracts import (AttributionOutput, CoarseCause, ConsistencyFactor, EvidenceQuality)

SCHEMA = "native_actual_before_control_adapter_v1"
ATTRIBUTION_SCHEMA = "frozen_final_route_attribution_adapter_v1"
RAW_FACTORS = ("visual_evidence_corrupted", "object_or_environment_state_changed",
               "execution_or_contact_deviated", "cross_view_conflict")
QUALITY_KEYS = ("primary_reliable", "wrist_reliable", "execution_reliable",
                "cross_view_conflict", "observer_evidence_available")
FORBIDDEN = frozenset(("actual_after", "outcomes", "success", "terminal_success",
    "labels", "teacher", "private_physics", "simulator_gt", "actual_observation_evidence"))


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode()).hexdigest()


def _prob(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + " requires a literal finite probability")
    return float(value)


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(name + " requires a literal bool")
    return value


def _no_future(value):
    if isinstance(value, Mapping):
        if any(str(k).lower() in FORBIDDEN for k in value):
            raise ValueError("Actual future/outcome/teacher is not a before input")
        for child in value.values():
            _no_future(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _no_future(child)


def convert_frozen_attribution(record, source_block_id):
    """Preserve the selected final route and its original direct confidence.

    cause_resolved is an explicitly named final-route flag, NOT a calibrated
    auxiliary probability. No scalar is increased or fitted. A conflicting
    frozen final class is rejected rather than silently reprojected.
    """
    if not isinstance(record, Mapping) or not isinstance(source_block_id, str) or not source_block_id:
        raise ValueError("Raw frozen prediction and source-scoped before block required")
    _no_future(record)
    cause = CoarseCause(record["predicted"])
    cp = record["class_probs"]
    if not isinstance(cp, Mapping) or set(cp) != {c.value for c in CoarseCause}:
        raise ValueError("All original direct coarse probabilities required")
    classes = {k: _prob(v, "direct class " + k) for k, v in cp.items()}
    if not math.isclose(sum(classes.values()), 1., abs_tol=1e-5):
        raise ValueError("Original direct coarse distribution must sum to1")
    confidence = _prob(record["confidence"], "selected final route confidence")
    if not math.isclose(confidence, classes[cause.value], rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("Selected-label confidence does not match direct[selected]; no replacement allowed")
    fp = record["factor_prob"]
    if not isinstance(fp, Mapping) or set(fp) != set(RAW_FACTORS):
        raise ValueError("All original four anomaly-head probabilities required")
    raw_factors = {k: _prob(v, "anomaly factor " + k) for k, v in fp.items()}
    unknown = _prob(record["unknown_prob"], "auxiliary unknown")
    normal = _prob(record["normal_prob"], "auxiliary normal")
    if record.get("unknown_threshold") != .64:
        raise ValueError("Frozen auxiliary unknown threshold metadata changed")
    if "factor_threshold" in record and record["factor_threshold"] != .5:
        raise ValueError("Frozen factor threshold metadata changed")
    conflict = raw_factors["cross_view_conflict"] >= .5
    if conflict and cause is not CoarseCause.UNKNOWN:
        raise ValueError("Frozen final route conflicts with cross-view-conflict contract; no silent override")
    if cause is CoarseCause.NORMAL and any(p >= .5 for p in raw_factors.values()):
        raise ValueError("Frozen final normal conflicts with its all-factor-negative contract")
    observation = 1. - max(raw_factors["visual_evidence_corrupted"], raw_factors["cross_view_conflict"])
    quality = EvidenceQuality(
        _bool(record["primary_available"], "primary availability") and observation >= .5,
        _bool(record["wrist_available"], "wrist availability") and observation >= .5,
        _bool(record["action_record_available"], "execution record availability"), conflict)
    factors = {
        ConsistencyFactor.OBSERVATION_RELIABLE.value: observation,
        ConsistencyFactor.WORLD_STATE_CONSISTENT.value: 1. - raw_factors["object_or_environment_state_changed"],
        ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value: 1. - raw_factors["execution_or_contact_deviated"],
        ConsistencyFactor.TASK_STAGE_CONSISTENT.value: 1.,
        ConsistencyFactor.CAUSE_RESOLVED.value: float(cause is not CoarseCause.UNKNOWN),
    }
    from .contracts import TriValue
    states = {k: TriValue.TRUE if p >= .5 else TriValue.FALSE for k, p in factors.items()}
    attr = AttributionOutput(factors, classes, cause, confidence,
        -sum(p * math.log(p) for p in classes.values() if p), quality, source_block_id,
        states, {k: max(p, 1.-p) for k, p in factors.items()})
    legacy_cause = CoarseCause.UNKNOWN if conflict or unknown >= .64 else cause
    audit = dict(schema=ATTRIBUTION_SCHEMA, raw_record_payload_sha256=digest(dict(record)),
        frozen_final_cause=cause.value, selected_direct_confidence=confidence,
        scalar_confidence_unchanged=True, final_route_is_sole_projection=True,
        auxiliary_unknown_probability=unknown, auxiliary_normal_probability=normal,
        original_direct_class_probs=classes, original_anomaly_factor_probs=raw_factors,
        cause_resolved_semantics="final_frozen_route_resolvedflag",
        cause_resolved_is_calibrated_probability=False,
        legacy_converter_projected_cause=legacy_cause.value,
        semantic_difference_from_legacy_converter=legacy_cause is not cause,
        thresholds_unchanged=True, prediction_or_model_modified=False,
        confidence_recalibrated=False)
    return attr, audit


def _npy(ref, reader, shape, name, rgb=False):
    path = reader.path(ref)
    if path.suffix.lower() != ".npy":
        raise ValueError(name + " must be a standalone NPY")
    x = np.load(path, allow_pickle=False)
    if x.shape != shape or not np.isfinite(x).all() or (rgb and x.dtype != np.uint8):
        raise ValueError(name + " actual shape/dtype/value mismatch")
    if not rgb and x.dtype.kind not in "fiu":
        raise ValueError(name + " must be numeric")
    return x


def load_native_before_pool(descriptor, reader):
    """Read actual saved before/own-prediction bytes, never native execution Y.

    All candidates must carry identical actual query arrays. No trustworthy
    historical prior is guessed: this first adapter intentionally supports
    only an explicit uninformative prior. Only genuine before-quality-false
    UNKNOWN/occlusion can activate the separate epistemic boundary16 goal.
    Canonical narrowed JSON sidecars and temporal/lineage producer audit remain
    separate requirements; returning a ControlPool does not pass the whole gate.
    """
    # The cloud source-only converter uses an older frozen namespace. Loading
    # a final-route sidecar must not eagerly import new producer/control code.
    from .attribution_control_factory import ControlPool
    from .contracts import BeliefState, CandidateVisualEvidence, Stage
    from .candidate_effects import parse_candidate_effect
    from .evidence_residual import typed_gate_effect
    from .recovery_contract import VISUAL_EVIDENCE_KEYS
    _no_future(descriptor)
    required = {"schema", "identity", "pool_id", "split", "block_index", "before",
        "quality_source", "attribution_source", "binding_source", "candidates", "budget",
        "prior_policy", "before_lineage", "ranking_reference"}
    if not isinstance(descriptor, Mapping) or set(descriptor) != required or descriptor["schema"] != SCHEMA:
        raise ValueError("Complete new native source descriptor required")
    if descriptor["prior_policy"] != "explicit_uninformative_no_verified_history":
        raise ValueError("This adapter may not invent or import an uncertified TRUE prior")
    reference = descriptor["ranking_reference"]
    if reference != dict(kind="official_value_reference",origin="own_query_official_value",frozen_model_applied=False):
        raise ValueError("Explicit temporary official-value reference required; actual frozen-backbone producer not supplied")
    identity = dict(descriptor["identity"])
    if set(identity) != {"dataset", "suite", "task", "state", "observed_block_id"}:
        raise ValueError("Five-field exact source pool identity required")
    if (any(type(identity[k]) is not int or identity[k] < 0 for k in ("task","state"))
        or any(type(identity[k]) is not str or not identity[k] for k in ("dataset","suite","observed_block_id"))
        or type(descriptor["block_index"]) is not int or descriptor["block_index"] < 0):
        raise ValueError("Literal complete source identity/block index required")
    if descriptor["split"] not in ("train", "val"):
        raise ValueError("Frozen split train/val required")
    before = descriptor["before"]
    if not isinstance(before, Mapping) or set(before) != {"primary", "wrist", "proprio"}:
        raise ValueError("Actual fresh query RGB/proprio sources required")
    actual = {v: _npy(before[v], reader, (9,) if v == "proprio" else (256,256,3),
                     "fresh actual query " + v, rgb=v != "proprio") for v in before}
    quality_record = reader.json(descriptor["quality_source"])
    if (quality_record.get("schema") != "actual_before_deployment_observer_quality_v1"
        or quality_record.get("role") != "deployable_before_observer"
        or quality_record.get("actual_after_used") is not False
        or quality_record.get("private_teacher_used") is not False):
        raise ValueError("Actual-before frozen deployment observer only; no AFTER quality")
    quality = quality_record["quality"]
    if set(quality) != set(QUALITY_KEYS) or any(type(quality[k]) is not bool for k in QUALITY_KEYS):
        raise ValueError("Literal five-field before observer quality required")
    if (quality_record.get("identity") != identity or type(quality_record.get("relative_step")) is not int
        or quality_record["relative_step"] != 0
        or quality_record.get("actual_before_sources") != before):
        raise ValueError("Before observer must bind its exact source identity/time/RGB/proprio")
    for name in ("observer_code_source", "observer_model_source"):
        reader.path(quality_record[name])
    record = reader.json(descriptor["attribution_source"])
    attr, semantic_audit = convert_frozen_attribution(record, identity["observed_block_id"])
    if {k:quality[k] for k in QUALITY_KEYS[:4]} != asdict(attr.evidence_quality):
        raise ValueError("Frozen predicted quality and actual before observer quality disagree")
    if quality_record.get("predicted_cause") != attr.projected_cause.value:
        raise ValueError("Before observer goal used another class projection; regenerate independent sidecar")
    binding = reader.json(descriptor["binding_source"])
    if set(binding) != {"language", "target", "anchor", "relation"} or any(not isinstance(v,str) or not v for v in binding.values()):
        raise ValueError("Frozen language/entity/relation binding required")
    line = descriptor["before_lineage"]
    if not isinstance(line, Mapping) or set(line) != {"snapshot_source", "query_source"}:
        raise ValueError("Actual shared snapshot and parent-query metadata sources required")
    reader.path(line["snapshot_source"])
    query = reader.json(line["query_source"])
    if (query.get("schema") != "native_shared_query_before_metadata_v1" or query.get("identity") != identity
        or query.get("actual_before_sources") != before
        or query.get("full_runtime_snapshot_sha256") != line["snapshot_source"]["sha256"]
        or query.get("before_quality_sha256") != descriptor["quality_source"]["sha256"]
        or query.get("before_attribution_sha256") != descriptor["attribution_source"]["sha256"]
        or not isinstance(query.get("query_observation_sha256"),str)
        or len(query["query_observation_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in query["query_observation_sha256"])):
        raise ValueError("Shared snapshot/observation/quality/attribution query-parent binding required")
    candidates = descriptor["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 4 or sorted(c.get("candidate_id") for c in candidates) != [0,1,2,3]:
        raise ValueError("Native complete K4 unique membership required")
    candidates = sorted(candidates, key=lambda c:c["candidate_id"])
    plans, effects, values, scores, visual_records = [], [], [], [], []
    for item in candidates:
        if set(item) != {"candidate_id", "query_inputs", "planned_actions_source",
                        "predicted_primary_source", "predicted_wrist_source", "candidate_payload_source",
                        "backbone_score"}:
            raise ValueError("Complete own-candidate before payload/source references required")
        if set(item["query_inputs"]) != set(before):
            raise ValueError("Each own-query RGB/proprio must be saved explicitly")
        for v in before:
            own = _npy(item["query_inputs"][v], reader, actual[v].shape, "own query " + v, rgb=v != "proprio")
            if not np.array_equal(own, actual[v]):
                raise ValueError("Candidate queried a different actual before observation")
        plan = _npy(item["planned_actions_source"], reader, (16,7), "own planned actions")
        for v in ("primary", "wrist"):
            # Frozen Cosmos query RGB is256, native own endpoint prediction
            # is224. Do not resize or rewrite those independently hashed bytes.
            _npy(item["predicted_"+v+"_source"], reader, (224,224,3), "own predicted " + v, rgb=True)
        payload = reader.json(item["candidate_payload_source"])
        _no_future(payload)
        if (payload.get("schema") != "preexecution_native_actual_before_own_prediction_v1"
            or payload.get("actual_after_or_teacher_in_X") is not False
            or payload.get("created_before_any_candidate_execution") is not True):
            raise ValueError("Native before-query candidate payload required")
        for key, ref in (("planned_action_sha256",item["planned_actions_source"]),
                         ("predicted_primary_sha256",item["predicted_primary_source"]),
                         ("predicted_wrist_sha256",item["predicted_wrist_source"])):
            if payload.get(key) != ref["sha256"]:
                raise ValueError("Own candidate payload does not bind its plan/prediction")
        if payload.get("before_quality") != quality_record or payload.get("typed_relation") != binding["relation"]:
            raise ValueError("Candidate before-quality/task relation binding mismatch")
        if payload.get("query_observation_sha256") != query["query_observation_sha256"]:
            raise ValueError("Own candidate prediction is not bound to the shared parent query")
        effect = typed_gate_effect(parse_candidate_effect(item["candidate_id"], plan,
            visual_evidence=CandidateVisualEvidence(**payload["candidate_visual"]), close_when_negative=False), binding["relation"])
        if effect.stage.value != payload.get("stage"):
            raise ValueError("Candidate primary stage mismatch; cannot rewrite stage to Observe")
        visual_records.append({k:dict(value=float(np.clip(effect.evidence[k],-1,1)) if k in effect.evidence else None,
                                     available=k in effect.evidence) for k in VISUAL_EVIDENCE_KEYS})
        value, score = payload["value"], item["backbone_score"]
        if type(value) not in (int,float) or type(score) not in (int,float) or not np.isfinite([value,score]).all():
            raise ValueError("Finite own official value and explicitly declared reference score required")
        if float(score) != float(value):
            raise ValueError("Temporary official-value reference must match own query value; no fabricated0/backbone score")
        plans.append(plan);effects.append(effect);values.append(float(value));scores.append(float(score))
    prior = BeliefState(identity["task"],binding["target"],binding["anchor"],task_stage=Stage.OBSERVE)
    info = quality["observer_evidence_available"] is False and attr.projected_cause in (CoarseCause.UNKNOWN,CoarseCause.VISUAL_OCCLUSION)
    pool = ControlPool(identity,descriptor["pool_id"],descriptor["split"],prior,attr,(),
        tuple(effects),tuple(plans),tuple(values),tuple(scores),descriptor["budget"],descriptor["block_index"],
        relation=binding["relation"], information_acquisition=bool(info),before_observer_available=quality["observer_evidence_available"])
    audit = dict(schema=SCHEMA,content_validated=True,uninformative_prior=True,
        no_true_prior_fabricated=True,observations_count=0,old_true_invalidation_need_available=False,
        attribution_semantics=semantic_audit,epistemic_goal16_eligible=bool(info),
        ranking_reference=reference,
        factory_backbone_score_slot_is_temporary_official_value_reference=True,
        actual_frozen_learned_backbone_applied=False,
        whole_original_frozen_backbone_requirement_satisfied=False,
        physical_active_needs_not_invented=True,source_descriptor_payload_sha256=digest(dict(descriptor)),
        canonical_sidecar_payloads=dict(quality=quality,binding=binding,candidate_visual=visual_records),
        fresh_actual_query_proprio_verified=True,all_own_query_arrays_equal=True,
        input_shape_contract=dict(actual_query_RGB=[256,256,3],own_predicted_RGB=[224,224,3],
            fresh_query_proprio=[9],planned_actions=[16,7],images_resized_or_rewritten=False),
        actual_future_read=False,models_loaded=False,training_ready=False,whole_gate_pass=False,
        remaining_producer_requirements=["freeze narrowed canonical JSON sidecars and their own source refs",
            "independent parent-query/time/code/model/source-scope producer audit",
            "actual original frozen ranking-backbone inference and score provenance before whole-run admission",
            "complete actual full/masked/shuffled/no-DAG/effect-only control payload and content audit",
            "derive actual nonzero active manifest from these before contexts, never from Y",
            "independent legal Y admission for every genuinely active learned target"])
    return pool, audit

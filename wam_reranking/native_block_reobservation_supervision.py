"""Finite native-block observer Y, separate from current/physical predicates.

The optional observer goal is frozen from deployment-observer BEFORE evidence.
It never changes native action stage, permits an entry gate, or backfills an
earlier use time. Private render roles associate actual RGB measurements with
a target; they are not a deployment quality source or an input feature.
"""
from __future__ import annotations

import math
import hashlib
import json
import numpy as np

SCHEMA = "native_block_secondary_observer_goal_silver_v1"
PURPOSE = "task_with_observation_recovery"
TARGET = "akita_black_bowl_1"
ATTRIBUTOR_SHA = "68b2b638f73c959fb170d39caecad63572427f7716f78fe4c9459ccaf8012bec"
OBSERVER_CODE_SHA = "c6213b301b98fc90fadb3dfe0fe90f4e2d40b84f04c9b422dbce4072720b3654"
IDENTITY = ("dataset", "suite", "task", "state", "candidate_id", "observed_block_id")


def identity(value):
    if not isinstance(value, dict) or set(value) != set(IDENTITY):
        raise ValueError("Complete six-field identity required")
    if any(type(value[k]) is not int or value[k] < 0 for k in ("task", "state", "candidate_id")):
        raise ValueError("Literal source-scoped integer identity required")
    if any(not isinstance(value[k], str) or not value[k] for k in ("dataset", "suite", "observed_block_id")):
        raise ValueError("Nonempty dataset/suite/block identity required")


def deployment_observer_quality(subject_maps, *, image_quality, predicted_cause, boundary="before"):
    """Actual RGB observer. AFTER may only be stored as private Y, never X."""
    from wam_reranking.observed_recovery import map_endpoint
    maps = np.asarray(subject_maps)
    if boundary not in ("before","after") or maps.ndim != 3 or maps.shape[0] != 2 or not np.isfinite(maps).all():
        raise ValueError("Exactly actual primary/wrist maps at one declared boundary required")
    fields = ("primary_reliable", "wrist_reliable", "execution_reliable", "cross_view_conflict")
    if not isinstance(image_quality, dict) or set(image_quality) != set(fields) or any(type(image_quality[k]) is not bool for k in fields):
        raise ValueError("Literal deployment-observer quality only")
    if predicted_cause not in ("normal", "visual_occlusion", "object_shift", "execution_contact_deviation", "unknown"):
        raise ValueError("Predicted before cause required")
    endpoints = [map_endpoint(m) for m in maps]
    available = not image_quality["cross_view_conflict"] and any(
        image_quality[v + "_reliable"] and endpoints[i]["usable"] for i, v in enumerate(("primary", "wrist")))
    quality = dict(image_quality, observer_evidence_available=bool(available))
    return dict(schema=("actual_after_frozen_deployment_observer_quality_v1" if boundary=="after" else "actual_before_deployment_observer_quality_v1"), quality=quality,
        map_endpoints=endpoints, predicted_cause=predicted_cause,
        role=("offline_supervision_only" if boundary=="after" else "deployable_before_observer"), actual_after_used=boundary=="after",private_teacher_used=False,
        actual_after_or_private_teacher_used=boundary=="after",
        observer_definition="Frozen relevance-map localization quality>=.5 in a predicted reliable actual view; not target physical absence")


def secondary_goal(before_quality, *, predicted_cause, purpose=PURPOSE):
    quality_keys=("primary_reliable","wrist_reliable","execution_reliable","cross_view_conflict","observer_evidence_available")
    if (purpose != PURPOSE or not isinstance(before_quality, dict) or set(before_quality)!=set(quality_keys) or
            any(type(before_quality[k]) is not bool for k in quality_keys) or
            before_quality.get("observer_evidence_available") is not False or
            predicted_cause not in ("unknown", "visual_occlusion")):
        raise ValueError("Frozen deployment-before epistemic deficit required")
    return dict(predicate="observer_evidence_available", deadline=16, kind="epistemic_goal",
        purpose=PURPOSE, source="before_epistemic_insufficiency", native_stage_unchanged=True,
        goal_never_unlocks_entry_or_earlier_use=True)


def check_target_binding(frame, *, target=TARGET):
    mapping = frame.get("observation_evidence", {}).get("private_collision_vs_visual_mapping", {}).get("target")
    if not isinstance(mapping, list) or not mapping or any(not isinstance(g, dict) for g in mapping):
        raise ValueError("Exact target visual-role supervision association required")
    names = [g.get("body_name") for g in mapping]
    if not all(isinstance(n, str) and (n == target or n.startswith(target + "_")) for n in names):
        raise ValueError("Actual RGB certificate role maps to a different target instance")
    return dict(target=target, binding_association_only=True, actual_RGB_identity_still_required=True)


def _qc(frame):
    record = frame.get("observation_evidence", {}).get("same_candidate_repeat_qc", {})
    if record.get("physics_passed") is not True or record.get("all_contact_atoms_match") is not True:
        return False
    for key, maximum in (("qpos_qvel_max_abs", 1e-3), ("qpos_qvel_p95", 1e-4)):
        v = record.get(key)
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= maximum:
            return False
    return True


def derive_native_goal(frames, before_source_frame, *, source_identity, native_identity,
        verified_prefix_evidence, helpers, prefix_helpers, native_stage, frozen_goal, before_deployment_quality,
        actual_after_deployment_quality,actual_after_quality_source,native_query_observation_sha256,actual_after_subject_maps):
    identity(source_identity); identity(native_identity)
    if native_stage not in ("observe", "approach", "grasp", "lift", "transport", "place", "uncertain"):
        raise ValueError("Original typed native stage required, not a rewritten Observe stage")
    if frozen_goal != secondary_goal(before_deployment_quality["quality"],
            predicted_cause=before_deployment_quality["predicted_cause"]):
        raise ValueError("Goal must match the immutable deployment-before declaration")
    prefix_helpers.validate_prefix_proof(verified_prefix_evidence, source_identity, 16, helpers)
    if before_source_frame.get("step") != 16 or not _qc(before_source_frame):
        raise ValueError("Before is actual source real step16 with prefix-only QC")
    binding = check_target_binding(before_source_frame)
    before_views = {v: prefix_helpers.view_available(before_source_frame, v, helpers) for v in ("primary", "wrist")}
    if any(before_views.values()):
        raise ValueError("Fixed before is no longer certificate-insufficient; no future selection or source replacement")
    if not isinstance(frames, list) or len(frames) != 17 or [f.get("step") for f in frames] != list(range(17)):
        raise ValueError("17 actual complete aligned frames; short endpoints are not padded")
    if not _qc(frames[0]) or frames[0].get("observation_evidence",{}).get("step")!=0:
        raise ValueError("Actual local0 also requires own repeat QC, never cached-query evidence")
    rows = [dict(step=0,original_source_step=16,views=before_views,value=False,mask=True,
        actual_post_full_native_block=True,cached_query_frame=False)]
    for f in frames[1:]:
        check_target_binding(f)
        if f.get("observation_evidence", {}).get("step") != f["step"] or not _qc(f):
            raise ValueError("Actual own-candidate frame/QC alignment required")
        views = {v: prefix_helpers.view_available(f, v, helpers) for v in ("primary", "wrist")}
        rows.append(dict(step=f["step"],views=views,actual_rgb_certificate_available=any(views.values()),value=None,mask=False,
            target_physical_absence_inferred=False))
    observed_after=rows[16]
    record=actual_after_deployment_quality
    valid_sha=lambda x:isinstance(x,str) and len(x)==64 and not set(x)-set("0123456789abcdef")
    def artifact(ref):
        return (isinstance(ref,dict) and isinstance(ref.get("path"),str) and bool(ref["path"]) and valid_sha(ref.get("sha256")))
    if (not isinstance(record,dict) or record.get("schema")!="actual_after_frozen_deployment_observer_quality_v1" or
            record.get("role")!="offline_supervision_only" or record.get("private_teacher_used") is not False or
            record.get("actual_after_used") is not True or record.get("identity")!=native_identity or
            type(record.get("step")) is not int or record["step"]!=16 or
            type(record.get("quality",{}).get("observer_evidence_available")) is not bool or
            record.get("frozen_attributor_model_sha256")!=ATTRIBUTOR_SHA or
            not valid_sha(native_query_observation_sha256) or record.get("query_observation_sha256")!=native_query_observation_sha256 or
            record.get("full_source_prefix_runtime_sha256")!=verified_prefix_evidence["new_runtime_snapshot_sha256"] or
            not artifact(actual_after_quality_source) or not artifact(record.get("actual_attribution_source")) or
            not artifact(record.get("subject_maps_source")) or
            not artifact(record.get("observer_code_source")) or record["observer_code_source"]["sha256"]!=OBSERVER_CODE_SHA or
            not artifact(record.get("observer_model_source")) or record["observer_model_source"]["sha256"]!=ATTRIBUTOR_SHA):
        raise ValueError("Same frozen deployment observer on identity/time-bound actual AFTER RGB required")
    image_sources=record.get("actual_array_sources",{})
    if set(image_sources)!={"primary","wrist"} or any(not artifact(image_sources[v]) or image_sources[v].get("frame_index")!=16 for v in image_sources):
        raise ValueError("Both actual AFTER RGB arrays must bind their source bytes and step16")
    rgb_sources=record.get("actual_after_rgb_sources",{})
    if set(rgb_sources)!={"primary","wrist"} or any(not artifact(rgb_sources[v]) for v in rgb_sources):
        raise ValueError("Two immutable actual AFTER HWC RGB sources required")
    binding={k:record[k] for k in ("identity","step","actual_after_rgb_sources","subject_maps_source","observer_code_source","observer_model_source")}
    lineage=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if record.get("observer_input_lineage_sha256")!=lineage:
        raise ValueError("Actual AFTER frozen observer input/model lineage binding required")
    image_quality={k:v for k,v in record["quality"].items() if k!="observer_evidence_available"}
    recomputed=deployment_observer_quality(actual_after_subject_maps,image_quality=image_quality,predicted_cause=record.get("predicted_cause"),boundary="after")
    if recomputed["quality"]!=record["quality"]:
        raise ValueError("Actual map_endpoint recomputation differs from frozen after quality")
    deploy_available=record["quality"]["observer_evidence_available"]
    if deploy_available is False and observed_after["actual_rgb_certificate_available"] is True:
        value,mask,reason=None,False,"actual_RGB_certificate_and_frozen_deployment_observer_conflict"
    elif deploy_available is False:
        value,mask,reason=False,True,"explicit_actual_after_deployment_observer_unavailable_with_complete_own_QC"
    elif observed_after["actual_rgb_certificate_available"] is True:
        value,mask,reason=True,True,"actual_after_deployment_observer_usable_and_actual_target_RGB_certificate"
    else:
        value,mask,reason=None,False,"deployment_observer_usable_but_actual_RGB_identity_certificate_missing"
    after=dict(observed_after,value=value,mask=mask,goal_value=value,goal_mask=mask,goal_reason=reason,
        deployment_observer_available=deploy_available)
    rows[16]=after
    return dict(schema=SCHEMA,role="offline_supervision_only",identity=dict(native_identity),
        source_identity=dict(source_identity),native_stage=native_stage,purpose=PURPOSE,
        frozen_goal=frozen_goal,before=rows[0],after=after,frames=rows,
        secondary_observer_goal=dict(predicate="observer_evidence_available",deadline=16,kind="epistemic_goal",
            value=value,mask=mask,reason=reason),binding=binding,
        actual_after_deployment_quality=record,
        actual_after_quality_source=actual_after_quality_source,
        observer_quality_certificate=dict(after_quality_source=actual_after_quality_source,
            actual_rgb_measurement_verified=True,no_physical_absence_claim=True,actual_rgb_observable=observed_after["actual_rgb_certificate_available"]),
        semantic_conflict_resolved=dict(value=None,mask=False),
        no_future_backfill_or_predicate_gate_unlock=True,actual_after_Y_only=True,
        physical_predicate_mapping_admitted=False,ranking_training_ready=False)


def pool_contrast(rows):
    """Complete same-source K4 endpoint contrast; no new source on failure."""
    if not isinstance(rows, list) or len(rows) != 4 or sorted(r.get("candidate_id") for r in rows) != list(range(4)):
        raise ValueError("Complete original native K4 pool required")
    if len({r.get("query_observation_sha256") for r in rows}) != 1 or any(
            not isinstance(r.get("query_observation_sha256"), str) or len(r["query_observation_sha256"]) != 64 for r in rows):
        raise ValueError("Exact common actual-before query hash required")
    for r in rows:
        identity(r.get("identity"))
        if r["identity"]["candidate_id"]!=r["candidate_id"]:raise ValueError("Native CID and full identity disagree")
    source_keys={tuple(r["identity"][k] for k in IDENTITY if k!="candidate_id") for r in rows}
    if len(source_keys)!=1:raise ValueError("Cannot form a native contrast across source tasks/states/datasets/blocks")
    qualified = [r for r in rows if r.get("goal_mask") is True and type(r.get("goal_value")) is bool and r.get("complete") is True]
    pos = sum(r["goal_value"] for r in qualified); neg = len(qualified) - pos
    valid = len(qualified) == 4 and pos > 0 and neg > 0
    return dict(qualified_goal_contrast=valid,qualified_candidates=len(qualified),positives=pos,negatives=neg,
        stop_before_next_pool=not valid,no_source_seed_or_candidate_replacement=True,
        training_ready=False,training_started=False)

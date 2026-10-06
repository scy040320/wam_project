"""Actual post-prefix observer-availability Y; no cached-query substitution.

The before label reads a verified historical postaction source frame and its
causal tracks only. New arm execution frames never backfill that before. This
auxiliary target is not physical target absence, semantic conflict resolution,
native candidate recovery, or a preexecution ranking feature.
"""
from __future__ import annotations

import math

SCHEMA = "post_prefix_reobservation_availability_silver_v1"
ARMS = ("hold", "peek_left", "peek_right", "peek_up")
IDENTITY = ("dataset", "suite", "task", "state", "candidate_id", "observed_block_id")


def _finite(x):
    return type(x) in (int,float) and math.isfinite(x)


def _identity(x):
    if not isinstance(x,dict) or set(x)!=set(IDENTITY):
        raise ValueError("Full source-scoped candidate identity required")
    if any(type(x[k]) is not int or x[k]<0 for k in ("task","state","candidate_id")) or any(
        not isinstance(x[k],str) or not x[k].strip() for k in ("dataset","suite","observed_block_id")):
        raise ValueError("Invalid source identity")


def _sha(x):
    return isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdefABCDEF" for c in x)


def _qc(frame):
    x=frame["observation_evidence"].get("same_candidate_repeat_qc",{})
    return (x.get("physics_passed") is True and x.get("all_contact_atoms_match") is True and
        _finite(x.get("qpos_qvel_max_abs")) and 0<=x["qpos_qvel_max_abs"]<=1e-3 and
        _finite(x.get("qpos_qvel_p95")) and 0<=x["qpos_qvel_p95"]<=1e-4)


def view_available(frame, view, helpers):
    """Exactly frozen per-view RGB/texture/tracks/noise gates, no GT atoms."""
    e=frame["observation_evidence"];x=e["views"][view];target=x["roles"]["target"]
    local=target["local_rgb"];global_rgb=x["global_rgb"];pixels=target["rendered_pixels"]
    if type(pixels) is not int or pixels<0:
        raise ValueError("Literal rendered role pixel count required")
    if (global_rgb.get("passed") is not True or not helpers.rgb_pass(global_rgb) or
            x.get("fresh_geometry_registered_to_actual_input") is not True or pixels<16 or
            local.get("passed") is not True or not helpers.rgb_pass(local)):
        return False
    repeat=x.get("repeat_noise",{}).get("target",{})
    if (not _finite(repeat.get("centroid_noise_px")) or repeat["centroid_noise_px"]<0 or
            repeat.get("rgb_repeat",{}).get("passed") is not True or not helpers.rgb_pass(repeat["rgb_repeat"])):
        return False
    intervals=e.get("actual_rgb_temporal_intervals",[])
    for a in intervals:
        start,end=a.get("start_step"),a.get("end_step")
        if type(start) is not int or type(end) is not int or end>frame["step"] or start>=end or start<1:
            raise ValueError("Actual identity cannot be backfilled from a future or cached query interval")
    for a in intervals:
        start,end=a["start_step"],a["end_step"]
        if end!=frame["step"] or a.get("view")!=view:
            continue
        summary=helpers.track_summary(a.get("tracks",{}).get("target"))
        role=helpers.current_role(summary,target)
        noise=a.get("same_candidate_repeat_noise",{}).get("target")
        if summary is not None and role["identity_rgb_unique"] is True and _finite(noise) and noise>=0:
            return True
    return False


def validate_prefix_proof(proof, source_identity, reference_step, helpers):
    """Original applied prefix and RGB/proprio evidence, not teacher-render X."""
    _identity(source_identity)
    if (not isinstance(proof,dict) or proof.get("schema")!="verified_original_postaction_prefix_v1" or
            proof.get("identity")!=source_identity or proof.get("source_hashes_verified") is not True or
            proof.get("reference_step")!=reference_step or proof.get("used_original_steps")!=list(range(1,reference_step+1)) or
            proof.get("original_query_snapshot_used_as_new_start") is not False or
            proof.get("actual_before_arrays_not_teacher_render") is not True):
        raise ValueError("Verified actual candidate prefix proof required")
    if not _sha(proof.get("new_runtime_snapshot_sha256")):
        raise ValueError("New complete post-prefix runtime snapshot must be source hashed")
    hashes=proof.get("source_sha256",{})
    if any(not _sha(hashes.get(k)) for k in ("original_snapshot","applied","primary","wrist","proprio","teacher","old_sidecar")):
        raise ValueError("Original prefix inputs and historical before evidence must be source hashed")
    checks=proof.get("checks")
    if not isinstance(checks,list) or [x.get("step") for x in checks]!=list(range(1,reference_step+1)):
        raise ValueError("Every exact original prefix action must be verified")
    for x in checks:
        if (x.get("passed") is not True or not _finite(x.get("proprio_max_abs")) or not 0<=x["proprio_max_abs"]<=1e-3 or
                any(x.get(v,{}).get("passed") is not True or not helpers.rgb_pass(x[v]) for v in ("primary","wrist"))):
            raise ValueError("Frozen actual dual-RGB/proprio prefix replay gate failed")


def derive_post_prefix_availability(frames, before_source_frame, *, source_identity,
        auxiliary_identity, arm, original_reference_step, verified_prefix_evidence, certificate_helpers):
    """Zero local time is a verified actual post-prefix frame, NOT cached query.

    Negative outcomes concern availability recovery only. No physical absence
    or contradiction is inferred. Persistent primary corruption is not cleared.
    """
    helpers=certificate_helpers;_identity(source_identity);_identity(auxiliary_identity)
    if arm not in ARMS or type(original_reference_step) is not int or not 2<=original_reference_step<=16:
        raise ValueError("Predeclared arm and real historical reference required")
    if any(not callable(getattr(helpers,k,None)) for k in ("rgb_pass","track_summary","current_role")):
        raise ValueError("Frozen certificate helpers required")
    validate_prefix_proof(verified_prefix_evidence,source_identity,original_reference_step,helpers)
    if (before_source_frame.get("step")!=original_reference_step or
            before_source_frame["observation_evidence"].get("step")!=original_reference_step or not _qc(before_source_frame)):
        raise ValueError("Before must bind the verified real historical source frame and prefix-only QC")
    if len(frames)!=17 or [f.get("step") for f in frames]!=list(range(17)) or any(
        f["observation_evidence"].get("step")!=f["step"] for f in frames):
        raise ValueError("17 aligned actual branch frames required; no padding")
    before_views={v:view_available(before_source_frame,v,helpers) for v in ("primary","wrist")}
    before=dict(step=0,original_source_step=original_reference_step,value=any(before_views.values()),mask=True,
        views=before_views,actual_post_prefix_frame=True,cached_query_frame=False,
        source="verified_original_actual_RGB_prefix_with_historical_causal_tracking")
    if before["value"] is not False:
        raise ValueError("Predeclared source no longer evidence-insufficient; retain failure, do not replace source")
    rows=[before]
    for f in frames[1:]:
        if not _qc(f):
            raise ValueError("Same-arm repeat physical/actual-image QC failed; no relabeling")
        views={v:view_available(f,v,helpers) for v in ("primary","wrist")}
        rows.append(dict(step=f["step"],value=any(views.values()),mask=True,views=views,
            actual_post_prefix_frame=True,cached_query_frame=False,target_physical_absence_inferred=False))
    after=rows[16];recovered=before["value"] is False and after["value"] is True
    return dict(schema=SCHEMA,role="constructed_auxiliary_supervision_only",identity=dict(auxiliary_identity),
        source_identity=dict(source_identity),arm=arm,original_reference_step=original_reference_step,
        before=before,after=after,frames=rows,observed_availability_recovered=recovered,
        availability_recovery=dict(value=recovered,mask=True),
        cross_view_conflict_resolved=dict(value=None,mask=False,
            reason="semantic_same_entity_contradiction_and_postprobe_agreement_certificate_missing"),
        before_not_replaced_by_new_arm_or_future_frame=True,original_query_snapshot_not_used_as_branch_start=True,
        source_hashes=verified_prefix_evidence["source_sha256"],
        target_absence_inferred=False,task_predicate_mapping_admitted=False,
        new_before_ranking_X_created=False,after_execution_observations_are_Y_only=True,
        ranking_training_ready=False,native_candidate_pool_benefit_claimed=False)

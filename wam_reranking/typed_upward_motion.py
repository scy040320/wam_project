"""Distinct, offline typed upward-motion silver; never literal lift truth.

An immutable simulator rise is Y only. Visual support requires causal actual
RGB tracks for both typed surfaces and matched actual RGB landmarks in two
calibrated views. Separate view medians, segmentation adjacency and simulator
body points are not cross-view correspondences. V1/V2 contact/lift masks are
never read as a substitute for these witnesses and are never modified.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .recovery_observability_contract import (
    FROZEN_CONTRACT as OBS, SCHEMA_V2, _forbidden, _identity,
    _registration, _hashes,
)
from .temporal_recovery_teacher import FROZEN_CONTRACT as PHYS

SCHEMA = "typed_upward_motion_auxiliary_certificate_v1"
CORRESPONDENCE_BASIS = "audited_actual_RGB_same_surface_landmarks"


def _finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def _xy(x):
    return isinstance(x, (list, tuple)) and len(x) == 2 and all(_finite(v) for v in x)


def _result(value, *reasons, **extra):
    return dict(schema=SCHEMA, value=value, mask=value is True, reasons=list(reasons),
        role="distinct_auxiliary_supervision_only", deployment_allowed=False,
        maps_to_literal_lifted_atom=False, original_masks_and_values_unchanged=True,
        negative_lift_or_grasp_inferred=False, training_ready=False, **extra)


def actual_track_summary(report):
    """Original 3-point, 1px LK/correspondence gates; camera-only subtraction."""
    if (not isinstance(report, Mapping) or report.get("verified") is not True or
            report.get("geometry_motion_support") is not True or
            report.get("evidence_kind") != "actual_RGB_geometry_correspondence"):
        return None
    tracks = report.get("tracks")
    if not isinstance(tracks, list) or len(tracks) < 3 or report.get("count") != len(tracks):
        return None
    actual, projected, seen = [], [], set()
    for point in tracks:
        if (not isinstance(point, Mapping) or not all(_xy(point.get(k)) for k in
                ("start_pixel_xy", "end_pixel_xy", "projected_end_pixel_xy", "camera_only_end_pixel_xy")) or
                not all(_finite(point.get(k)) and 0 <= point[k] <= 1. for k in
                    ("forward_backward_error_px", "projected_endpoint_error_px"))):
            return None
        if np.linalg.norm(np.subtract(point["end_pixel_xy"],point["projected_end_pixel_xy"])) > 1.:
            return None
        key = tuple(point["start_pixel_xy"])
        if key in seen:
            return None
        seen.add(key)
        actual.append(np.subtract(point["end_pixel_xy"], point["camera_only_end_pixel_xy"]))
        projected.append(np.subtract(point["projected_end_pixel_xy"], point["camera_only_end_pixel_xy"]))
    a, p = np.median(actual, axis=0), np.median(projected, axis=0)
    if np.linalg.norm(a-p) > OBS.maximum_motion_fit_error_px:
        return None
    return dict(count=len(tracks), actual=a, projected=p, tracks=tracks)


def _support(evidence, view, key):
    binding = evidence.get("private_typed_support_mapping", {}).get(key, {})
    desc = evidence.get("views", {}).get(view, {}).get("support_instances", {}).get(key, {})
    body = binding.get("body_id")
    valid = (type(body) is int and body > 0 and key == "body_" + str(body) and
        binding.get("support_instance_key") == key and binding.get("identity_ambiguous") is False and
        desc.get("support_instance_key") == key and desc.get("body_id") == body and
        desc.get("identity_ambiguous") is False)
    return desc if valid else None


def _current_identity(frame, view, role, key):
    evidence = frame.get("observation_evidence", {})
    raw = evidence.get("views", {}).get(view, {})
    desc = raw.get("roles", {}).get(role, {}) if role == "target" else _support(evidence, view, key)
    if not isinstance(desc, Mapping):
        return False
    local = desc.get("local_rgb", {})
    if (raw.get("fresh_geometry_registered_to_actual_input") is not True or
            _registration(raw.get("global_rgb"), OBS) is not True or
            type(desc.get("rendered_pixels")) is not int or desc["rendered_pixels"] < OBS.minimum_role_pixels or
            local.get("passed") is not True or _registration(local, OBS) is not True or
            local.get("texture_sufficient") is not True or
            not _finite(local.get("actual_texture_range")) or local["actual_texture_range"] <= 0):
        return False
    for interval in evidence.get("actual_rgb_temporal_intervals", []):
        start, end = interval.get("start_step"), interval.get("end_step")
        if type(start) is not int or not 1 <= start < frame["step"] or end != frame["step"] or interval.get("view") != view:
            continue
        report = (interval.get("tracks", {}).get("target") if role == "target" else
            interval.get("typed_support_tracks", {}).get(key, {}).get("track"))
        if actual_track_summary(report) is not None:
            return True
    return False


def _prefix(frames, start, end, key):
    """No future backfill, no missing noise -> zero, no aggregate world role."""
    noises = {}
    for view in ("primary", "wrist"):
        own = []
        for step in range(start, end+1):
            frame = frames[step]; evidence = frame.get("observation_evidence", {})
            proprio = frame.get("proprio_max_abs")
            if not _finite(proprio) or not 0 <= proprio <= OBS.proprio_max_abs:
                return None, "actual_frame_proprio_alignment_unverified"
            qc = evidence.get("same_candidate_repeat_qc", {})
            if (qc.get("physics_passed") is not True or qc.get("all_contact_atoms_match") is not True or
                    not _finite(qc.get("qpos_qvel_max_abs")) or not 0 <= qc["qpos_qvel_max_abs"] <= 1e-3 or
                    not _finite(qc.get("qpos_qvel_p95")) or not 0 <= qc["qpos_qvel_p95"] <= 1e-4):
                return None, "same_candidate_repeat_QC_unverified"
            for role in ("target", "support"):
                if not _current_identity(frame, view, role, key):
                    return None, "both_views_typed_causal_actual_RGB_identity_required"
                raw = evidence["views"][view]
                repeat = (raw.get("repeat_noise", {}).get("target", {}) if role == "target" else
                    _support(evidence, view, key).get("repeat_noise", {}))
                noise, rgb = repeat.get("centroid_noise_px"), repeat.get("rgb_repeat", {})
                if (not _finite(noise) or noise < 0 or rgb.get("passed") is not True or
                        _registration(rgb, OBS) is not True):
                    return None, "own_surface_same_candidate_repeat_noise_missing"
                own.append(noise)
        noises[view] = max(own)
    return noises, None


def _camera(raw):
    if not isinstance(raw, Mapping):
        raise ValueError("Saved actual-image camera calibration required")
    position = np.asarray(raw.get("camera_position_m"), dtype=float)
    rotation = np.asarray(raw.get("camera_rotation_world_from_camera"), dtype=float)
    shape, convention, fovy = raw.get("image_shape"), raw.get("image_convention"), raw.get("fovy_degrees")
    if (position.shape != (3,) or rotation.shape != (3,3) or not np.isfinite(position).all() or
            not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6) or
            not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-6) or
            not isinstance(shape, (list,tuple)) or len(shape) != 2 or
            any(type(x) is not int or x <= 0 for x in shape) or convention not in (-1,1) or
            not _finite(fovy) or not 0 < fovy < 180):
        raise ValueError("Finite rigid GL camera calibration and saved-image convention required")
    h,w = shape; focal = h/(2*math.tan(math.radians(fovy)/2))
    return dict(position=position, rotation=rotation, h=h, w=w, focal=focal, convention=convention)


def _ray(camera, xy):
    x,y = xy
    if not (0 <= x < camera["w"] and 0 <= y < camera["h"]):
        return None
    if camera["convention"] == -1:
        y = camera["h"]-1-y
    cv = np.array([(x-camera["w"]/2)/camera["focal"], (y-camera["h"]/2)/camera["focal"], 1.])
    ray = camera["rotation"] @ (cv*np.array([1.,-1.,-1.]))
    return ray/np.linalg.norm(ray)


def _project(camera, point):
    cv = (camera["rotation"].T @ (point-camera["position"]))*np.array([1.,-1.,-1.])
    if not np.isfinite(cv).all() or cv[2] <= 0:
        return None
    xy = camera["focal"]*cv[:2]/cv[2]+np.array([camera["w"]/2,camera["h"]/2])
    if camera["convention"] == -1:
        xy[1] = camera["h"]-1-xy[1]
    return xy if 0 <= xy[0] < camera["w"] and 0 <= xy[1] < camera["h"] else None


def _triangulate(cameras, pixels):
    rays = [_ray(c,p) for c,p in zip(cameras,pixels)]
    if any(r is None for r in rays):
        return None
    matrix = np.column_stack((rays[0],-rays[1]))
    if np.linalg.matrix_rank(matrix) < 2:
        return None
    distance = np.linalg.lstsq(matrix,cameras[1]["position"]-cameras[0]["position"],rcond=None)[0]
    if min(distance) <= 0:
        return None
    point = np.mean([c["position"]+d*r for c,d,r in zip(cameras,distance,rays)],axis=0)
    projected = [_project(c,point) for c in cameras]
    if any(p is None for p in projected) or max(np.linalg.norm(p-x) for p,x in zip(projected,pixels)) > OBS.maximum_motion_fit_error_px:
        return None
    return point


def calibrated_world_upward_witness(correspondence, interval_by_view, camera_by_view, noises):
    """Matched observed landmarks identify world z; no simulator point input.

    Caller supplies independently audited SAME actual-RGB surface landmarks,
    not same-body simulator points or separately detected view corners. Every
    supplied endpoint must bind an existing own-surface actual LK track.
    """
    if (not isinstance(correspondence, Mapping) or
            correspondence.get("basis") != CORRESPONDENCE_BASIS or
            correspondence.get("actual_RGB_correspondence_audited") is not True):
        return None, "missing_actual_RGB_cross_view_correspondence"
    matches = correspondence.get("matches")
    if not isinstance(matches,list) or len(matches) < 3:
        return None, "three_independent_cross_view_landmarks_required"
    cameras = {v:{t:_camera(camera_by_view[v][t]) for t in ("start","end")} for v in ("primary","wrist")}
    summaries = {v:{r:actual_track_summary(interval_by_view[v][r]) for r in ("target","support")} for v in cameras}
    if any(s is None for roles in summaries.values() for s in roles.values()):
        return None, "own_surface_actual_projection_tracks_unverified"
    used = {v:set() for v in cameras}; rises=[]; signed={v:[] for v in cameras}
    for match in matches:
        if not isinstance(match,Mapping) or match.get("actual_RGB_same_landmark_verified") is not True:
            return None, "same_landmark_actual_RGB_audit_missing"
        points={}
        for view in cameras:
            index=match.get(view + "_target_track_index")
            if type(index) is not int or index in used[view] or not 0 <= index < summaries[view]["target"]["count"]:
                return None, "duplicate_or_unbound_actual_RGB_landmark"
            used[view].add(index);points[view]=summaries[view]["target"]["tracks"][index]
        before = _triangulate([cameras[v]["start"] for v in cameras],[points[v]["start_pixel_xy"] for v in cameras])
        after = _triangulate([cameras[v]["end"] for v in cameras],[points[v]["end_pixel_xy"] for v in cameras])
        if before is None or after is None:
            return None, "actual_multiview_triangulation_unidentifiable"
        rises.append(float(after[2]-before[2]))
        for view in cameras:
            camera=cameras[view]["end"];baseline=_project(camera,before)
            shifted=_project(camera,before+np.array([0.,0.,PHYS.vertical_rise_m]))
            if baseline is None or shifted is None or np.linalg.norm(shifted-baseline) <= 0:
                return None, "calibrated_world_vertical_axis_not_identifiable"
            axis=(shifted-baseline)/np.linalg.norm(shifted-baseline)
            actual=np.subtract(points[view]["end_pixel_xy"],points[view]["camera_only_end_pixel_xy"])
            relative=actual-summaries[view]["support"]["actual"]
            signed[view].append(float(relative @ axis))
    # All audited landmarks, not an advantageous median subset, must rise.
    if min(rises) < PHYS.vertical_rise_m:
        return None, "actual_triangulated_world_vertical_rise_below_frozen_1cm"
    for view,components in signed.items():
        if min(components) < OBS.minimum_motion_px or min(components) <= OBS.repeat_noise_multiplier*noises[view]:
            return None, "signed_world_vertical_RGB_motion_below_own_repeat_floor"
    return dict(actual_landmark_rises_m=rises,minimum_actual_rise_m=min(rises),
        signed_vertical_relative_motion_px=signed,world_vertical_identified=True,
        direction_basis="actual_RGB_landmark_multiview_triangulation_plus_saved_camera_calibration",
        simulator_world_points_used_as_visual_correspondence=False),None


def certify_typed_upward_motion(frames, visual_evidence, physical_label, *, carried_event_cert=None):
    """A new auxiliary target; never unmask target_support_contact or lifted.

    ``frames`` contain step/observation_evidence ONLY, no physics. ``physical_label``
    contains immutable Y and hashes. Missing visual direction means unknown,
    never no-rise/negative-lift. Composite carrying is a separate output target.
    """
    _forbidden(visual_evidence); _forbidden(physical_label)
    identity=_identity(visual_evidence.get("identity")); label_id=_identity(physical_label.get("identity"))
    if len(frames) != 17 or [f.get("step") for f in frames] != list(range(17)):
        raise ValueError("All 17 source-aligned observation frames required")
    if any("physics" in f for f in frames):
        raise ValueError("Simulator physics belongs in separate Y, never visual frames")
    if any(_identity(f.get("identity")) != identity or
            f.get("observation_evidence",{}).get("step") != f["step"] or
            f.get("observation_evidence",{}).get("role") != "offline_observation_supervision_only" for f in frames):
        raise ValueError("Every actual frame must bind the same full candidate and time")
    start,end=visual_evidence.get("start_step"),visual_evidence.get("end_step");key=visual_evidence.get("support_instance_key")
    base=dict(identity=dict(visual_evidence["identity"]),name="typed_upward_motion",start_step=start,end_step=end,
        support_instance_key=key,carried_and_raised=dict(value=None,mask=False),
        no_after_execution_observation_in_preexecution_X=True)
    if identity != label_id or physical_label.get("step") != end or physical_label.get("reference_step") != start:
        return _result(None,"physical_Y_reference_or_candidate_mismatch",**base)
    if type(start) is not int or type(end) is not int or not 2 <= start < end <= 16:
        return _result(None,"reference_must_be_real_causally_certified_step2_or_later",**base)
    used=visual_evidence.get("used_steps",list(range(start,end+1)))
    if used != list(range(start,end+1)):
        return _result(None,"future_or_incomplete_visual_prefix",**base)
    if not isinstance(key,str) or not key.startswith("body_"):
        return _result(None,"specific_nonworld_typed_support_required",**base)
    if (not _hashes(visual_evidence) or not _hashes(physical_label) or
            "physical_measurement" not in physical_label["evidence_sha256"] or
            visual_evidence["evidence_sha256"]["actual_rgb"] != physical_label["evidence_sha256"]["actual_rgb"]):
        return _result(None,"verified_source_Y_and_actual_RGB_hashes_required",**base)
    rise=physical_label.get("vertical_rise_from_reference_m")
    if (physical_label.get("source") != "offline_simulator_measurement" or
            physical_label.get("measurement_valid") is not True or not _finite(rise) or rise < PHYS.vertical_rise_m):
        return _result(None,"frozen_physical_positive_rise_Y_required_not_negative",**base)
    noises,reason=_prefix(frames,start,end,key)
    if reason:
        return _result(None,reason,**base)
    intervals={}; cameras={}
    for view in ("primary","wrist"):
        matches=[a for a in frames[end]["observation_evidence"].get("actual_rgb_temporal_intervals",[]) if
            a.get("start_step")==start and a.get("end_step")==end and a.get("view")==view]
        if len(matches) != 1:
            return _result(None,"unique_same_window_actual_RGB_interval_required",**base)
        a=matches[0];support=a.get("typed_support_tracks",{}).get(key,{})
        if (support.get("identity_ambiguous") is not False or support.get("support_instance_key") != key or
                support.get("body_id") != _support(frames[end]["observation_evidence"],view,key).get("body_id")):
            return _result(None,"typed_support_interval_binding_unverified",**base)
        interval_noise=(a.get("same_candidate_repeat_noise",{}).get("target"),support.get("same_candidate_repeat_noise_px"))
        if not all(_finite(n) and n >= 0 for n in interval_noise):
            return _result(None,"own_surface_interval_repeat_noise_missing",**base)
        noises[view]=max(noises[view],*interval_noise)
        intervals[view]=dict(target=a.get("tracks",{}).get("target"),support=support.get("track"))
        cameras[view]={t:frames[s]["observation_evidence"]["views"][view].get("camera") for t,s in (("start",start),("end",end))}
    witness,reason=calibrated_world_upward_witness(visual_evidence.get("world_vertical_direction_evidence"),intervals,cameras,noises)
    if reason:
        return _result(None,reason,**base)
    carry=carried_event_cert or {}
    carry_ok=(carry.get("schema")==SCHEMA_V2 and carry.get("name")=="carried_sufficient_evidence" and
        carry.get("new_mask") is True and carry.get("physical_value") is True and carry.get("measurement_valid") is True and
        carry.get("step")==end and carry.get("role")=="offline_supervision_only" and carry.get("deployment_allowed") is False and
        isinstance(carry.get("identity"),Mapping) and _identity(carry["identity"])==identity)
    base["carried_and_raised"]=dict(value=True if carry_ok else None,mask=carry_ok)
    return _result(True,**base,visual_witness=witness,repeat_noise_px=noises,
        physical_Y=dict(vertical_rise_from_reference_m=rise,source="offline_simulator_measurement"),
        supervision_basis="typed_causal_actual_RGB_world_vertical_motion_plus_distinct_offline_rise_Y")


def derive_typed_upward_motion_sidecar(teacher, replay, evidence_sha256, *, source_hashes_verified,
                                     cross_view_landmarks=None, carried_certificates=None):
    """Pure new sidecar from frozen observations; no replay or original writes.

    Landmark evidence is keyed by ``(start_step,end_step,support_instance_key)``.
    Missing landmarks remain missing; private simulator point tracks are NOT
    synthesized into actual cross-view correspondences. Physical rise is
    measured from the same real reference, not silently inherited from frame0.
    """
    identity=teacher.get("identity");_identity(identity)
    raw=teacher.get("frames");checks=replay.get("checks")
    if (not isinstance(raw,list) or len(raw)!=17 or [f.get("step") for f in raw]!=list(range(17)) or
            not isinstance(checks,list) or len(checks)!=17 or [f.get("step") for f in checks]!=list(range(17))):
        raise ValueError("Frozen teacher/replay 17-frame index alignment required")
    frames=[dict(step=f["step"],identity=dict(identity),proprio_max_abs=checks[f["step"]].get("proprio_max_abs"),
        observation_evidence=f.get("observation_evidence",{})) for f in raw]
    landmark_table=cross_view_landmarks or {};carry_table=carried_certificates or {};labels=[]
    for end in range(17):
        possible={}
        for interval in frames[end]["observation_evidence"].get("actual_rgb_temporal_intervals",[]):
            start,view=interval.get("start_step"),interval.get("view")
            if type(start) is not int or not 2<=start<end or interval.get("end_step")!=end or view not in ("primary","wrist"):
                continue
            for key in interval.get("typed_support_tracks",{}):
                possible.setdefault((start,end,key),set()).add(view)
        attempts=[]
        for window,views in sorted(possible.items()):
            if views != {"primary","wrist"}:
                continue
            start,_,key=window
            before=raw[start].get("physics",{}).get("target_pos_m")
            after=raw[end].get("physics",{}).get("target_pos_m")
            positions=all(isinstance(x,(list,tuple)) and len(x)==3 and all(_finite(v) for v in x) for x in (before,after))
            rise=after[2]-before[2] if positions else None
            visual=dict(identity=dict(identity),start_step=start,end_step=end,support_instance_key=key,
                used_steps=list(range(start,end+1)),source_hashes_verified=source_hashes_verified,
                evidence_sha256=dict(evidence_sha256),world_vertical_direction_evidence=landmark_table.get(window))
            physical=dict(identity=dict(identity),reference_step=start,step=end,
                source="offline_simulator_measurement",measurement_valid=positions,
                vertical_rise_from_reference_m=rise,source_hashes_verified=source_hashes_verified,
                evidence_sha256=dict(evidence_sha256))
            attempts.append(certify_typed_upward_motion(frames,visual,physical,carried_event_cert=carry_table.get(end)))
        chosen=next((a for a in attempts if a["mask"]),attempts[0] if attempts else
            _result(None,"no_aligned_two_view_typed_support_interval",identity=dict(identity),name="typed_upward_motion",end_step=end,
                carried_and_raised=dict(value=None,mask=False)))
        labels.append(dict(step=end,certificate=chosen,all_attempt_reasons=sorted({r for a in attempts for r in a["reasons"]}),
            attempted_windows=len(attempts)))
    return dict(schema=SCHEMA+"_sidecar",identity=dict(identity),labels=labels,
        role="distinct_auxiliary_supervision_only",original_masks_and_values_unchanged=True,
        literal_target_support_and_lifted_masks_untouched=True,training_ready=False,
        deployment_features_created=False,no_after_execution_observation_in_preexecution_X=True,
        actual_RGB_cross_view_correspondences_synthesized_from_simulator=False,
        evidence_sha256=dict(evidence_sha256))

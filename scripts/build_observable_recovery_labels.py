"""Build new offline observation certificates from immutable verified replay.

No model fitting, policy queries, deployment features, label overwrites, or
terminal outcomes. Geometry assists offline point association; actual RGB LK
tracks, local interfaces, and SAME-candidate repeat noise supply observation
proxies. These algorithmic silver certificates are not human annotation or a
claim that the deployed model already recognizes the events.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import importlib.util
import types
import copy

import numpy as np

SCHEMA = "observable_recovery_labels_sidecars_v1"
SCHEMA_V2 = "observable_recovery_labels_sidecars_v2"
SOURCE_DATASET = "known_task_recovery_v9_20261006_v5_paired_recovery"
OBSERVATION_SOURCE_PATTERN = r"known_task_recovery_observation_audit_20261006_v(?:[34]_(?:depth_interface|view_masks)|5_motion_typed_support)_(?:pilot|full)"
CONTEXTS = ("normal", "visual_occlusion", "object_shift", "execution_contact_deviation", "unknown")
RAW_FILES = ("primary.npy", "wrist.npy", "proprio.npy", "applied.npy", "requested.npy", "planned_actions.npy", "exact_start_state.npy")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def contained(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Evidence reference must be relative")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Evidence reference escapes its source root")
    return path


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def vector(value):
    return isinstance(value, list) and len(value) == 2 and all(finite(x) for x in value)


def rgb_pass(metrics):
    return (isinstance(metrics, dict) and all(finite(metrics.get(k)) for k in ("mean_abs", "p95", "fraction_gt5", "psnr")) and
        0 <= metrics["mean_abs"] <= 2 and 0 <= metrics["p95"] <= 8 and
        0 <= metrics["fraction_gt5"] <= .06 and metrics["psnr"] >= 30)


def track_summary(report):
    """Do not substitute segmentation centroids or projection for actual flow."""
    if (not isinstance(report, dict) or report.get("verified") is not True or
        report.get("geometry_motion_support") is not True or
        report.get("evidence_kind") != "actual_RGB_geometry_correspondence"):
        return None
    tracks = report.get("tracks")
    if not isinstance(tracks, list) or len(tracks) < 3 or report.get("count") != len(tracks):
        return None
    starts = set()
    actual, projected = [], []
    for point in tracks:
        if (not isinstance(point, dict) or not all(vector(point.get(k)) for k in
            ("start_pixel_xy", "end_pixel_xy", "projected_end_pixel_xy", "camera_only_end_pixel_xy")) or
            not finite(point.get("forward_backward_error_px")) or not 0 <= point["forward_backward_error_px"] <= 1 or
            not finite(point.get("projected_endpoint_error_px")) or not 0 <= point["projected_endpoint_error_px"] <= 1):
            return None
        key = tuple(point["start_pixel_xy"])
        if key in starts:
            return None
        starts.add(key)
        camera = point["camera_only_end_pixel_xy"]
        actual.append([point["end_pixel_xy"][i] - camera[i] for i in (0, 1)])
        projected.append([point["projected_end_pixel_xy"][i] - camera[i] for i in (0, 1)])
    return dict(count=len(tracks), actual=[statistics.median(x[i] for x in actual) for i in (0, 1)],
        projected=[statistics.median(x[i] for x in projected) for i in (0, 1)],
        identity_basis="audited_actual_rgb_instance_correspondence",
        correspondence_is_algorithmic_silver_not_human_truth=True)


def current_role(track, descriptor):
    local = descriptor.get("local_rgb", {}) if isinstance(descriptor, dict) else {}
    structure = local.get("texture_sufficient") is True and finite(local.get("actual_texture_range")) and local["actual_texture_range"] > 0
    enough = track is not None and structure and local.get("passed") is True and rgb_pass(local)
    return dict(rendered_pixels=descriptor.get("rendered_pixels") if isinstance(descriptor, dict) else None,
        local_rgb=local, identity_rgb_unique=enough,
        identity_basis="audited_actual_rgb_instance_correspondence" if enough else None,
        identity_quality=1. if enough else None, actual_rgb_structure_nonconstant=structure,
        identity_source="causal_actual_RGB_unique_surface_tracks" if enough else "unverified",
        identity_is_algorithmic_annotation_proxy=True)


def interface_input(saved, raw_physics, roles, *, support_instance_key=None):
    """A literal positive interface only; no synthetic no-contact gap witness."""
    contact = saved.get("physics_contact", {})
    report = saved.get("interface_evidence", {})
    side = contact.get("role")
    name = {"left": "left_finger_target_contact", "right": "right_finger_target_contact", "support": "target_support_contact"}.get(side)
    if name is None or raw_physics.get(name) is not True:
        return None
    other = "support" if side == "support" else "eef"
    # Body0/world is not one unique rendered support. This key must match a
    # causally tracked concrete counterpart, never a union of scene surfaces.
    if ((side == "support" and (not support_instance_key or support_instance_key == "body_0" or
        contact.get("support_instance_key") != support_instance_key)) or
        contact.get("counterpart_body_id") is None or not contact.get("counterpart_body_name")):
        return None
    required = (report.get("passed") is True and report.get("supports_interface_observability") is True and
        report.get("full_patch_in_bounds") is True and report.get("role_reference_floor_passed") is True and
        all(report.get("role_depth_matched_patch_pixels", {}).get(k, 0) > 0 for k in ("target", "finger")) and
        all(report.get(k, {}).get("passed") is True and rgb_pass(report.get(k)) for k in
            ("shared_patch_registration", "target_registration", "finger_registration")) and
        roles["target"].get("identity_rgb_unique") is True and roles.get(other, {}).get("identity_rgb_unique") is True)
    tolerance = report.get("surface_depth_tolerance_m")
    if not required or not finite(tolerance) or not 0 < tolerance <= .002:
        return None
    boundary = report.get("actual_rgb_boundary_pixels")
    if (report.get("actual_rgb_boundary_verified") is not True or
        type(boundary) is not int or boundary < 2):
        return None
    return dict(physics_contact=True, other_surface_role=other, projection_in_frame=report["full_patch_in_bounds"],
        depth_error_m=tolerance, depth_error_is_passed_frozen_upper_bound=True,
        rgb_boundary_pixels=boundary,
        local_rgb=report["shared_patch_registration"], surfaces_rgb_visible=required,
        counterpart_body_id=contact["counterpart_body_id"], counterpart_body_name=contact["counterpart_body_name"],
        support_instance_key=support_instance_key if side == "support" else None)


def actual_interface_boundaries(teacher, arrays_path, original_folder, *, separation_function=None):
    """Measure actual-RGB boundary witnesses; never rename mask area evidence.

    Segmentation only selects private annotation roles. Boundary pixels require
    four-neighbour target/finger adjacency in the saved contact patch and actual
    RGB contrast above the frozen 2-unit registration floor / 3x measured repeat
    residual. Preserve surrounding gradients for review, not a grasp verdict.
    """
    result = copy.deepcopy(teacher)
    with np.load(arrays_path, allow_pickle=False) as saved:
        segments = {view: saved[view] for view in ("primary", "wrist")}
    actual = {view: np.load(Path(original_folder) / (view + ".npy"), allow_pickle=False)
        for view in ("primary", "wrist")}
    if any(segments[v].shape != (17, 256, 256) or actual[v].shape != (17, 256, 256, 3) for v in segments):
        raise ValueError("Original RGB / segmentation temporal shapes differ")
    for frame in result["frames"]:
        step = frame["step"]; evidence = frame["observation_evidence"]
        mapping = evidence["private_collision_vs_visual_mapping"]
        for view, record in evidence["views"].items():
            geom = segments[view][step]; rgb = actual[view][step].astype(float)
            target = np.isin(geom, [g["geom_id"] for g in mapping["target"]])
            for witness in record["contact_interfaces"]:
                report = witness["interface_evidence"]; side = witness["physics_contact"]["role"]
                report.update(actual_rgb_boundary_verified=False, actual_rgb_boundary_pixels=None,
                    boundary_role="offline_actual_RGB_interface_witness_not_contact_truth")
                if step == 0 or side not in ("left", "right", "support"):
                    continue
                support_key = witness["physics_contact"].get("support_instance_key")
                typed = evidence.get("private_typed_support_mapping", {}).get(support_key, {})
                if side == "support" and (typed.get("identity_ambiguous") is not False or
                        typed.get("body_id") in (None, 0) or typed.get("support_instance_key") != support_key):
                    continue
                xy = report.get("contact_pixel_xy"); radius = report.get("patch_radius_px")
                other_noise = (record.get("support_instances", {}).get(support_key, {}).get("repeat_noise", {})
                    if side == "support" else record.get("repeat_noise", {}).get(side, {}))
                noises = [record.get("repeat_noise", {}).get("target", {}).get("rgb_repeat", {}).get("mean_abs"),
                    other_noise.get("rgb_repeat", {}).get("mean_abs")]
                if not vector(xy) or not finite(radius) or not all(finite(n) and n >= 0 for n in noises):
                    continue
                other_geoms = typed.get("visual_geoms", []) if side == "support" else mapping[side]
                finger = np.isin(geom, [g["geom_id"] for g in other_geoms])
                yy, xx = np.ogrid[:256, :256]
                patch = (xx-xy[0])**2 + (yy-xy[1])**2 <= radius**2
                floor = max(2., 3. * max(noises))
                boundary_pixels = set(); contrasts = []; surrounding = []
                for dy, dx in ((0, 1), (1, 0)):
                    sa = (slice(0, 256-dy), slice(0, 256-dx))
                    sb = (slice(dy, 256), slice(dx, 256))
                    cross = ((target[sa] & finger[sb]) | (finger[sa] & target[sb])) & patch[sa] & patch[sb]
                    contrast = np.max(np.abs(rgb[sa] - rgb[sb]), axis=-1)
                    internal = ((target[sa] & target[sb]) | (finger[sa] & finger[sb])) & patch[sa] & patch[sb]
                    surrounding.extend(contrast[internal].tolist()); contrasts.extend(contrast[cross].tolist())
                    ys, xs = np.nonzero(cross & (contrast > floor))
                    for y, x in zip(ys, xs):
                        boundary_pixels.add((int(y), int(x)))
                        boundary_pixels.add((int(y+dy), int(x+dx)))
                report.update(actual_rgb_boundary_pixels=len(boundary_pixels),
                    actual_rgb_boundary_verified=bool(record.get("fresh_geometry_registered_to_actual_input") is True
                        and report.get("passed") is True and len(boundary_pixels) >= 2),
                    actual_rgb_cross_role_contrasts=contrasts,
                    actual_rgb_internal_gradient_p95=float(np.quantile(surrounding, .95)) if surrounding else None,
                    actual_rgb_boundary_contrast_floor=floor, same_candidate_repeat_rgb_mean_abs=noises,
                    boundary_coordinate_pixels=sorted([list(p) for p in boundary_pixels]),
                    boundary_measurement="four-connected actual RGB contrast within recorded contact projection patch")
            # Negative interfaces are independent annotations on a COPY, using
            # actual RGB and this same side's causal tracks. No EEF identity,
            # command, missing event or terminal outcome supplies a zero label.
            record["negative_interfaces"] = {}
            qc = evidence.get("same_candidate_repeat_qc", {})
            qc_ok = (qc.get("physics_passed") is True and qc.get("all_contact_atoms_match") is True and
                finite(qc.get("qpos_qvel_max_abs")) and 0 <= qc["qpos_qvel_max_abs"] <= 1e-3 and
                finite(qc.get("qpos_qvel_p95")) and 0 <= qc["qpos_qvel_p95"] <= 1e-4)
            if separation_function is None or step == 0 or not qc_ok:
                continue
            for side in ("left", "right"):
                if frame["physics"].get(side + "_finger_target_contact") is not False:
                    continue
                summaries = {role:None for role in ("target",side)}
                for interval in evidence.get("actual_rgb_temporal_intervals", []):
                    if (interval.get("view") != view or interval.get("end_step") != step or
                            type(interval.get("start_step")) is not int or not 1 <= interval["start_step"] < step):
                        continue
                    for role in summaries:
                        current = track_summary(interval.get("tracks", {}).get(role))
                        if current is not None and (summaries[role] is None or current["count"] > summaries[role]["count"]):
                            summaries[role] = current
                roles = {role:current_role(summaries[role],record.get("roles",{}).get(role,{})) for role in summaries}
                repeated = {role:record.get("repeat_noise",{}).get(role,{}) for role in summaries}
                if not all(v.get("rgb_repeat",{}).get("passed") is True and rgb_pass(v.get("rgb_repeat")) for v in repeated.values()):
                    continue
                witness = separation_function(actual_rgb=actual[view][step],target_mask=target,
                    finger_mask=np.isin(geom,[g["geom_id"] for g in mapping[side]]),
                    target_registration=record.get("roles",{}).get("target",{}).get("local_rgb",{}),
                    finger_registration=record.get("roles",{}).get(side,{}).get("local_rgb",{}),
                    target_identity_certified=roles["target"].get("identity_rgb_unique") is True,
                    finger_identity_certified=roles[side].get("identity_rgb_unique") is True,
                    target_repeat_noise_px=repeated["target"].get("centroid_noise_px"),
                    finger_repeat_noise_px=repeated[side].get("centroid_noise_px"),
                    target_repeat_rgb_mean_abs=repeated["target"].get("rgb_repeat",{}).get("mean_abs"),
                    finger_repeat_rgb_mean_abs=repeated[side].get("rgb_repeat",{}).get("mean_abs"),
                    fresh_geometry_registered=record.get("fresh_geometry_registered_to_actual_input"),
                    physical_contact=frame["physics"].get(side+"_finger_target_contact"),side=side)
                if witness is not None:
                    record["negative_interfaces"][side] = witness
    return result


def annotation_modules(root):
    """Private flat CODE namespace when supplied; never edit installed packages."""
    directory = Path(__file__).resolve().parent
    if all((directory / name).is_file() for name in ("recovery_observability_contract.py", "temporal_recovery_teacher.py")):
        name = "_source_scoped_observation_" + hashlib.sha256(str(directory).encode("utf-8")).hexdigest()[:12]
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
        loaded = []
        for module in ("temporal_recovery_teacher", "recovery_observability_contract"):
            spec = importlib.util.spec_from_file_location(name + "." + module, directory / (module + ".py"))
            result = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = result
            spec.loader.exec_module(result)
            loaded.append(result)
        return loaded[1], loaded[0].derive_temporal_recovery
    sys.path.insert(0, str(root))
    from wam_reranking import recovery_observability_contract as contract
    from wam_reranking.temporal_recovery_teacher import derive_temporal_recovery
    return contract, derive_temporal_recovery


def separation_extractor(root):
    """Load explicitly hash-reported offline CPU helper, including flat CODE."""
    directory = Path(__file__).resolve().parent
    path = directory / "observable_separation.py"
    if not path.is_file():
        path = Path(root) / "wam_reranking" / "observable_separation.py"
    if not path.is_file():
        raise ValueError("Offline separation extractor source is required")
    name = "_source_scoped_separation_" + sha(path)[:12]
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.negative_contact_witness, path


def actual_proprio_evidence(teacher, replay, original_proprio, evidence_hashes):
    """Original hashed per-frame qpos/EEF only; no command-derived closure.

    Original shape is qpos(2), EEF position(3), EEF quaternion(4). Frame0 is a
    cached diagnostic and deliberately excluded from new label references.
    The aperture sum has the inherited 2*per-coordinate error bound.
    """
    values = np.asarray(original_proprio)
    if values.shape != (17, 9) or not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("Original finite 17x9 proprio is required for V2 lineage")
    digest = evidence_hashes.get("actual_proprio")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError("Original proprio file SHA256 is mandatory")
    rows = []
    for step, raw in enumerate(teacher["frames"]):
        p, check = raw["physics"], replay["checks"][step]
        measured_eef = np.asarray(p.get("eef_pos_m"))
        aperture = p.get("gripper_aperture_m")
        error = check.get("proprio_max_abs")
        if step > 0 and (measured_eef.shape != (3,) or not np.isfinite(measured_eef).all() or
            not finite(aperture) or aperture < 0 or not finite(error) or not 0 <= error <= 1e-3 or
            float(np.max(np.abs(values[step, 2:5] - measured_eef))) > 1e-3 or
            abs(float(np.abs(values[step, :2]).sum()) - aperture) > 2e-3):
            raise ValueError("Hashed original proprio differs from replay physics at step " + str(step))
        rows.append(dict(source="original_hashed_proprio_npy", sha256=digest, frame=step,
            gripper_qpos_m=values[step, :2].astype(float).tolist(), eef_pos_m=values[step, 2:5].astype(float).tolist(),
            replay_proprio_max_abs=error, cached_diagnostic_only=step == 0,
            measured_position_is_not_command_or_grasp_label=True))
    return rows


def candidate_certificates(teacher, measurements, replay, evidence_hashes, contract, *,
                           certificate_version="v1", original_proprio=None):
    identity = teacher["identity"]
    raw_frames = teacher["frames"]
    checks = replay["checks"]
    if len(raw_frames) != 17 or len(measurements["frames"]) != 17 or len(checks) != 17:
        raise ValueError("All original 17 actual frames and alignment checks are mandatory")
    if certificate_version not in ("v1", "v2"):
        raise ValueError("Unknown observation certificate version")
    v2 = certificate_version == "v2"
    proprio_rows = actual_proprio_evidence(teacher, replay, original_proprio, evidence_hashes) if v2 else None
    motion_function = contract.certify_temporal_motion_v2 if v2 else contract.certify_temporal_motion
    label_function = contract.certify_supervision_v2 if v2 else contract.certify_supervision
    frame_function = contract.certify_frame_v2 if v2 else contract.certify_frame
    certificates, interval_rows = [], []
    for step, raw in enumerate(raw_frames):
        if raw.get("step") != step or checks[step].get("step") != step:
            raise ValueError("Frame/check index mismatch")
        evidence = raw.get("observation_evidence", {})
        if evidence.get("step") != step or evidence.get("role") != "offline_observation_supervision_only":
            raise ValueError("A pure physics replay is not an observation source")
        intervals = evidence.get("actual_rgb_temporal_intervals", [])
        views = {}
        for view in ("primary", "wrist"):
            source_view = evidence.get("views", {}).get(view, {})
            tracks = {role: None for role in ("target", "eef", "anchor", "support")}
            finger_tracks = {side: None for side in ("left", "right")}
            support_tracks = {}
            # Only causal intervals ending at CURRENT can establish its role
            # identity. Start frame1 remains unverified: do not backfill it.
            for interval in intervals:
                if (interval.get("view") != view or interval.get("end_step") != step or
                    type(interval.get("start_step")) is not int or not 1 <= interval["start_step"] < step):
                    continue
                for role in ("target", "eef", "anchor"):
                    summary = track_summary(interval.get("tracks", {}).get(role))
                    if summary is not None and (tracks[role] is None or summary["count"] > tracks[role]["count"]):
                        tracks[role] = summary
                if v2:
                    for side in ("left", "right"):
                        summary = track_summary(interval.get("tracks", {}).get(side))
                        if summary is not None and (finger_tracks[side] is None or summary["count"] > finger_tracks[side]["count"]):
                            finger_tracks[side] = summary
                    for key, entry in interval.get("typed_support_tracks", {}).items():
                        if (entry.get("identity_ambiguous") is not False or entry.get("body_id") in (None, 0) or
                            entry.get("support_instance_key") != key):
                            continue
                        summary = track_summary(entry.get("track"))
                        if summary is not None and (key not in support_tracks or summary["count"] > support_tracks[key]["count"]):
                            support_tracks[key] = summary
            roles = {role: current_role(tracks[role], source_view.get("roles", {}).get(role, {})) for role in tracks}
            finger_roles = {side:current_role(finger_tracks[side],source_view.get("roles",{}).get(side,{}))
                for side in ("left","right")}
            support_roles = {}
            for key, summary in support_tracks.items():
                descriptor = source_view.get("support_instances", {}).get(key, {})
                if descriptor.get("identity_ambiguous") is False and descriptor.get("support_instance_key") == key:
                    support_roles[key] = current_role(summary, descriptor.get("descriptor", descriptor))
            interfaces = {}
            for saved in source_view.get("contact_interfaces", []):
                key = saved.get("physics_contact", {}).get("support_instance_key")
                side = saved.get("physics_contact", {}).get("role")
                # Each support witness is validated against its own typed role.
                bound_roles = dict(roles, support=support_roles.get(key, roles["support"]))
                if v2 and side in ("left","right"):
                    bound_roles["eef"] = finger_roles[side]
                item = interface_input(saved, raw["physics"], bound_roles, support_instance_key=key if v2 else None)
                if item is not None:
                    interfaces[saved["physics_contact"]["role"]] = item
                    if item["other_surface_role"] == "support":
                        roles["support"] = bound_roles["support"]
            if v2:
                for side, item in source_view.get("negative_interfaces",{}).items():
                    measured_name = side + "_finger_target_contact"
                    if (side in ("left","right") and raw["physics"].get(measured_name) is False and
                        item.get("physics_contact") is False and item.get("own_finger_side") == side and
                        finger_roles[side].get("identity_rgb_unique") is True):
                        interfaces[side] = item
            # Fresh geometry must be registered to the ACTUAL input, not merely
            # to another cached render. Unreliable primary need not mask wrist.
            geometry_eligible = source_view.get("fresh_geometry_registered_to_actual_input") is True
            views[view] = dict(global_rgb=source_view.get("global_rgb", {}) if geometry_eligible else {},
                roles=roles, interfaces=interfaces,independent_finger_roles=finger_roles if v2 else {})
        frame_input = dict(identity=identity, step=step, used_steps=list(range(step + 1)), views=views,
            proprio_max_abs=checks[step].get("proprio_max_abs"), source_hashes_verified=True,
            evidence_sha256=evidence_hashes)
        certificates.append(frame_function(frame_input))
        for interval in intervals:
            start, end, view = interval.get("start_step"), interval.get("end_step"), interval.get("view")
            if type(start) is not int or start < 2 or end != step or view not in ("primary", "wrist"):
                continue
            target = track_summary(interval.get("tracks", {}).get("target"))
            for reference in ("eef", "anchor"):
                other = track_summary(interval.get("tracks", {}).get(reference))
                noise = interval.get("same_candidate_repeat_noise", {})
                noises = [noise.get("target"), noise.get(reference)]
                noise_ok = all(finite(x) and x >= 0 for x in noises)
                qc_ok = True
                for s in range(start, end + 1):
                    evidence_s = raw_frames[s].get("observation_evidence", {})
                    qc = evidence_s.get("same_candidate_repeat_qc", {})
                    qc_ok = qc_ok and (qc.get("physics_passed") is True and qc.get("all_contact_atoms_match") is True and
                        finite(qc.get("qpos_qvel_max_abs")) and 0 <= qc["qpos_qvel_max_abs"] <= 1e-3 and
                        finite(qc.get("qpos_qvel_p95")) and 0 <= qc["qpos_qvel_p95"] <= 1e-4)
                    for role in ("target", reference):
                        repeated = evidence_s.get("views", {}).get(view, {}).get("repeat_noise", {}).get(role, {})
                        repeated_rgb = repeated.get("rgb_repeat", {})
                        qc_ok = qc_ok and (repeated_rgb.get("passed") is True and rgb_pass(repeated_rgb) and
                            finite(repeated.get("centroid_noise_px")) and repeated["centroid_noise_px"] >= 0)
                motion_input = dict(identity=identity, same_candidate_repeat_identity=identity,
                    start_step=start, end_step=end, view=view, source_hashes_verified=True,
                    evidence_sha256=evidence_hashes, repeat_noise_verified=qc_ok and noise_ok,
                    repeat_noise_px=max(noises) if noise_ok else None,
                    camera_motion_compensated=target is not None and other is not None,
                    actual_rgb_tracking_verified=target is not None and other is not None,
                    actual_rgb_tracking_basis="actual_rgb_geometric_correspondence",
                    actual_rgb_tracking_points=min(target["count"], other["count"]) if target is not None and other is not None else None,
                    actual_subject_delta_px=target["actual"] if target is not None else None,
                    projected_subject_delta_px=target["projected"] if target is not None else None,
                    actual_reference_delta_px=other["actual"] if other is not None else None,
                    projected_reference_delta_px=other["projected"] if other is not None else None)
                motion_certificate = motion_function(certificates, motion_input, reference_role=reference)
                interval_rows.append(dict(input=motion_input, certificate=motion_certificate,
                    actual_motion_definition="median(actual_end - camera_only_end)",
                    projected_motion_definition="median(projected_end - camera_only_end)",
                    reference_window_kind=interval.get("interval_kind", "existing_causal_window")))
    labels, previous = [], {}
    for step, measured in enumerate(measurements["frames"]):
        if measured.get("step") != step:
            raise ValueError("Measurement frame indices are not aligned")
        per_atom = {}
        for name in contract.ATOM_NAMES:
            old = measured["physics"].get(name, dict(value=None, measurement_valid=False, supervision_mask=False,
                source="offline_simulator_measurement", unavailable_reason="not_applicable_to_entity_kind"))
            atom = dict(old, name=name, identity=identity, step=step, entity_kind=teacher["kind"],
                source_hashes_verified=True, evidence_sha256=evidence_hashes)
            attempts = []
            motions = [row["certificate"] for row in interval_rows if row["certificate"].get("end_step") == step]
            if name == "joint_state_changed":
                motions = [m for m in motions if m.get("reference_role") == "anchor"]
            elif name == "carried_sufficient_evidence":
                motions = [m for m in motions if m.get("reference_role") == "eef"]
            else:
                motions = [None]  # No support / negative-separation witnesses.
            for motion in motions or [None]:
                prepared = dict(atom)
                if name == "joint_state_changed" and motion is not None:
                    start = motion.get("start_step")
                    a, b = raw_frames[start]["physics"].get("target_joint_qpos"), raw_frames[step]["physics"].get("target_joint_qpos")
                    prepared.update(joint_unit=teacher.get("joint_unit"), physical_reference_step=start,
                        joint_delta_from_reference=b - a if finite(a) and finite(b) else None)
                if v2 and name == "carried_sufficient_evidence" and motion is not None:
                    start = motion.get("start_step")
                    if type(start) is int:
                        prepared["physical_carry_window"] = [dict(step=s, identity=identity,
                            physics=raw_frames[s]["physics"], actual_proprio=proprio_rows[s]) for s in range(start, step+1)]
                attempts.append(label_function(prepared, certificates, motion_cert=motion,
                    before_certificate=previous.get(name)))
            chosen = next((a for a in attempts if a["new_mask"]), attempts[0])
            chosen["all_attempt_reasons"] = sorted({r for a in attempts for r in a["reasons"]})
            if name in ("target_support_contact", "lifted_sufficient_evidence", "released_sufficient_evidence"):
                chosen["missing_additional_witness"] = "typed_support_instance_track_and_or_actual_separation_reference"
            per_atom[name] = chosen
        labels.append(dict(step=step, atoms=per_atom))
        previous = per_atom
    first = {name: next((row["step"] for row in labels if row["atoms"][name]["new_mask"] and
        row["atoms"][name]["physical_value"] is True), None) for name in contract.ATOM_NAMES}
    return dict(schema=SCHEMA_V2 if v2 else SCHEMA, inherits=[SCHEMA] if v2 else [],
        identity=identity, role="offline_supervision_only", frame_certificates=certificates,
        motion_certificates=interval_rows, labels=labels, first_certified_events=first,
        original_first_measured_events=measurements["first_measured_events"], old_masks_and_values_unchanged=True,
        cached_frame0_diagnostic_only=True, motion_reference_minimum_step=2,
        future_information_labels_only=True, deployment_features_created=False, training_started=False,
        training_ready=False, evidence_sha256=evidence_hashes,
        actual_proprio_lineage=proprio_rows if v2 else None,
        carrying_silver_not_literal_contact_truth=v2,
        absolute_aperture_closure_threshold_used=False)


def verify_manifest(source):
    table = {}
    for line in (source / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = contained(source, relative)
        if relative in table or not re.fullmatch("[0-9a-fA-F]{64}", expected) or sha(path).lower() != expected.lower():
            raise ValueError("Derived replay evidence manifest mismatch: " + relative)
        table[relative] = expected.lower()
    required = ("records.json", "source_sha256.json", "completion_audit.json", "protocol.json", "observation_adapter_manifest.json")
    if any(name not in table for name in required):
        raise ValueError("Completed observation source manifest is incomplete")
    return table


def check_code_hash(root, source, expected, basename):
    if not isinstance(expected, str) or not re.fullmatch("[0-9a-fA-F]{64}", expected):
        raise ValueError("Annotation code digest missing: " + basename)
    candidates = (root / "scripts" / basename, root / "wam_reranking" / basename,
        root / "research_runs" / source.name / basename,
        root / "research_runs" / source.name.rsplit("_", 1)[0] / basename,
        Path(__file__).resolve().parent / basename,
        root / "research_runs" / "known_task_temporal_recovery_supervision_20261006_v2_scoped_segmentation" / basename)
    match = next((p for p in candidates if p.is_file() and sha(p).lower() == expected.lower()), None)
    if match is None:
        raise ValueError("Annotation source code cannot be verified: " + basename)
    return str(match.relative_to(root)) if root in match.parents else str(match)


def build(root, source, output):
    root, source, output = Path(root).resolve(), Path(source).resolve(), Path(output).resolve()
    if (output.exists() or output == source or source in output.parents or output in source.parents or
        output == root or output in root.parents):
        raise ValueError("Refuse overwrite or source/root mutation")
    if root not in source.parents or not re.fullmatch(OBSERVATION_SOURCE_PATTERN, source.name):
        raise ValueError("Only the named 32/160 candidate observation source is allowed")
    if (source / "failure.json").exists():
        raise ValueError("Failed/incomplete source cannot certify learning labels")
    manifest = verify_manifest(source)
    completion, protocol, provenance = (load(source / name) for name in
        ("completion_audit.json", "protocol.json", "observation_adapter_manifest.json"))
    records = load(source / "records.json")
    expected = 32 if source.name.endswith("_pilot") else 160
    if (completion.get("passed") is not True or completion.get("completed_candidates") != expected or len(records) != expected or
        completion.get("source_unchanged") is not True or completion.get("training_started") is not False or
        protocol.get("source") != SOURCE_DATASET or protocol.get("role") != "offline_supervision_only" or
        provenance.get("schema") != "actual_rgb_recovery_observation_evidence_v1" or
        provenance.get("new_policy_queries") != 0 or provenance.get("new_candidate_actions") != 0 or
        provenance.get("old_physical_teacher_masks_unchanged") is not True or completion.get("all_steps_aligned") is not True):
        raise ValueError("Observation/source role or cardinality contract failed")
    code_paths = {name: check_code_hash(root, source, provenance.get(key), name) for name, key in
        (("collect_recovery_observation_evidence.py", "adapter_sha256"), ("observation_geometry.py", "geometry_sha256"),
         ("replay_temporal_recovery_evidence.py", "base_script_sha256"))}
    original_hashes = load(source / "source_sha256.json")
    for relative, expected_sha in original_hashes.items():
        if sha(contained(root, relative)) != expected_sha:
            raise ValueError("Immutable candidate source changed: " + relative)
    original_dataset = root / "outputs" / SOURCE_DATASET
    original_pools_path = original_dataset / "pools.json"
    original_pools = load(original_pools_path)
    original_audit = load(original_dataset / "completion_audit.json")
    if original_audit.get("passed") is not True or len(original_pools) != 40:
        raise ValueError("Original paired 40-pool source was not accepted")
    original_membership = {p["pool_id"]: p for p in original_pools}
    if len(original_membership) != 40:
        raise ValueError("Original source has duplicate pools")
    contract, derive_temporal_recovery = annotation_modules(root)
    negative_witness_function, negative_witness_path = separation_extractor(root)
    pool_roles, groups, identities, datasets = {}, {}, set(), set()
    counts, contexts, signatures, rows = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter)), defaultdict(dict), []
    output.mkdir(parents=True, exist_ok=False)
    try:
        for record in records:
            pool, cid, split = record["pool_id"], record["candidate_id"], record["split"]
            parsed = re.fullmatch(r"task(\d+)_state(\d+)_(.+)", pool)
            if not parsed or parsed[3] not in CONTEXTS or type(cid) is not int or cid not in range(4) or split not in ("train", "val"):
                raise ValueError("Unexpected source pool/candidate/membership")
            if (pool, cid) in identities or pool_roles.setdefault(pool, split) != split:
                raise ValueError("Duplicated candidate or pool split conflict")
            original_pool = original_membership.get(pool, {})
            if (original_pool.get("split") != split or original_pool.get("task") != int(parsed[1]) or
                original_pool.get("state") != int(parsed[2]) or
                {a["candidate_id"] for a in original_pool.get("candidates", [])} != set(range(4))):
                raise ValueError("Membership does not inherit the original frozen paired pool")
            identities.add((pool, cid))
            directory = contained(source, record["directory"])
            for name in ("temporal_teacher.json", "temporal_measurements.json", "replay_audit.json", "physical_teacher_arrays.npz"):
                if str((directory / name).relative_to(source)).replace("\\", "/") not in manifest:
                    raise ValueError("Unhashed per-candidate teacher evidence")
            teacher, measurements, replay = (load(directory / name) for name in
                ("temporal_teacher.json", "temporal_measurements.json", "replay_audit.json"))
            identity = teacher["identity"]
            correct = dict(dataset=SOURCE_DATASET, suite="libero90", task=int(parsed[1]), state=int(parsed[2]),
                candidate_id=cid, observed_block_id=pool + "/first_block")
            if identity != correct or teacher.get("role") != "offline_supervision_only" or teacher.get("split") != split:
                raise ValueError("Full source-scoped identity mismatch")
            group = (identity["dataset"], identity["suite"], identity["task"], identity["state"])
            if groups.setdefault(group, split) != split:
                raise ValueError("Task/state group leaks between train/val")
            datasets.add(identity["dataset"])
            if (replay.get("passed") is not True or replay.get("exact_start") is not True or replay.get("source_unchanged") is not True or
                not all(c.get("passed") is True for c in replay["checks"])):
                raise ValueError("Original frame alignment failed")
            if derive_temporal_recovery(teacher["frames"], entity_kind=teacher["kind"], joint_unit=teacher.get("joint_unit")) != measurements:
                raise ValueError("Frozen physical teacher/old masks differ from saved measurements")
            original = contained(root, teacher["source_folder"])
            if original != contained(root, f"outputs/{SOURCE_DATASET}/pools/{pool}/candidate_{cid}"):
                raise ValueError("Original candidate directory does not match full source identity")
            raw_hashes = {}
            for name in RAW_FILES:
                relative = str((original / name).relative_to(root)).replace("\\", "/")
                if relative not in original_hashes:
                    raise ValueError("Original actual arrays were not source-hashed")
                raw_hashes[name] = original_hashes[relative]
            evidence_hashes = dict(actual_rgb=hashlib.sha256((raw_hashes["primary.npy"] + raw_hashes["wrist.npy"]).encode("ascii")).hexdigest(),
                actual_primary=raw_hashes["primary.npy"], actual_wrist=raw_hashes["wrist.npy"], actual_proprio=raw_hashes["proprio.npy"],
                physical_measurement=sha(directory / "temporal_measurements.json"), private_projection=sha(directory / "temporal_teacher.json"),
                private_segmentation_arrays=sha(directory / "physical_teacher_arrays.npz"),
                replay_audit=sha(directory / "replay_audit.json"))
            observed_teacher = actual_interface_boundaries(teacher, directory / "physical_teacher_arrays.npz", original,
                separation_function=negative_witness_function)
            sidecar = candidate_certificates(observed_teacher, measurements, replay, evidence_hashes, contract,
                certificate_version="v2", original_proprio=np.load(original / "proprio.npy", allow_pickle=False))
            sidecar["actual_rgb_boundary_evidence"] = [dict(step=f["step"], views={v: x["contact_interfaces"]
                for v, x in f["observation_evidence"]["views"].items()}) for f in observed_teacher["frames"]]
            sidecar["actual_rgb_negative_interface_evidence"] = [dict(step=f["step"],views={v:x.get("negative_interfaces",{})
                for v,x in f["observation_evidence"]["views"].items()}) for f in observed_teacher["frames"]]
            sidecar["negative_interface_extractor_sha256"] = sha(negative_witness_path)
            sidecar["source_directory"] = record["directory"]
            sidecar["split"] = split
            sidecar["actual_rgb_hash_definition"] = "SHA256(primary_file_SHA256 + wrist_file_SHA256), ASCII concatenation"
            sidecar["report_context_only"] = parsed[3]
            dest = output / "certificates" / pool / f"candidate_{cid}" / "observability_certificates.json"
            dump(dest, sidecar)
            rows.append(dict(pool_id=pool, candidate_id=cid, split=split, identity=identity,
                sidecar=str(dest.relative_to(output)).replace("\\", "/"), sidecar_sha256=sha(dest)))
            for name in contract.ATOM_NAMES:
                physical = any(r["physics"].get(name, {}).get("value") is True and r["physics"].get(name, {}).get("measurement_valid") is True for r in measurements["frames"])
                enabled = any(r["atoms"][name]["new_mask"] and r["atoms"][name]["physical_value"] is True for r in sidecar["labels"])
                counts[name]["physical_positive_candidates"] += int(physical)
                counts[name]["certified_positive_candidates"] += int(enabled)
                contexts[parsed[3]][name]["physical_positive_candidates"] += int(physical)
                contexts[parsed[3]][name]["certified_positive_candidates"] += int(enabled)
                signatures[(pool, name)][cid] = dict(physical_positive=physical, certified_positive=enabled,
                    first_certified=sidecar["first_certified_events"][name])
                for row in sidecar["labels"]:
                    atom = row["atoms"][name]
                    counts[name]["measured_positive_frames"] += int(atom["measurement_valid"] and atom["physical_value"] is True)
                    counts[name]["certified_positive_frames"] += int(atom["new_mask"] and atom["physical_value"] is True)
                    counts[name]["certified_negative_frames"] += int(atom["new_mask"] and atom["physical_value"] is False)
                    counts[name]["masked_frames"] += int(not atom["new_mask"])
                    counts[name]["transition_supervised_frames"] += int(atom["transition_mask"])
        if len(identities) != expected or len(pool_roles) != expected // 4 or any(
            {cid for p, cid in identities if p == pool} != set(range(4)) for pool in pool_roles):
            raise ValueError("All four candidates from every frozen source pool must be retained")
        differences = []
        for (pool, name), members in signatures.items():
            values = list(members.values())
            differences.append(dict(pool_id=pool, atom=name, candidates=members,
                physical_positive_variation=len({v["physical_positive"] for v in values}) > 1,
                certified_positive_vs_no_certification_variation=len({v["certified_positive"] for v in values}) > 1,
                certified_event_timing_variation=len({v["first_certified"] for v in values}) > 1,
                no_certification_is_not_negative_physical_truth=True))
        # A missing sufficient event/certificate is never a supervised zero.
        # Report legal negatives explicitly before anyone fits an event head.
        deadlines = []
        for name in contract.ATOM_NAMES:
            by_deadline = Counter()
            pool_supervision = defaultdict(list)
            for row in rows:
                sidecar = load(output / row["sidecar"])
                atoms = [x["atoms"][name] for x in sidecar["labels"] if 1 <= x["step"] <= 16]
                positive = any(x["new_mask"] and x["physical_value"] is True for x in atoms)
                # A literal observed negative at the last frame is an endpoint
                # negative, NOT proof that no event occurred before the deadline.
                last = atoms[-1]
                endpoint_negative = last["new_mask"] and last["physical_value"] is False
                by_deadline["observed_positive_by_step16"] += int(positive)
                by_deadline["legal_observed_negative_at_step16"] += int(endpoint_negative)
                by_deadline["legal_no_event_by_deadline_negative"] += 0
                by_deadline["unresolved_by_deadline"] += int(not positive)
                pool_supervision[row["pool_id"]].append((positive, bool(endpoint_negative)))
            by_deadline["same_pool_supervised_positive_vs_negative_differences"] = sum(
                any(a for a,b in values) and any(b for a,b in values) for values in pool_supervision.values())
            deadlines.append(dict(atom=name, deadline_step=16, counts=by_deadline,
                missing_certificate_is_not_negative=True,
                endpoint_negative_is_not_no_event_by_deadline=True))
        report = dict(schema=SCHEMA_V2, inherits=[SCHEMA], passed=True, integrity_passed=True, training_ready=False,
            source=str(source), completed_candidates=expected, pools=expected // 4, supervision_only=True,
            physical_supervision_partial=True, observability_recovery_label_ready=False,
            no_original_masks_or_values_changed=True, physical_vs_certified_counts=counts,
            context_report_only_counts={k: dict(v) for k, v in contexts.items()},
            within_pool_candidate_differences=differences, train_val_group_leakage=0,
            event_by_deadline_coverage=deadlines,
            split_candidates=dict(Counter(r["split"] for r in rows)),
            annotation_code_verified=code_paths, source_manifest_sha256=sha(source / "SHA256SUMS.txt"),
            original_membership_sha256=sha(original_pools_path),
            builder_sha256=sha(Path(__file__)), contract_sha256=sha(Path(contract.__file__)),
            negative_interface_extractor_sha256=sha(negative_witness_path),
            new_queries=0, new_candidates=0, training_started=False,
            limitations=["Actual RGB/depth correspondence is algorithmic silver, not human truth or deployed detector capability.",
                "Frame0 and unmatched frame1 cannot establish an earlier event; motion references start at step2 or later.",
                "A visible interface on one finger does not unmask the unseen other finger's contact head.",
                "Typed support interfaces can certify only a concrete visible counterpart; absent observed no-contact separation / vertical reference keeps lift or release masked.",
                "Carrying is probabilistic silver support from persistent actual RGB tracks, frozen 3D teacher and measured proprio, not literal finger-contact truth.",
                "Different-depth surface pixel displacements need not be equal; each surface retains the original <=2px actual-versus-projection fit gate.",
                "Event supervision can be positive-only; missing certificates are not negatives and do not make a ranking head training-ready.",
                "No explicit re-observation action or resolved cross-view-conflict certificate is present in this source.",
                "No certified event is unknown, not negative grasp truth; candidate variation is not a validated ranking benefit.",
                "Context only groups reports and never supplies observation or physical supervision."])
        dump(output / "records.json", rows)
        dump(output / "observability_label_audit.json", report)
        hashes = {str(p.relative_to(output)).replace("\\", "/"): sha(p) for p in output.rglob("*") if p.is_file()}
        (output / "SHA256SUMS.txt").write_text("".join(f"{h}  {p}\n" for p, h in sorted(hashes.items())), encoding="utf-8")
        print(json.dumps(dict(passed=True, candidates=expected, pools=expected // 4, counts=counts, training_ready=False), ensure_ascii=False))
        return report
    except Exception as error:
        dump(output / "failure.json", dict(passed=False, completed_candidates=len(rows), error_type=type(error).__name__,
            message=str(error), source_retained=True, training_started=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.source, args.output)


if __name__ == "__main__":
    main()

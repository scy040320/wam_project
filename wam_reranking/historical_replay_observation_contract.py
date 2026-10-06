"""Independent offline certificate for an immutable, actually executed chain.

New replay proprio is NOT original proprio. Cross-block candidate identities
are retained. All masks below are new sidecars, never amendments to old labels.
Only the file-loading entry point verifies lineage and issues a certificate.
This is conservative algorithmic silver evidence, not deployment X or a
universal grasp detector. Missing positive evidence is unknown, not negative.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from .observation_geometry import local_rgb_registration
from .observable_separation import negative_contact_witness
from .temporal_recovery_teacher import FROZEN_CONTRACT

SCHEMA = "historical_replay_obs_certificate_v1"
SOURCE_SCHEMA = "archived_terminal_tail_observation_metrics_v1"
IDENTITY_FIELDS = ("dataset", "suite", "task", "state", "candidate_id", "observed_block_id", "source_step")
VIEWS = ("primary", "wrist")
ROLES = ("target", "eef", "anchor", "left", "right")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""): h.update(part)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _within(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Relative evidence reference required")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Evidence reference escapes root")
    return path


def _finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def _rgb_pass(x):
    return (isinstance(x, dict) and x.get("passed") is True and
        all(_finite(x.get(k)) for k in ("mean_abs", "p95", "fraction_gt5", "psnr")) and
        0 <= x["mean_abs"] <= 2 and 0 <= x["p95"] <= 8 and
        0 <= x["fraction_gt5"] <= .06 and x["psnr"] >= 30)


def _qc(q):
    return (isinstance(q, dict) and q.get("passed") is True and
        all(_finite(q.get(k)) for k in ("sim_max_abs", "sim_p95", "fraction_gt_1e3")) and
        0 <= q["sim_max_abs"] <= 1e-3 and 0 <= q["sim_p95"] <= 1e-4 and
        q["fraction_gt_1e3"] == 0 and q.get("contact_atoms_match") is True and
        q.get("terminal_sequence_match") is True)


def _identity(x):
    if not isinstance(x, dict) or set(x) != set(IDENTITY_FIELDS):
        raise ValueError("Exact seven-field original source identity required")
    if any(type(x[k]) is not int or x[k] < 0 for k in ("task", "state", "candidate_id", "source_step")):
        raise ValueError("Original identity has invalid numeric fields")
    if any(not isinstance(x[k], str) or not x[k] for k in ("dataset", "suite", "observed_block_id")):
        raise ValueError("Original identity strings required")
    return dict(x)


def _vector(x, n=3):
    if not isinstance(x, list) or len(x) != n or not all(_finite(v) for v in x):
        return None
    return np.asarray(x, dtype=float)


def _track(report, masks, start, end, view, role):
    """Actual RGB tracks only; camera projection alone never supplies motion."""
    if (not isinstance(report, dict) or report.get("verified") is not True or
        report.get("geometry_motion_support") is not True or
        report.get("evidence_kind") != "actual_RGB_geometry_correspondence"):
        return None
    points = report.get("tracks", [])
    if len(points) < 3 or report.get("count") != len(points): return None
    starts, actual, projected = set(), [], []
    for p in points:
        keys = ("start_pixel_xy", "end_pixel_xy", "projected_end_pixel_xy", "camera_only_end_pixel_xy")
        vec = [_vector(p.get(k), 2) for k in keys]
        if any(v is None for v in vec): return None
        if any(not _finite(p.get(k)) or not 0 <= p[k] <= 1 for k in
               ("forward_backward_error_px", "projected_endpoint_error_px")): return None
        a, b, pr, ca = vec
        if np.linalg.norm(b-pr) > 1: return None
        key = tuple(a)
        if key in starts: return None
        starts.add(key)
        for coordinate, step in ((a, start), (b, end)):
            mask = masks[f"{view}_role_{role}"][step]
            xx, yy = np.floor(coordinate + .5).astype(int)
            if not (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx]): return None
        actual.append(b-ca); projected.append(pr-ca)
    a = np.asarray([statistics.median(v[i] for v in actual) for i in (0, 1)])
    p = np.asarray([statistics.median(v[i] for v in projected) for i in (0, 1)])
    return dict(actual=a, projected=p, count=len(points)) if np.linalg.norm(a-p) <= 2 else None


def _view_ok(frames, arrays, step, view):
    if step == 0: return False
    saved = frames[step]["observation_evidence"].get("views", {}).get(view, {})
    if saved.get("fresh_geometry_registered_to_actual_input") is not True or not _rgb_pass(saved.get("global_rgb")):
        return False
    actual, rendered = arrays[f"{view}_actual"][step], arrays[f"{view}_rendered"][step]
    return _rgb_pass(local_rgb_registration(actual, rendered, np.ones(actual.shape[:2], bool)))


def _noise(frames, step, view, role):
    value = frames[step]["observation_evidence"].get("views", {}).get(view, {}).get("repeat_noise", {}).get(role, {})
    n = value.get("centroid_noise_px")
    r = value.get("rgb_repeat")
    return (n, r) if _finite(n) and n >= 0 and _rgb_pass(r) else None


def _role(frames, arrays, step, view, role, track, view_ready=None, role_ready=None):
    if track is None or not (view_ready[(step,view)] if view_ready is not None else _view_ok(frames, arrays, step, view)): return None
    if role_ready is not None and (step,view,role) in role_ready: return role_ready[(step,view,role)]
    mask = arrays[f"{view}_role_{role}"][step]
    local = local_rgb_registration(arrays[f"{view}_actual"][step], arrays[f"{view}_rendered"][step], mask)
    saved = frames[step]["observation_evidence"]["views"][view].get("roles", {}).get(role, {})
    if (mask.sum() < 16 or saved.get("rendered_pixels") != int(mask.sum()) or
        not _rgb_pass(local) or local.get("texture_sufficient") is not True or
        not _finite(local.get("actual_texture_range")) or local["actual_texture_range"] <= 0):
        if role_ready is not None: role_ready[(step,view,role)] = None
        return None
    if role_ready is not None: role_ready[(step,view,role)] = local
    return local


def _interval(frames, arrays, interval, end, view_ready=None, role_ready=None):
    start, stop, view = interval.get("start_step"), interval.get("end_step"), interval.get("view")
    if type(start) is not int or type(stop) is not int or not 0 < start < stop == end or view not in VIEWS:
        return None
    # No backfilled candidate identity, and an endpoint belongs to its OWN block.
    if interval.get("source_start_identity") != frames[start]["source_identity"] or interval.get("source_end_identity") != frames[stop]["source_identity"]:
        raise ValueError("RGB interval source identity changed or was backfilled")
    if interval.get("no_future_identity_backfill") is not True or interval.get("cross_block_context_preserves_distinct_candidate_ids") is not True:
        return None
    if not all((view_ready[(s,view)] if view_ready is not None else _view_ok(frames, arrays, s, view)) and _qc(frames[s]["repeat_qc"]) for s in range(start, stop+1)):
        return None
    tracks, noises, locals_ = {}, {}, {}
    for role in ROLES:
        track = _track(interval.get("tracks", {}).get(role), arrays, start, stop, view, role)
        noises_pair = [_noise(frames, s, view, role) for s in (start, stop)]
        if any(n is None for n in noises_pair): continue
        reported = interval.get("same_candidate_repeat_noise", {}).get(role)
        expected = max(n[0] for n in noises_pair)
        if not _finite(reported) or reported != expected: continue
        first = _role(frames, arrays, start, view, role, track, view_ready, role_ready)
        last = _role(frames, arrays, stop, view, role, track, view_ready, role_ready)
        if first is not None and last is not None:
            tracks[role] = track; noises[role] = expected; locals_[role] = last
    return dict(start=start, end=stop, view=view, tracks=tracks, noise=noises, local=locals_)


def _motion(interval, relative=False):
    tracks = interval["tracks"]
    if not all(r in tracks for r in ("target", "eef")): return False
    a, b = tracks["target"]["actual"], tracks["eef"]["actual"]
    noise = max(interval["noise"][r] for r in ("target", "eef"))
    if relative:
        return np.linalg.norm(a-b) >= 2 and np.linalg.norm(a-b) > 3*noise
    # Perspective/camera geometry makes different 2D translations legitimate.
    # Common motion/rigidity is measured in 3D, never pixel-delta equality.
    return (np.linalg.norm(a) >= 2 and np.linalg.norm(b) >= 2 and
            np.linalg.norm(a) > 3*interval["noise"]["target"] and
            np.linalg.norm(b) > 3*interval["noise"]["eef"])


def _carried_physics(window):
    if len(window) != FROZEN_CONTRACT.sustained_frames: return False
    if not all(f["physics"].get("left_finger_target_contact") is True and
               f["physics"].get("right_finger_target_contact") is True and
               f["physics"].get("target_support_contact") is False for f in window): return False
    target = [_vector(f["physics"].get("target_pos_m")) for f in window]
    eef = [_vector(f["physics"].get("eef_pos_m")) for f in window]
    if any(v is None for v in target+eef): return False
    r = target[0]-eef[0]
    return bool(min(np.linalg.norm(target[-1]-target[0]), np.linalg.norm(eef[-1]-eef[0])) >= FROZEN_CONTRACT.common_motion_min_m and
            max(np.linalg.norm(t-e-r) for t, e in zip(target, eef)) <= FROZEN_CONTRACT.relative_drift_max_m)


def _near_interface_rgb(witness):
    """Nearest visible surface points must themselves be actual RGB witnesses.

    A textured far-away edge cannot certify an untextured nearby fingertip.
    This remains a visible-surface proxy, not proof of hidden 3D clearance.
    """
    typed = witness.get("actual_boundary_coordinates", {})
    target, finger = typed.get("target", []), typed.get("finger", [])
    if not target or not finger: return False
    nearest_textured = min(float(np.linalg.norm(np.asarray(a)-b)) for a in target for b in finger)
    return nearest_textured <= witness["actual_surface_gap_px"] + 1e-9


def _certify_verified(teacher, arrays, lineage, original_physical_events=None):
    """Internal pure evaluator; caller MUST complete load/file/lineage checks."""
    frames = teacher["frames"]; reports = []; prior = None
    view_ready={(s,v):_view_ok(frames,arrays,s,v) for s in range(len(frames)) for v in VIEWS};role_ready={}
    valid_intervals = []
    for step, frame in enumerate(frames):
        intervals = [_interval(frames, arrays, x, step, view_ready, role_ready) for x in
                     frame["observation_evidence"].get("actual_rgb_temporal_intervals", [])]
        intervals = [x for x in intervals if x is not None]; valid_intervals.append(intervals)
        carry, release, reasons = False, False, []
        window = frames[max(0, step-2):step+1]
        physical_carry = _carried_physics(window)
        if physical_carry:
            # Sustained identities must belong to the SAME motion view. Do not
            # splice different cameras into unaudited cross-view correspondence.
            carry = any(x["start"] == step-2 and _motion(x) and
                all(any(y["view"] == x["view"] and all(r in y["tracks"] for r in ("target", "eef"))
                    for y in valid_intervals[s]) for s in range(step-2,step+1)) for x in intervals)
            if carry: prior = step
            else: reasons.append("carry_actual_RGB_common_motion_or_identity_unverified")
        physical_release = False; witnesses = []
        if prior is not None and step > prior and step >= 1:
            p = frames[prior]["physics"]; now = frame["physics"]
            release_window = frames[step-1:step+1]
            no_contact = all(f["physics"].get("left_finger_target_contact") is False and f["physics"].get("right_finger_target_contact") is False for f in release_window)
            pt, pe = _vector(p.get("target_pos_m")), _vector(p.get("eef_pos_m"))
            nt, ne = _vector(now.get("target_pos_m")), _vector(now.get("eef_pos_m"))
            pa, na = p.get("gripper_aperture_m"), now.get("gripper_aperture_m")
            physical_release = bool(no_contact and all(x is not None for x in (pt, pe, nt, ne)) and
                _finite(pa) and _finite(na) and na-pa >= FROZEN_CONTRACT.release_opening_min_m and
                np.linalg.norm((nt-ne)-(pt-pe)) >= FROZEN_CONTRACT.release_separation_min_m)
            if physical_release:
                for s in (step-1, step):
                    sides = {}
                    for side in ("left", "right"):
                        for interval in valid_intervals[s]:
                            if not all(r in interval["tracks"] for r in ("target", side)): continue
                            view = interval["view"]; tn, tr = _noise(frames, s, view, "target"); fn, fr = _noise(frames, s, view, side)
                            witness = negative_contact_witness(actual_rgb=arrays[f"{view}_actual"][s],
                                target_mask=arrays[f"{view}_role_target"][s], finger_mask=arrays[f"{view}_role_{side}"][s],
                                target_registration=interval["local"]["target"], finger_registration=interval["local"][side],
                                target_identity_certified=True, finger_identity_certified=True,
                                target_repeat_noise_px=tn, finger_repeat_noise_px=fn,
                                target_repeat_rgb_mean_abs=tr["mean_abs"], finger_repeat_rgb_mean_abs=fr["mean_abs"],
                                fresh_geometry_registered=True, physical_contact=False, side=side)
                            if witness is not None and _near_interface_rgb(witness):
                                sides[side] = dict(view=view, witness=witness); break
                    witnesses.append(dict(chain_index=s, own_fingers=sides))
                own_motion = any(prior <= x["start"] <= step-1 and _motion(x, relative=True) for x in intervals)
                release = bool(all(len(x["own_fingers"]) == 2 for x in witnesses) and own_motion)
                if not release: reasons.append("release_own_finger_no_contact_or_actual_relative_motion_unverified")
        if not physical_carry and not physical_release: reasons.append("sufficient_event_not_established_not_a_negative_label")
        reports.append(dict(chain_index=step, source_identity=frame["source_identity"],
            carried_sufficient_evidence=dict(value=True if carry else None, supervision_mask=carry, physical_sufficient=physical_carry),
            released_sufficient_evidence=dict(value=True if release else None, supervision_mask=release,
                certified_prior_required_release_eval=physical_release,
                raw_physical_release=(original_physical_events or {}).get(step),
                raw_physical_event_source="hashed_temporal_measurements_original_atom_unchanged" if step in (original_physical_events or {}) else None,
                prior_certified_carry_source=frames[prior]["source_identity"] if physical_release else None,
                prior_certified_carry_chain_index=prior if physical_release else None),
            holding_after_measured_release=dict(value=False if release else None, supervision_mask=release),
            own_no_contact_witnesses=witnesses, unavailable_reasons=reasons))
    return dict(schema=SCHEMA, source_schema=teacher["schema"], split=teacher["split"], frames=reports,
        lineage_audit=lineage, new_observation_sidecar_only=True, original_masks_modified=False,
        newly_measured_proprio_is_not_original=True, candidate_ids_preserved=True,
        certificates_are_algorithmic_silver=True, training_ready=False, deployment_features_created=False,
        limitations=["Selected executed arms do not provide pool-wide counterfactual supervision.",
            "Positive sufficient carry/release evidence does not identify every grasp/release.",
            "Surface-gap witnesses concern visible typed surfaces, not hidden 3D clearance.",
            "Depth/flow correspondence inherits the source-SHA frozen adapter; no complete optical-flow recomputation is claimed.",
            "Control-boundary contact samples cannot establish substep contact continuity."])


def certify_historical_chain(project_root, observation_root, scenario):
    """Read-only file/hash/RGB/proprio/chain audit, then independent certificates."""
    project_root, observation_root = Path(project_root).resolve(), Path(observation_root).resolve()
    manifest = _read(observation_root/"frozen_membership.json")
    if manifest.get("complete_blocks_count_unchanged") != 797 or manifest.get("same_arm_repeat_not_surrogate") is not True:
        raise ValueError("Frozen historical source and own-arm repeat manifest required")
    rows = [x for x in manifest.get("chains", []) if x.get("scenario") == scenario]
    if len(rows) != 1: raise ValueError("Unique frozen chain required")
    row = rows[0]; folder = _within(observation_root, "chains/"+scenario)
    if row.get("split") not in ("train", "val"): raise ValueError("Frozen train/val role required")
    hashes = manifest.get("source_sha256", {})
    if not hashes: raise ValueError("Original source hash table required")
    for name, digest in hashes.items():
        if sha256(_within(project_root, name)) != digest: raise ValueError("Original source hash changed: "+name)
    sums = observation_root/"SHA256SUMS.txt"
    entries = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in entries or sha256(_within(observation_root, name)) != digest:
            raise ValueError("Observation output manifest mismatch")
        entries[name] = digest
    for name in ("temporal_teacher.json", "temporal_measurements.json", "replay_arrays.npz", "replay_audit.json"):
        if f"chains/{scenario}/{name}" not in entries: raise ValueError("Incomplete hash manifest")
    teacher = _read(folder/"temporal_teacher.json"); audit = _read(folder/"replay_audit.json")
    measurements = _read(folder/"temporal_measurements.json")
    if (teacher.get("schema") != SOURCE_SCHEMA or teacher.get("role") != "offline_supervision_only" or
        teacher.get("split") != row["split"] or teacher.get("training_ready") is not False or
        teacher.get("original_short_endpoint_proprio_available") is not False or
        teacher.get("original_per_step_proprio_not_fabricated") is not True or
        teacher.get("original_complete_blocks_count_unchanged") != 797 or audit.get("passed") is not True or
        audit.get("all_original_RGB_and_terminal_lengths_match") is not True or audit.get("new_mask_not_created") is not True or
        audit.get("original_short_proprio_check") is not None or not _qc(audit.get("combined_repeat_qc")) or
        not _qc(audit.get("short_repeat_qc")) or len(audit.get("prefix_repeat_qc", [])) != len(row.get("chain", [])) or
        not all(_qc(q) for q in audit.get("prefix_repeat_qc", []))):
        raise ValueError("Historical technical replay audit incomplete")
    arrays_path = folder/"replay_arrays.npz"; digest = sha256(arrays_path)
    if teacher.get("replay_arrays_sha256") != digest: raise ValueError("Replay NPZ lineage changed")
    with np.load(arrays_path, allow_pickle=False) as loaded: arrays = {k: loaded[k].copy() for k in loaded.files}
    blocks = row.get("chain", [])+[row["tail"]]
    expected = []
    for index, block in enumerate(blocks):
        length = block.get("valid_length", 16) if index == len(blocks)-1 else 16
        for step in range(0 if index == 0 else 1, length+1):
            expected.append(dict(dataset=manifest["source"], suite="libero90", task=row["task"], state=row["state"],
                candidate_id=block["candidate_id"], observed_block_id=block["directory"], source_step=step))
    frames = teacher.get("frames", [])
    if len(frames) != len(expected) or teacher.get("real_short_length") != row["tail"]["valid_length"]:
        raise ValueError("Actual historical chain length changed or padded")
    n = len(frames)
    for key in ("proprio", "qpos", "qvel"):
        a = arrays.get(key)
        if a is None or a.ndim != 2 or len(a) != n or not np.isfinite(a).all(): raise ValueError("Finite actual per-frame replay arrays required")
    if arrays["proprio"].shape != (n, 9): raise ValueError("Actual replay proprio must have nine dimensions")
    for view in VIEWS:
        for kind in ("actual", "rendered"):
            a = arrays.get(f"{view}_{kind}")
            if a is None or a.dtype != np.uint8 or a.ndim != 4 or a.shape[0] != n or a.shape[-1] != 3:
                raise ValueError("Exact actual uint8 RGB frames required")
        for role in ROLES:
            a = arrays.get(f"{view}_role_{role}")
            if a is None or a.dtype != bool or a.shape != arrays[f"{view}_actual"].shape[:3]: raise ValueError("Typed per-frame role masks required")
    cache = {}; endpoint_count = 0
    for index, (frame, identity) in enumerate(zip(frames, expected)):
        if frame.get("chain_index") != index or _identity(frame.get("source_identity")) != identity:
            raise ValueError("Original chain source identity/CID/step changed")
        evidence = frame.get("observation_evidence", {})
        if evidence.get("source_identity") != identity or evidence.get("chain_index") != index or not _qc(frame.get("repeat_qc")):
            raise ValueError("Observation/repeat source identity mismatch")
        mp = frame.get("measured_proprio", {})
        if (mp.get("source") != "newly_measured_verified_historical_replay" or mp.get("array_index") != index or
            mp.get("replay_lineage_sha256") != digest or mp.get("gripper_qpos_m") != arrays["proprio"][index, :2].tolist() or
            mp.get("eef_pos_m") != arrays["proprio"][index, 2:5].tolist()):
            raise ValueError("Newly measured replay proprio lineage invalid; cannot claim original")
        phys = frame.get("physics", {})
        if (phys.get("eef_pos_m") != mp["eef_pos_m"] or not _finite(phys.get("gripper_aperture_m")) or
            abs(phys["gripper_aperture_m"]-float(np.abs(arrays["proprio"][index, :2]).sum())) > 1e-12):
            raise ValueError("Physics/new replay proprio mismatch")
        checks = frame.get("original_replay_checks", {})
        for view in VIEWS:
            ref = frame.get("original_image_sources", {}).get(view, {})
            path = ref.get("path")
            if hashes.get(path) != ref.get("sha256"): raise ValueError("Original RGB reference not in frozen source table")
            original = _within(project_root, path)
            if path not in cache:
                if ref.get("array_name") is None: cache[path] = np.load(original, allow_pickle=False)
                else:
                    with np.load(original, allow_pickle=False) as saved: cache[path] = {v: saved[v].copy() for v in VIEWS}
            expected_path = str(Path(identity["observed_block_id"])/("trajectory.npz" if identity["source_step"] else f"evidence/block/O_t_{view}.npy")).replace("\\", "/")
            if path != expected_path or ref.get("array_index") != (identity["source_step"]-1 if identity["source_step"] else None) or ref.get("array_name") != (view if identity["source_step"] else None):
                raise ValueError("Original image array reference is not own source frame")
            image = cache[path][view][ref["array_index"]] if ref["array_name"] else cache[path]
            if not np.array_equal(image, arrays[f"{view}_actual"][index]): raise ValueError("Replay actual RGB is not immutable original RGB")
            if index > 0 and not _rgb_pass(checks.get(view)): raise ValueError("Original RGB replay gate failed")
        if identity["source_step"] == 16:
            endpoint = str(Path(identity["observed_block_id"])/"evidence/block/actual_observed_proprio.npy").replace("\\", "/")
            if endpoint not in hashes: raise ValueError("Available original endpoint proprio hash missing")
            original = np.load(_within(project_root, endpoint), allow_pickle=False)
            error = float(np.abs(original-arrays["proprio"][index]).max())
            reported = checks.get("endpoint_proprio_max_abs")
            if original.shape != (9,) or error > 1e-3 or not _finite(reported) or reported > 1e-3 or abs(error-reported)>1e-12:
                raise ValueError("Available original endpoint proprio gate failed")
            endpoint_count += 1
        if identity["observed_block_id"] == row["tail"]["directory"] and (checks.get("endpoint_proprio_max_abs") is not None or checks.get("original_endpoint_proprio_available") is not False):
            raise ValueError("Short original endpoint proprio was fabricated")
    bridge = dict(source_file_hashes_verified=True, observation_file_hashes_verified=True, original_RGB_exact_verified=True,
        available_original_endpoint_proprio_verified=endpoint_count, original_short_endpoint_proprio_available=False,
        replay_arrays_sha256=digest, chain_frames=n, frozen_complete_blocks_unchanged=797,
        blocks=[dict(observed_block_id=b["directory"], candidate_id=b["candidate_id"]) for b in blocks],
        bridge_basis="frozen_actual_chain_membership_plus_original_RGB_hash_and_source_step_continuity",
        replay_proprio_source="newly_measured_verified_historical_replay_not_original_per_step_proprio")
    physical_events={}
    for measurement in measurements.get("frames", []):
        source=measurement.get("current_source", {})
        index=source.get("chain_step")
        identity={k:source.get(k) for k in IDENTITY_FIELDS if k!="source_step"}|{"source_step":source.get("local_step")}
        if type(index) is not int or not 0 <= index < n or identity!=frames[index]["source_identity"] or index in physical_events:
            raise ValueError("Hashed temporal physical measurement source identity mismatch")
        atom=measurement.get("physics", {}).get("released_sufficient_evidence", {})
        physical_events[index]=dict(value=atom.get("value"), measurement_valid=atom.get("measurement_valid"),
            source_identity=identity, chain_index=index, original_supervision_mask=atom.get("supervision_mask"))
    certificate = _certify_verified(teacher, arrays, bridge, physical_events)
    for name, original_digest in hashes.items():
        if sha256(_within(project_root,name)) != original_digest: raise ValueError("Original source changed during certificate audit")
    certificate["lineage_audit"]["source_hashes_unchanged_after_audit"] = True
    return certificate

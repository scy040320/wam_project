"""Offline observation certificates; never deployment features or grasp truth.

Three independent layers are preserved: the frozen simulator measurement, an
actual-RGB/proprio observation proxy, and a learnable supervision mask. Geometry
co-presence and full-frame render registration do not certify an interface.
All accepted evidence is source/candidate scoped, prefix causal, and hash-linked.
The extractor, not this CPU validator, must measure ROI/flow/depth witnesses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

from .temporal_recovery_teacher import FROZEN_CONTRACT as PHYSICS_CONTRACT

SCHEMA = "recovery_observability_certificate_v1"
INHERITS = ("candidate_recovery_labels_v2", "temporal_recovery_measurements_v1")
IDENTITY_FIELDS = ("dataset", "suite", "task", "state", "candidate_id", "observed_block_id")
ROLE_NAMES = ("target", "eef", "anchor", "support")
INTERFACE_NAMES = ("left", "right", "support", "anchor")
IDENTITY_BASES = ("unique_scene_instance_with_actual_rgb_track", "audited_actual_rgb_instance_correspondence")
TRACKING_BASES = ("actual_rgb_geometric_correspondence", "audited_actual_rgb_instance_track")
ATOM_NAMES = ("left_finger_target_contact", "right_finger_target_contact", "target_support_contact",
              "carried_sufficient_evidence", "lifted_sufficient_evidence", "released_sufficient_evidence",
              "joint_state_changed")


@dataclass(frozen=True)
class ObservationContract:
    image_mean_abs: float = 2.
    image_p95: float = 8.
    image_fraction_gt5: float = .06
    image_psnr: float = 30.
    proprio_max_abs: float = 1e-3
    minimum_role_pixels: int = 16
    identity_quality: float = .5
    minimum_interface_boundary_pixels: int = 2
    maximum_interface_depth_error_m: float = .002
    minimum_motion_px: float = 2.
    maximum_motion_fit_error_px: float = 2.
    repeat_noise_multiplier: float = 3.


FROZEN_CONTRACT = ObservationContract()


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _identity(value):
    if not isinstance(value, Mapping) or set(value) != set(IDENTITY_FIELDS):
        raise ValueError("Six source-scoped identity fields required")
    for name in ("dataset", "suite", "observed_block_id"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError("Nonempty dataset/suite/block identity required")
    for name in ("task", "state", "candidate_id"):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError("Task/state/candidate IDs must be nonnegative integers")
    return tuple(value[name] for name in IDENTITY_FIELDS)


def _forbidden(value):
    if isinstance(value, Mapping):
        for name, item in value.items():
            if str(name).lower() in ("attribution", "success", "final_success", "class_probs", "factor_probs", "cause", "gripper_command") or str(name).lower().startswith("predicted_"):
                raise ValueError("Outcome, predicted attribution or command cannot certify physical observation")
            _forbidden(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _forbidden(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite evidence metric")


def _hashes(record):
    hashes = record.get("evidence_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        return False
    for key, value in hashes.items():
        if not isinstance(key, str) or not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
            raise ValueError("Evidence hashes must be named SHA256 values")
    return record.get("source_hashes_verified") is True and "actual_rgb" in hashes


def _witness_steps(record, step):
    used = record.get("used_steps", [step])
    if not isinstance(used, list) or any(type(s) is not int or s < 0 or s > step for s in used):
        raise ValueError("Evidence may not use future frames to backfill an earlier certificate")


def _cert(value, *reasons, **extra):
    return dict(schema=SCHEMA, value=value, reasons=list(reasons), source="offline_actual_rgb_proprio_certificate",
        deployment_allowed=False, observation_proxy_only=True, **extra)


def _registration(metrics, contract):
    if not isinstance(metrics, Mapping) or not all(_finite_number(metrics.get(k)) for k in ("mean_abs", "p95", "fraction_gt5", "psnr")):
        return None
    if any(metrics[k] < 0 for k in ("mean_abs", "p95", "fraction_gt5")) or metrics["fraction_gt5"] > 1:
        raise ValueError("Invalid RGB residual units/range")
    return (metrics["mean_abs"] <= contract.image_mean_abs and metrics["p95"] <= contract.image_p95 and
            metrics["fraction_gt5"] <= contract.image_fraction_gt5 and metrics["psnr"] >= contract.image_psnr)


def _role(metrics, frame_usable, contract):
    if not isinstance(metrics, Mapping) or frame_usable is not True:
        return dict(available=None, identity_certified=None, reasons=["actual_frame_or_local_role_evidence_missing"])
    local = _registration(metrics.get("local_rgb"), contract)
    pixels = metrics.get("rendered_pixels")
    if type(pixels) is not int or pixels < 0:
        return dict(available=None, identity_certified=None, reasons=["rendered_role_pixels_missing"])
    available = local is True and pixels >= contract.minimum_role_pixels
    quality = metrics.get("identity_quality")
    identity = (available and metrics.get("identity_rgb_unique") is True and
        metrics.get("identity_basis") in IDENTITY_BASES and metrics.get("actual_rgb_structure_nonconstant") is True and
        _finite_number(quality) and quality >= contract.identity_quality)
    return dict(available=available if local is not None else None,
        identity_certified=identity if available else None,
        reasons=[] if identity else ["local_registration_visibility_or_actual_rgb_identity_unverified"])


def _interface(metrics, roles, frame_usable, contract):
    if not isinstance(metrics, Mapping) or frame_usable is not True:
        return _cert(None, "contact_interface_evidence_missing", contact_value=None)
    contact = metrics.get("physics_contact")
    if contact is not None and type(contact) is not bool:
        raise ValueError("Contact truth is an offline boolean, never a command")
    second = metrics.get("other_surface_role", "eef")
    if second not in ROLE_NAMES:
        raise ValueError("Contact surface must bind an audited rendered-body role")
    local = _registration(metrics.get("local_rgb"), contract)
    surfaces = all(roles.get(name, {}).get("identity_certified") is True for name in ("target", second))
    common = surfaces and local is True and metrics.get("surfaces_rgb_visible") is True
    count = metrics.get("rgb_boundary_pixels")
    if type(count) is not int or count < 0:
        return _cert(None, "actual_rgb_interface_boundary_not_measured", contact_value=contact)
    if not common or count < contract.minimum_interface_boundary_pixels:
        return _cert(False if local is False else None, "actual_rgb_interface_not_certified", contact_value=contact)
    if contact is True:
        depth = metrics.get("depth_error_m")
        if metrics.get("projection_in_frame") is not True or not _finite_number(depth):
            return _cert(None, "contact_point_projection_or_depth_missing", contact_value=contact)
        value = abs(depth) <= contract.maximum_interface_depth_error_m
        return _cert(value, *([] if value else ["contact_point_occluded_or_depth_inconsistent"]), contact_value=contact)
    if contact is False:
        gap = metrics.get("actual_surface_gap_px")
        value = (metrics.get("separation_rgb_verified") is True and _finite_number(gap) and gap >= contract.minimum_motion_px)
        return _cert(True if value else None, *([] if value else ["no_contact_requires_visible_actual_rgb_separation"]), contact_value=contact)
    return _cert(None, "physical_contact_not_measured", contact_value=None)


def certify_frame(frame_evidence, *, contract=FROZEN_CONTRACT):
    """Validate extracted LOCAL RGB/depth witnesses without reading simulator GT.

    ``roles`` refer to rendered finger BODY geoms, not contact collision geoms.
    ``interfaces`` keep left/right/support independent. Frame0 is deliberately
    diagnostic-only even if a whole-image replay check happened to pass.
    """
    if contract != FROZEN_CONTRACT:
        raise ValueError("Observation thresholds require a new version to change")
    _forbidden(frame_evidence); _identity(frame_evidence.get("identity"))
    step = frame_evidence.get("step")
    if type(step) is not int or step not in range(17):
        raise ValueError("Actual frames have indices 0..16")
    _witness_steps(frame_evidence, step)
    linked = _hashes(frame_evidence)
    pr = frame_evidence.get("proprio_max_abs")
    actual = step > 0 and linked and _finite_number(pr) and 0 <= pr <= contract.proprio_max_abs
    views = {}
    for name in ("primary", "wrist"):
        raw = frame_evidence.get("views", {}).get(name, {})
        global_rgb = _registration(raw.get("global_rgb"), contract)
        usable = actual and global_rgb is True
        roles = {role: _role(raw.get("roles", {}).get(role), usable, contract) for role in ROLE_NAMES}
        interfaces = {side: _interface(raw.get("interfaces", {}).get(side), roles, usable, contract) for side in INTERFACE_NAMES}
        views[name] = dict(registered_actual=usable, roles=roles, interfaces=interfaces)
    return dict(schema=SCHEMA, inherits=INHERITS, identity=dict(frame_evidence["identity"]), step=step,
        reference_certified=any(v["registered_actual"] for v in views.values()), views=views,
        frame0_diagnostic_only=step == 0, evidence_sha256=dict(frame_evidence.get("evidence_sha256", {})),
        contract=asdict(contract), deployment_allowed=False)


def _vec(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_finite_number(x) for x in value):
        return None
    return [float(x) for x in value]


def _norm(value):
    return math.sqrt(sum(x*x for x in value))


def _delta(a, b):
    return [x-y for x,y in zip(a,b)]


def _frame_index(certificates):
    frames = {}
    for frame in certificates:
        _identity(frame.get("identity"))
        step = frame.get("step")
        if (frame.get("schema") != SCHEMA or frame.get("deployment_allowed") is not False or
                type(step) is not int or step not in range(17)):
            raise ValueError("Only source-scoped frame certificates may define a motion window")
        if step in frames:
            raise ValueError("Duplicate frame certificates may not hide an identity mismatch")
        frames[step] = frame
    return frames


def _positive_event_certificate(record, name, identity, step=None):
    if not isinstance(record, Mapping):
        return False
    return (record.get("schema") == SCHEMA and record.get("name") == name and
        record.get("role") == "offline_supervision_only" and record.get("deployment_allowed") is False and
        record.get("new_mask") is True and record.get("physical_value") is True and
        record.get("measurement_valid") is True and type(record.get("step")) is int and
        1 <= record["step"] <= 16 and (step is None or record["step"] == step) and
        record.get("identity") is not None and _identity(record["identity"]) == identity)


def certify_temporal_motion(frame_certificates, motion_evidence, *, reference_role="eef", contract=FROZEN_CONTRACT):
    """Compare observed RGB motion to projection, above SAME-candidate noise.

    Projection alone never certifies motion. Actual and projected displacements
    must use the same camera-compensated coordinate system. ``reference_role``
    is eef (carrying/separation) or anchor (typed state/joint evidence).
    """
    if contract != FROZEN_CONTRACT or reference_role not in ("eef", "anchor", "support"):
        raise ValueError("Unsupported motion contract/reference role")
    _forbidden(motion_evidence); identity = _identity(motion_evidence.get("identity"))
    start, end = motion_evidence.get("start_step"), motion_evidence.get("end_step")
    if type(start) is not int or type(end) is not int or not 1 <= start < end <= 16:
        return _cert(None, "motion_reference_requires_real_post_action_frames", used_steps=[])
    _witness_steps(motion_evidence, end)
    base = dict(identity=dict(motion_evidence["identity"]), view=motion_evidence.get("view"),
                start_step=start, end_step=end, used_steps=list(range(start,end+1)), reference_role=reference_role)
    if not _hashes(motion_evidence) or motion_evidence.get("repeat_noise_verified") is not True:
        return _cert(None, "actual_rgb_or_same_candidate_repeat_noise_unverified", **base)
    if motion_evidence.get("same_candidate_repeat_identity") is None or _identity(motion_evidence["same_candidate_repeat_identity"]) != identity:
        return _cert(None, "repeat_noise_must_match_full_candidate_identity", **base)
    frames = _frame_index(frame_certificates)
    view = base["view"]
    for step in base["used_steps"]:
        f=frames.get(step)
        if f is None or _identity(f["identity"]) != identity or not f["reference_certified"]:
            return _cert(None, "motion_window_source_or_frame_unverified", **base)
        v=f["views"].get(view,{})
        if any(v.get("roles",{}).get(role,{}).get("identity_certified") is not True for role in ("target",reference_role)):
            return _cert(None, "actual_rgb_temporal_identity_unverified", **base)
    tracking_points = motion_evidence.get("actual_rgb_tracking_points")
    if (motion_evidence.get("camera_motion_compensated") is not True or
        motion_evidence.get("actual_rgb_tracking_verified") is not True or
        motion_evidence.get("actual_rgb_tracking_basis") not in TRACKING_BASES or
        type(tracking_points) is not int or tracking_points < 3):
        return _cert(None, "camera_motion_or_actual_rgb_tracking_unverified", **base)
    vectors = {name:_vec(motion_evidence.get(name)) for name in ("actual_subject_delta_px","actual_reference_delta_px","projected_subject_delta_px","projected_reference_delta_px")}
    noise = motion_evidence.get("repeat_noise_px")
    if any(v is None for v in vectors.values()) or not _finite_number(noise) or noise < 0:
        return _cert(None, "motion_measurements_or_repeat_noise_missing", **base)
    if max(_norm(_delta(vectors["actual_"+name],vectors["projected_"+name])) for name in ("subject_delta_px","reference_delta_px")) > contract.maximum_motion_fit_error_px:
        return _cert(False, "actual_rgb_and_projection_disagree", **base)
    a,b=vectors["actual_subject_delta_px"],vectors["actual_reference_delta_px"]
    signal_floor=max(contract.minimum_motion_px, contract.repeat_noise_multiplier*noise)
    def visible(signal):return signal >= contract.minimum_motion_px and signal > contract.repeat_noise_multiplier*noise
    relative=_delta(a,b)
    common=visible(min(_norm(a),_norm(b))) and _norm(relative)<=contract.maximum_motion_fit_error_px
    changing=visible(_norm(relative))
    return _cert(True, identity=dict(motion_evidence["identity"]), view=view,start_step=start,end_step=end,
        used_steps=base["used_steps"],reference_role=reference_role, common_motion=common,relative_change=changing,
        subject_motion=visible(_norm(a)), signal_floor_px=signal_floor,
        actual_relative_delta_px=relative, projected_relative_delta_px=_delta(vectors["projected_subject_delta_px"],vectors["projected_reference_delta_px"]))


def certify_supervision(atom, frame_certificates, *, motion_cert=None, prior_event_cert=None,
                        support_reference_cert=None, carried_event_cert=None, before_certificate=None):
    """Decide new masks without overwriting measurement values or OLD masks.

    ``atom`` supplies a frozen teacher value/validity/source/hash/identity, not a
    guessed physical value. Positive events require their independent witnesses.
    A missing event is never converted to a negative grasp/lift/release label.
    Actual future-to-query frames may supply labels only, never query features.
    """
    _forbidden(atom); identity=_identity(atom.get("identity")); step=atom.get("step");name=atom.get("name")
    if name not in ATOM_NAMES or type(step) is not int or step not in range(17):
        raise ValueError("Unsupported physical atom or step")
    old=atom.get("supervision_mask",False)
    if type(old) is not bool or type(atom.get("measurement_valid")) is not bool or atom.get("source")!="offline_simulator_measurement":
        raise ValueError("Frozen simulator measurement and explicit old mask required")
    value=atom.get("value")
    if value is not None and type(value) is not bool:
        raise ValueError("This event contract accepts boolean teacher atoms only")
    frames=_frame_index(frame_certificates)
    current=frames.get(step);new=False;reasons=[]
    if (not atom["measurement_valid"] or value is None or not _hashes(atom) or
        "physical_measurement" not in atom.get("evidence_sha256", {})):
        reasons.append("measurement_or_actual_evidence_hash_missing")
    elif current is None or _identity(current["identity"])!=identity or not current["reference_certified"]:
        reasons.append("cached_or_unverified_actual_event_frame")
    elif name.endswith("_contact"):
        side={"left_finger_target_contact":"left","right_finger_target_contact":"right","target_support_contact":"support"}[name]
        new=any(v["interfaces"][side]["value"] is True and v["interfaces"][side]["contact_value"] is value for v in current["views"].values())
        if not new:reasons.append("this_specific_contact_interface_not_visible")
    elif value is not True:
        reasons.append("absence_of_sufficient_event_is_not_a_physical_negative")
    else:
        m=motion_cert or {};view=m.get("view");start=m.get("start_step")
        consistent=(m.get("schema")==SCHEMA and m.get("source")=="offline_actual_rgb_proprio_certificate" and
            m.get("deployment_allowed") is False and m.get("value") is True and m.get("end_step")==step and
            m.get("identity") is not None and _identity(m["identity"])==identity and type(start) is int and
            m.get("used_steps")==list(range(start,step+1)) and view in ("primary","wrist") and
            all(s in frames and _identity(frames[s]["identity"])==identity and
                frames[s]["reference_certified"] and frames[s]["views"].get(view,{}).get("registered_actual") is True
                for s in range(start,step+1)))
        if not consistent:reasons.append("candidate_specific_actual_rgb_motion_missing")
        elif name=="carried_sufficient_evidence":
            window=list(range(max(1,step-PHYSICS_CONTRACT.sustained_frames+1),step+1))
            new=(atom.get("entity_kind")=="rigid" and len(window)==PHYSICS_CONTRACT.sustained_frames and
                 start==window[0] and m.get("reference_role")=="eef" and m.get("common_motion") is True and
                 all(any(frames[s]["views"][view]["interfaces"][side]["value"] is True and
                         frames[s]["views"][view]["interfaces"][side]["contact_value"] is True for side in ("left","right")) for s in window))
            if not new:reasons.append("sustained_rgb_common_motion_and_visible_interface_required")
        elif name=="joint_state_changed":
            unit=atom.get("joint_unit");delta=atom.get("joint_delta_from_reference")
            threshold=PHYSICS_CONTRACT.joint_change_rad if unit=="rad" else PHYSICS_CONTRACT.joint_change_m
            new=(atom.get("entity_kind")=="articulated" and unit in ("rad","m") and m.get("reference_role")=="anchor" and
                m.get("relative_change") is True and _finite_number(delta) and abs(delta)>=threshold)
            if not new:reasons.append("typed_joint_state_needs_visible_relative_change_from_real_reference")
        elif name=="lifted_sufficient_evidence":
            ref=support_reference_cert or {};rise=atom.get("vertical_rise_from_reference_m")
            direction=_vec(atom.get("projected_vertical_direction_px"))
            d=m.get("actual_relative_delta_px");projected=m.get("projected_relative_delta_px")
            axis_ok=(direction is not None and _norm(direction)>0 and d is not None and projected is not None and
                atom.get("vertical_component_identifiable") is True and
                abs(sum(x*y for x,y in zip(d,direction))/_norm(direction))>=FROZEN_CONTRACT.minimum_motion_px and
                abs(sum(x*y for x,y in zip(d,direction))/_norm(direction))>m.get("signal_floor_px",math.inf))
            new=(atom.get("entity_kind")=="rigid" and m.get("reference_role") in ("anchor","support") and axis_ok and
                _finite_number(rise) and rise>=PHYSICS_CONTRACT.vertical_rise_m and
                _positive_event_certificate(ref,"target_support_contact",identity,start) and
                any(v["interfaces"]["support"]["value"] is True and v["interfaces"]["support"]["contact_value"] is False for v in current["views"].values()) and
                _positive_event_certificate(carried_event_cert,"carried_sufficient_evidence",identity,step))
            if not new:reasons.append("lift_requires_real_support_reference_carried_and_identifiable_vertical_clearance")
        elif name=="released_sufficient_evidence":
            prior=prior_event_cert or {};opening=atom.get("actual_proprio_opening_m")
            window=list(range(max(1,step-PHYSICS_CONTRACT.release_frames+1),step+1))
            new=(len(window)==PHYSICS_CONTRACT.release_frames and m.get("relative_change") is True and
                m.get("reference_role")=="eef" and
                _positive_event_certificate(prior,"carried_sufficient_evidence",identity) and prior["step"]<window[0] and
                _finite_number(opening) and opening>=PHYSICS_CONTRACT.release_opening_min_m and atom.get("opening_source")=="actual_proprio" and
                atom.get("actual_proprio_opening_start_step")==prior["step"] and
                atom.get("actual_proprio_opening_end_step")==step and
                all(any(frames[s]["views"][view]["interfaces"][side]["value"] is True and frames[s]["views"][view]["interfaces"][side]["contact_value"] is False for side in ("left","right")) for s in window))
            if not new:reasons.append("release_requires_prior_certified_carrying_visible_separation_and_actual_opening")
    before=before_certificate or {}
    before_mask=(before.get("schema")==SCHEMA and before.get("name")==name and
        before.get("role")=="offline_supervision_only" and before.get("deployment_allowed") is False and
        before.get("new_mask") is True and before.get("measurement_valid") is True and
        type(before.get("physical_value")) is bool and before.get("identity") is not None and
        _identity(before["identity"])==identity and type(before.get("step")) is int and 1<=before["step"]<step)
    return dict(schema=SCHEMA,identity=dict(atom["identity"]),name=name,step=step,physical_value=value,
        measurement_valid=atom["measurement_valid"],old_mask=old,new_mask=bool(new),before_mask=before_mask,
        after_mask=bool(new),transition_mask=bool(new and before_mask),reasons=reasons,
        role="offline_supervision_only",deployment_allowed=False,training_ready=False,
        future_actual_frames_used_only_as_labels=True)


# V1 remains callable for reproduction. V2 corrects a coordinate assumption,
# not a numeric tolerance: projected motions of surfaces at different depths
# need not be equal while the corresponding rigid bodies move together in 3D.
SCHEMA_V2 = "recovery_observability_certificate_v2"


def certify_frame_v2(frame_evidence, *, contract=FROZEN_CONTRACT):
    """Literal contact heads require the corresponding finger's OWN identity.

    Base frame schema stays V1 for immutable frame/motion compatibility. The
    decision version is recorded explicitly. EEF union identity can support
    carrying motion, but is never borrowed by an unseen left or right surface.
    """
    result = certify_frame(frame_evidence,contract=contract)
    result["observation_decision_version"] = SCHEMA_V2
    for view_name, view in result["views"].items():
        raw = frame_evidence.get("views",{}).get(view_name,{})
        fingers = {side:_role(raw.get("independent_finger_roles",{}).get(side),
            view["registered_actual"],contract) for side in ("left","right")}
        view["independent_finger_roles"] = fingers
        for side in ("left","right"):
            bound_roles = dict(view["roles"],eef=fingers[side])
            metrics = raw.get("interfaces",{}).get(side)
            if isinstance(metrics,Mapping) and metrics.get("physics_contact") is False:
                noise,gap = metrics.get("same_candidate_repeat_noise_px"),metrics.get("actual_surface_gap_px")
                if (metrics.get("own_finger_side") != side or metrics.get("own_finger_identity_certified") is not True or
                    metrics.get("actual_target_boundary_verified") is not True or
                    metrics.get("actual_finger_boundary_verified") is not True or not _finite_number(noise) or noise < 0 or
                    not _finite_number(gap) or gap < contract.minimum_motion_px or gap <= contract.repeat_noise_multiplier*noise):
                    metrics = None
            view["interfaces"][side] = _interface(metrics,
                bound_roles,view["registered_actual"],contract)
            view["interfaces"][side]["specific_finger_identity_required"] = side
    return result


def certify_temporal_motion_v2(frame_certificates, motion_evidence, *, reference_role="eef", contract=FROZEN_CONTRACT):
    """Own-surface actual RGB versus projection, never a 2D rigidity test.

    The V1 per-surface fit, identity, camera compensation and same-candidate
    repeat gates are retained. ``pixel_delta_equality_diagnostic`` is explicitly
    not evidence for carrying: perspective and rotation break that invariant.
    A projected vector without actual tracks still cannot enable supervision.
    """
    result = certify_temporal_motion(frame_certificates, motion_evidence,
        reference_role=reference_role, contract=contract)
    result = dict(result, schema=SCHEMA_V2, inherits=(SCHEMA,),
        pixel_delta_equality_diagnostic=result.get("common_motion"),
        pixel_delta_equality_required_for_carrying=False,
        physical_common_motion_certified=False,
        reference_kind="cumulative_from_real_step2" if motion_evidence.get("start_step") == 2 and
            motion_evidence.get("end_step", 0) > 4 else "short_real_frame_window")
    if result.get("value") is True:
        actual_reference = _vec(motion_evidence.get("actual_reference_delta_px"))
        noise = motion_evidence["repeat_noise_px"]
        result["reference_motion"] = (_norm(actual_reference) >= contract.minimum_motion_px and
            _norm(actual_reference) > contract.repeat_noise_multiplier * noise)
        result["independent_surfaces_motion_supported"] = (result.get("subject_motion") is True and
            result["reference_motion"] is True)
        result["per_surface_actual_projection_fit_threshold_px"] = contract.maximum_motion_fit_error_px
    return result


def _vec3(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_finite_number(x) for x in value):
        return None
    return list(value)


def _carry_physics_and_proprio(atom, start, end, identity):
    """Recheck frozen sufficient evidence; measured aperture is not closure.

    No absolute aperture threshold is appropriate across object widths. Actual
    qpos/aperture and EEF provenance verify measurement lineage, not grasp truth.
    Bilateral contact + sustained 3D common motion remain OFFLINE teacher support.
    """
    rows = atom.get("physical_carry_window")
    hashes = atom.get("evidence_sha256", {})
    if not isinstance(rows, list) or len(rows) != PHYSICS_CONTRACT.sustained_frames or start < 2:
        return False, "frozen_three_frame_physical_window_missing", {}
    if not hashes.get("actual_proprio"):
        return False, "source_hashed_actual_proprio_missing", {}
    objects, eefs = [], []
    for step, row in zip(range(start, end + 1), rows):
        if not isinstance(row, Mapping) or row.get("step") != step:
            return False, "physical_window_time_alignment_missing", {}
        if row.get("identity") is None or _identity(row["identity"]) != identity:
            return False, "physical_window_source_identity_mismatch", {}
        p = row.get("physics", {})
        obj, eef = _vec3(p.get("target_pos_m")), _vec3(p.get("eef_pos_m"))
        if obj is None or eef is None or any(p.get(k) is not True for k in
                ("left_finger_target_contact", "right_finger_target_contact")) or p.get("target_support_contact") is not False:
            return False, "sustained_offline_physical_teacher_support_missing", {}
        pr = row.get("actual_proprio", {})
        qpos = pr.get("gripper_qpos_m")
        measured_eef = _vec3(pr.get("eef_pos_m"))
        aperture = p.get("gripper_aperture_m")
        alignment = pr.get("replay_proprio_max_abs")
        if (pr.get("source") != "original_hashed_proprio_npy" or pr.get("sha256") != hashes["actual_proprio"] or
            pr.get("frame") != step or measured_eef is None or not isinstance(qpos, (list, tuple)) or len(qpos) != 2 or
            not all(_finite_number(x) for x in qpos) or not _finite_number(aperture) or aperture < 0 or
            not _finite_number(alignment) or not 0 <= alignment <= FROZEN_CONTRACT.proprio_max_abs):
            return False, "actual_proprio_window_unverified", {}
        # Sum(abs(two qpos)) inherits twice the frozen per-coordinate bound.
        if max(abs(a-b) for a,b in zip(measured_eef,eef)) > FROZEN_CONTRACT.proprio_max_abs or \
                abs(sum(abs(x) for x in qpos)-aperture) > 2*FROZEN_CONTRACT.proprio_max_abs:
            return False, "actual_proprio_and_physical_measurement_disagree", {}
        objects.append(obj); eefs.append(eef)
    initial_relative = _delta(objects[0], eefs[0])
    drift = max(_norm(_delta(_delta(a,b),initial_relative)) for a,b in zip(objects,eefs))
    obj_motion, eef_motion = _norm(_delta(objects[-1], objects[0])), _norm(_delta(eefs[-1], eefs[0]))
    valid = (min(obj_motion,eef_motion) >= PHYSICS_CONTRACT.common_motion_min_m and
        drift <= PHYSICS_CONTRACT.relative_drift_max_m)
    return valid, "frozen_3d_common_motion_or_relative_drift_failed" if not valid else None, dict(
        target_motion_m=obj_motion,eef_motion_m=eef_motion,relative_drift_max_m=drift,
        actual_aperture_is_measurement_not_grasp_classifier=True,
        absolute_aperture_closure_threshold_used=False)


def certify_supervision_v2(atom, frame_certificates, *, motion_cert=None, prior_event_cert=None,
                          support_reference_cert=None, carried_event_cert=None, before_certificate=None):
    """Conservative probabilistic silver support for a frozen physical event.

    Carrying no longer requires observing literal finger interfaces in each
    frame. Its OWN sufficient certificate requires persistent actual-RGB
    instance tracks, own-surface projection fits, frozen 3D teacher support and
    original measured proprio. This never enables an unseen contact head.
    Labels can use execution-after-query evidence; deployment inputs cannot.
    """
    _forbidden(atom)
    legacy_motion = dict(motion_cert or {})
    if legacy_motion.get("schema") == SCHEMA_V2:
        legacy_motion["schema"] = SCHEMA
    legacy_before = dict(before_certificate or {})
    if legacy_before.get("schema") == SCHEMA_V2:
        legacy_before["schema"] = SCHEMA
    def legacy_event(value):
        event = dict(value or {})
        if event.get("schema") == SCHEMA_V2:
            event["schema"] = SCHEMA
        return event
    result = certify_supervision(atom, frame_certificates, motion_cert=legacy_motion,
        prior_event_cert=legacy_event(prior_event_cert), support_reference_cert=legacy_event(support_reference_cert),
        carried_event_cert=legacy_event(carried_event_cert), before_certificate=legacy_before)
    result.update(schema=SCHEMA_V2, inherits=(SCHEMA,),
        supervision_basis="literal_visible_interface_or_typed_relative_observation",
        algorithmic_silver_not_human_annotation=True,
        physical_values_and_old_masks_unchanged=True)
    if atom.get("name") != "carried_sufficient_evidence":
        if atom.get("name") == "joint_state_changed":
            result["joint_reference_step"] = atom.get("physical_reference_step")
            result["cumulative_reference_does_not_backfill_frame0"] = True
            m = motion_cert or {}
            if (m.get("schema") != SCHEMA_V2 or type(m.get("start_step")) is not int or
                    m["start_step"] < 2 or atom.get("physical_reference_step") != m["start_step"]):
                result.update(new_mask=False,after_mask=False,transition_mask=False)
                result["reasons"].append("v2_joint_reference_requires_aligned_real_step2_or_later")
        return result
    # Do not silently fall back to V1 carry certification if its new necessary
    # provenance fields are missing; every V2 carry certificate checks them.
    result.update(new_mask=False, after_mask=False, transition_mask=False)
    identity, step = _identity(atom["identity"]), atom["step"]
    m = motion_cert or {}; start = m.get("start_step")
    frames = _frame_index(frame_certificates)
    end_ok = (m.get("schema") == SCHEMA_V2 and m.get("value") is True and m.get("end_step") == step and
        type(start) is int and start >= 2 and start == step-PHYSICS_CONTRACT.sustained_frames+1 and
        m.get("used_steps") == list(range(start,step+1)) and m.get("reference_role") == "eef" and
        m.get("source") == "offline_actual_rgb_proprio_certificate" and m.get("deployment_allowed") is False and
        m.get("identity") is not None and _identity(m["identity"]) == identity and
        m.get("independent_surfaces_motion_supported") is True)
    physical_ok = (atom.get("entity_kind") == "rigid" and atom.get("value") is True and
        atom.get("measurement_valid") is True and _hashes(atom) and "physical_measurement" in atom["evidence_sha256"])
    reasons = []
    metrics = {}
    if not physical_ok:
        reasons.append("frozen_physical_positive_and_evidence_hash_required")
    elif not end_ok:
        reasons.append("persistent_actual_rgb_own_surface_motion_missing")
    else:
        view = m.get("view")
        identities_ok = view in ("primary","wrist") and all(s in frames and
            _identity(frames[s]["identity"]) == identity and frames[s]["reference_certified"] and
            all(frames[s]["views"].get(view,{}).get("roles",{}).get(r,{}).get("identity_certified") is True
                for r in ("target","eef")) for s in range(start,step+1))
        valid, reason, metrics = _carry_physics_and_proprio(atom,start,step,identity)
        if not identities_ok:
            reasons.append("actual_rgb_persistent_identity_unverified")
        elif not valid:
            reasons.append(reason)
        else:
            result.update(new_mask=True,after_mask=True,transition_mask=bool(result["before_mask"]))
    result.update(reasons=reasons, frozen_physical_window_metrics=metrics,
        supervision_basis="actual_RGB_persistent_own_surface_motion_plus_offline_3d_teacher_and_measured_proprio",
        carrying_silver_not_literal_contact_truth=True,
        unseen_contact_heads_remain_independently_masked=True,
        pixel_delta_equality_required=False)
    return result

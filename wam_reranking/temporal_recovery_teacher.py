"""CPU-only offline measurement teacher for time-aligned recovery evidence.

This module NEVER creates deployment features. Simulator truth and deployable
observability masks are separate. A sampled contact is not a grasp. Positive
carrying evidence is a conservative sufficient condition, NOT a complete grasp
detector; absence of that evidence is unknown, not a negative grasp label.
All events are detected causally from frames up to their reported time. Contacts
are at control boundaries, not proof of uninterrupted substep contact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

SCHEMA = "temporal_recovery_measurements_v1"


@dataclass(frozen=True)
class MeasurementContract:
    expected_frames: int = 17
    sustained_frames: int = 3
    release_frames: int = 2
    joint_change_rad: float = 0.01
    joint_change_m: float = 0.002
    vertical_rise_m: float = 0.01
    common_motion_min_m: float = 0.005
    relative_drift_max_m: float = 0.003
    release_opening_min_m: float = 0.004
    release_separation_min_m: float = 0.003


FROZEN_CONTRACT = MeasurementContract()
CONTACT_NAMES = (
    "left_finger_target_contact", "right_finger_target_contact",
    "target_support_contact",
)
OBSERVABILITY_NAMES = (
    "target_visible", "eef_visible", "primary_reliable", "wrist_reliable",
    "identity_consistent", "cross_view_consistent", "contact_observable",
    "target_motion_observable", "joint_state_observable",
)
FORBIDDEN_KEYS = {
    "attribution", "attribution_prediction", "class_probs", "factor_probs",
    "success", "final_success", "terminal_success", "predicted_class",
    "predicted_cause", "cause", "intervention", "condition",
}


def _no_forbidden(value: Any, path: str = "frames") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_KEYS or str(key).lower().startswith("predicted_"):
                raise ValueError(f"Forbidden label input: {path}.{key}")
            _no_forbidden(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _no_forbidden(item, f"{path}[{index}]")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real finite measurement")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real finite measurement") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: Any, name: str) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a three-vector in metres")
    try:
        seq = list(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a three-vector in metres") from error
    if len(seq) != 3:
        raise ValueError(f"{name} must be a three-vector in metres")
    return [_number(x, name) for x in seq]


def _optional_bool(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool or None, not a score or command")
    return value


def _minus(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _evidence(value: Any, *, known: bool, mask: bool, reason: str = "") -> dict:
    return dict(value=value, measurement_valid=known, supervision_mask=bool(known and mask),
                source="offline_simulator_measurement", unavailable_reason=reason)


def _visual_mask(observations: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    for obs in observations:
        if obs.get("target_visible") is not True or obs.get("eef_visible") is not True:
            return False, "target_or_eef_not_reliably_observed"
        if obs.get("identity_consistent") is not True:
            return False, "instance_identity_unverified"
        if not (obs.get("primary_reliable") is True or obs.get("wrist_reliable") is True):
            return False, "no_reliable_camera"
        # Single-view evidence is allowed; two views both claimed reliable need
        # explicit consistency certification, not equal image coordinates.
        if obs.get("primary_reliable") is True and obs.get("wrist_reliable") is True:
            if obs.get("cross_view_consistent") is not True:
                return False, "cross_view_consistency_unverified_or_conflicting"
    return True, ""


def derive_temporal_recovery(
    frames: Sequence[Mapping[str, Any]], *, entity_kind: str = "rigid",
    joint_unit: str | None = None,
    contract: MeasurementContract = FROZEN_CONTRACT,
) -> dict:
    """Return causal, time-stamped atoms and conservative physical evidence.

    Input frames are indices 0..16. Each has ``physics.eef_pos_m`` and optional
    ``target_pos_m``, ``gripper_aperture_m``, the three CONTACT_NAMES booleans,
    ``target_joint_qpos``. ``observability`` flags are external certifications;
    missing flags do not default to reliable. Body/entity and contact-role
    binding must be audited by the capture adapter. Articulated joint units are
    explicit (rad or m), and cabinet body motion cannot certify lifting.

    ``carried_sufficient_evidence`` uses a three-frame window: persistent dual
    finger contact, no recorded surface support, both bodies moving >=5 mm,
    common translation and <=3 mm relative drift. This can establish positive
    carrying support, never a
    universal grasp-negative. Release additionally needs previous carrying
    support, two sampled no-contact frames, >=4 mm opening and >=3 mm relative
    separation. Physics truth can be valid while its learning mask is false.
    """
    if entity_kind not in ("rigid", "articulated"):
        raise ValueError("entity_kind must be rigid or articulated")
    if entity_kind == "articulated" and joint_unit not in ("rad", "m"):
        raise ValueError("Articulated joints require explicit rad or m units")
    if entity_kind == "rigid" and joint_unit is not None:
        raise ValueError("Rigid entity cannot use joint-state supervision")
    if contract != FROZEN_CONTRACT:
        raise ValueError("Frozen measurement thresholds require a separately versioned contract to change")
    if len(frames) != contract.expected_frames:
        raise ValueError("Exactly 17 raw actual frames are required; no interpolation")
    _no_forbidden(frames)
    clean = []
    for index, frame in enumerate(frames):
        if frame.get("step") != index or isinstance(frame.get("step"), bool):
            raise ValueError("Frames must have unique, contiguous indices 0..16")
        physics = frame.get("physics")
        obs = frame.get("observability", {})
        if not isinstance(physics, Mapping) or not isinstance(obs, Mapping):
            raise ValueError("physics and observability must be mappings")
        p = dict(eef_pos_m=_vector(physics.get("eef_pos_m"), "eef_pos_m"))
        for name in ("target_pos_m",):
            p[name] = None if physics.get(name) is None else _vector(physics[name], name)
        for name in ("gripper_aperture_m", "target_joint_qpos"):
            p[name] = None if physics.get(name) is None else _number(physics[name], name)
        if p["gripper_aperture_m"] is not None and p["gripper_aperture_m"] < 0:
            raise ValueError("Gripper aperture is measured nonnegative distance, not a command")
        for name in CONTACT_NAMES:
            p[name] = _optional_bool(physics.get(name), name)
        for name in ("qpos", "qvel"):
            if name in physics:
                for x in physics[name]:
                    _number(x, name)
        o = {name: _optional_bool(obs.get(name), name) for name in OBSERVABILITY_NAMES}
        clean.append(dict(step=index, physics=p, observability=o))
    output, first_events, transitions = [], {}, []
    prior_carried = None
    first = clean[0]["physics"]
    for step, frame in enumerate(clean):
        p, obs = frame["physics"], frame["observability"]
        mask, mask_reason = _visual_mask([obs])
        atoms = {}
        for name in CONTACT_NAMES:
            value = p[name]
            available = value is not None
            atoms[name] = _evidence(value, known=available,
                mask=mask and obs["contact_observable"] is True,
                reason="" if available and mask and obs["contact_observable"] is True else
                    ("contact_role_not_recorded" if not available else "contact_not_visually_certified"))
        if p["target_pos_m"] is not None and first["target_pos_m"] is not None:
            displacement = _minus(p["target_pos_m"], first["target_pos_m"])
            relative = _minus(p["target_pos_m"], p["eef_pos_m"])
            endpoint_mask, endpoint_reason = _visual_mask([clean[0]["observability"], obs])
            if not (clean[0]["observability"]["target_motion_observable"] is True and obs["target_motion_observable"] is True):
                endpoint_mask = False
                endpoint_reason = "target_motion_observability_unverified"
            atoms["target_displacement_m"] = _evidence(displacement, known=True, mask=endpoint_mask, reason=endpoint_reason)
            atoms["target_relative_to_eef_m"] = _evidence(relative, known=True, mask=mask, reason=mask_reason)
            atoms["target_vertical_displacement_m"] = _evidence(displacement[2], known=True, mask=endpoint_mask, reason=endpoint_reason)
            is_rigid = entity_kind == "rigid"
            atoms["target_vertical_rise"] = _evidence(
                displacement[2] >= contract.vertical_rise_m if is_rigid else None,
                known=is_rigid, mask=endpoint_mask,
                reason=endpoint_reason if is_rigid else "articulated_body_height_is_not_object_lift")
        else:
            for name in ("target_displacement_m", "target_relative_to_eef_m",
                         "target_vertical_displacement_m", "target_vertical_rise"):
                atoms[name] = _evidence(None, known=False, mask=False, reason="target_position_not_recorded")
        if entity_kind == "articulated":
            available = p["target_joint_qpos"] is not None and first["target_joint_qpos"] is not None
            delta = p["target_joint_qpos"] - first["target_joint_qpos"] if available else None
            threshold = contract.joint_change_rad if joint_unit == "rad" else contract.joint_change_m
            endpoint_mask, endpoint_reason = _visual_mask([clean[0]["observability"], obs])
            if not (clean[0]["observability"]["joint_state_observable"] is True and obs["joint_state_observable"] is True):
                endpoint_mask = False
                endpoint_reason = "joint_state_observability_unverified"
            atoms["joint_displacement"] = _evidence(delta, known=available, mask=endpoint_mask,
                reason=endpoint_reason if available else "joint_measurement_not_recorded")
            atoms["joint_state_changed"] = _evidence(abs(delta) >= threshold if available else None,
                known=available, mask=endpoint_mask, reason=endpoint_reason if available else "joint_measurement_not_recorded")
        # No direct contact -> grasp equivalence. Only emit positive sufficient
        # support, or an explicit measured release following such support.
        carried = None
        carried_reason = "sustained_common_motion_not_established"
        window = clean[max(0, step-contract.sustained_frames+1):step+1]
        window_mask, window_reason = _visual_mask([x["observability"] for x in window])
        if not all(x["observability"]["contact_observable"] is True and
                   x["observability"]["target_motion_observable"] is True for x in window):
            window_mask = False
            window_reason = "contact_or_common_motion_not_observably_certified"
        if entity_kind == "articulated":
            carried_reason = "articulated_entity_has_no_whole_body_carry_label"
        elif len(window) == contract.sustained_frames and all(
            x["physics"]["target_pos_m"] is not None and
            x["physics"]["left_finger_target_contact"] is True and
            x["physics"]["right_finger_target_contact"] is True and
            x["physics"]["target_support_contact"] is False for x in window):
            start = window[0]["physics"]
            eef_motion = _minus(p["eef_pos_m"], start["eef_pos_m"])
            obj_motion = _minus(p["target_pos_m"], start["target_pos_m"])
            initial_relative = _minus(start["target_pos_m"], start["eef_pos_m"])
            drift = max(_norm(_minus(_minus(x["physics"]["target_pos_m"], x["physics"]["eef_pos_m"]), initial_relative)) for x in window)
            if min(_norm(eef_motion), _norm(obj_motion)) >= contract.common_motion_min_m and drift <= contract.relative_drift_max_m:
                carried = True
                carried_reason = window_reason
                prior_carried = dict(step=step, aperture=p["gripper_aperture_m"],
                    relative=_minus(p["target_pos_m"], p["eef_pos_m"]), supervision_mask=window_mask)
        atoms["carried_sufficient_evidence"] = _evidence(carried, known=carried is not None,
            mask=window_mask, reason=carried_reason)
        release = None
        release_reason = "no_prior_sufficient_carrying_evidence"
        release_window = clean[max(0, step-contract.release_frames+1):step+1]
        release_mask, release_mask_reason = _visual_mask([x["observability"] for x in release_window])
        if not all(x["observability"]["contact_observable"] is True and
                   x["observability"]["target_motion_observable"] is True for x in release_window):
            release_mask = False
            release_mask_reason = "contact_loss_or_relative_separation_not_observably_certified"
        if entity_kind == "rigid" and prior_carried is not None and step > prior_carried["step"]:
            if not prior_carried["supervision_mask"]:
                release_mask = False
                release_mask_reason = "prior_carrying_event_not_observable"
            release_reason = "release_opening_contact_loss_or_separation_unverified"
            no_contact = len(release_window) == contract.release_frames and all(
                x["physics"]["left_finger_target_contact"] is False and
                x["physics"]["right_finger_target_contact"] is False for x in release_window)
            position_known = p["target_pos_m"] is not None
            opening_known = p["gripper_aperture_m"] is not None and prior_carried["aperture"] is not None
            if no_contact and position_known and opening_known:
                opening = p["gripper_aperture_m"] - prior_carried["aperture"]
                separation = _norm(_minus(_minus(p["target_pos_m"], p["eef_pos_m"]), prior_carried["relative"]))
                if opening >= contract.release_opening_min_m and separation >= contract.release_separation_min_m:
                    release = True
                    release_reason = release_mask_reason
        atoms["released_sufficient_evidence"] = _evidence(release, known=release is not None,
            mask=release_mask, reason=release_reason)
        # A lift-positive additionally needs leaving a recorded initial support.
        # Starting already off-support leaves this unlabelled, not falsely low.
        lift = None
        if entity_kind == "rigid" and carried is True and first["target_support_contact"] is True and p["target_support_contact"] is False:
            if atoms["target_vertical_rise"]["value"] is True:
                lift = True
        initial_support_mask = (output[0]["physics"]["target_support_contact"]["supervision_mask"]
                                if step else atoms["target_support_contact"]["supervision_mask"])
        lift_mask = window_mask and mask and atoms["target_vertical_rise"]["supervision_mask"] and initial_support_mask
        atoms["lifted_sufficient_evidence"] = _evidence(lift, known=lift is not None,
            mask=lift_mask,
            reason="" if lift is True and lift_mask else
                "support_clearance_carrying_and_initial_reference_require_joint_certification")
        atoms["holding_after_measured_release"] = _evidence(
            False if release is True else None, known=release is True,
            mask=release_mask, reason=release_mask_reason if release is True else "no_current_certified_release")
        for name, atom in atoms.items():
            if atom["value"] is True and atom["measurement_valid"] and name not in first_events:
                first_events[name] = step
        if step:
            before = output[-1]
            for name in CONTACT_NAMES + (("joint_state_changed",) if entity_kind == "articulated" else ()):
                a, b = before["physics"][name], atoms[name]
                if a["measurement_valid"] and b["measurement_valid"] and a["value"] != b["value"]:
                    transitions.append(dict(step=step, name=name, event="entered" if b["value"] else "left",
                        supervision_mask=a["supervision_mask"] and b["supervision_mask"], source="offline_simulator_measurement"))
            for name in OBSERVABILITY_NAMES:
                a, b = before["observability"][name], obs[name]
                if a is not b and b is not None:
                    event = "established" if a is None else ("recovered" if b else "degraded")
                    transitions.append(dict(step=step, name=name, event=event,
                        supervision_mask=True, source="external_observability_certification"))
        output.append(dict(step=step, physics=atoms, observability=obs))
    return dict(schema=SCHEMA, contract=asdict(contract), entity_kind=entity_kind,
        joint_unit=joint_unit, frames=output, first_measured_events=first_events,
        temporal_transitions=transitions,
        limitations=["Control-boundary contact samples do not prove substep contact continuity.",
            "Sufficient carrying/release evidence is conservative, not complete grasp truth.",
            "Uncertified or invisible physical evidence is masked for deployable learning.",
            "No final outcome, predicted attribution or endpoint backfill is used.",
            "Entity IDs/contact geometry roles and SI units require an audited capture adapter."],
        deployment_features_created=False, training_started=False)

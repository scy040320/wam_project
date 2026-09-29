"""Auditable action-chunk semantics used before learned effect prediction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .contracts import CandidateEffect, CandidateVisualEvidence, Stage, TriValue


def candidate_effect_record(effect: CandidateEffect) -> dict[str, object]:
    """Project one candidate into the stable, auditable effect contract.

    The legacy flat ``evidence`` mapping remains available for compatibility.
    This structured view is used by evaluation and paper artifacts so that
    target motion, relation change, grasp/release and trajectory risk cannot
    be silently collapsed into one opaque score.
    """
    ev = effect.evidence

    def number(name: str, default: float = 0.0) -> float:
        value = float(ev.get(name, default))
        if not np.isfinite(value):
            raise ValueError(f"candidate effect field {name} must be finite")
        return value

    return {
        "candidate_id": int(effect.candidate_id),
        "stage": effect.stage.value,
        "confidence": float(effect.confidence),
        "target_displacement": {
            "dx": number("target_delta_x"),
            "dy": number("target_delta_y"),
            "magnitude": number("target_displacement"),
            "anchor_dx": number("anchor_delta_x"),
            "anchor_dy": number("anchor_delta_y"),
        },
        "target_anchor_relation": {
            "score_before": number("relation_score_before"),
            "score_after": number("relation_score_after"),
            "score_delta": number("relation_score_delta"),
            "distance_delta": number("target_anchor_distance_delta"),
            "affinity_delta": number("target_anchor_affinity_delta"),
            "progress": number("relation_progress"),
            "confidence": number("relation_confidence"),
            "cross_view_agreement": number("cross_view_agreement"),
        },
        "grasp_release": {
            "grasp_support_before": number("grasp_support_before"),
            "grasp_support_after": number("grasp_support_after"),
            "grasp_support_delta": number("grasp_support_delta"),
            "predicted_grasp_support": number("predicted_grasp_support"),
            "predicted_release_support": number("predicted_release_support"),
            "close_strength": number("close_strength"),
            "open_strength": number("open_strength"),
            "contact_confidence": number("contact_confidence"),
        },
        "trajectory": {
            "path_length": number("trajectory_path_length"),
            "net_displacement": number("trajectory_net_displacement"),
            "path_efficiency": number("trajectory_path_efficiency"),
            "max_step": number("trajectory_max_step"),
            "max_acceleration": number("trajectory_max_acceleration"),
            "max_jerk": number("trajectory_max_jerk"),
            "cumulative_excursion": number("trajectory_cumulative_excursion"),
            "gripper_transitions": number("trajectory_gripper_transitions"),
            "risk": number("trajectory_risk"),
        },
        "required_facts": dict(effect.required_facts),
        "proposed_effects": {
            name: {"value": value.value, "confidence": float(confidence)}
            for name, (value, confidence) in effect.proposed_effects.items()
        },
        "hard_violations": dict(effect.hard_violations),
    }


def _centroid(array: np.ndarray) -> tuple[float, float, float]:
    """Return normalized x/y centroid and localization contrast quality."""
    array = np.clip(np.asarray(array, dtype=np.float64), 0.0, 1.0)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("candidate relevance maps must be finite aligned 2-D arrays")
    height, width = array.shape
    mass = float(array.sum())
    if mass <= 1e-8:
        return 0.5, 0.5, 0.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    x = float((array * xx).sum() / mass / max(width - 1, 1))
    y = float((array * yy).sum() / mass / max(height - 1, 1))
    q50, q90, q99 = np.quantile(array, (0.50, 0.90, 0.99))
    contrast = float(max(0.0, q99 - q50) / max(float(q99), 0.05))
    tail = float(max(0.0, q99 - q90) / max(float(q99), 0.05))
    quality = float(np.clip(0.75 * contrast + 0.25 * tail, 0.0, 1.0))
    return x, y, quality


def _relation_score(subject: np.ndarray, anchor: np.ndarray, relation: str) -> float:
    """Auditable image-space satisfaction proxy for one typed relation."""
    sx, sy, _ = _centroid(subject)
    ax, ay, _ = _centroid(anchor)
    distance, affinity, _ = _centroid_geometry(subject, anchor)
    relation = str(relation).lower()
    if relation == "right_of":
        return float(np.clip(0.5 + 2.0 * (sx - ax), 0.0, 1.0))
    if relation == "left_of":
        return float(np.clip(0.5 + 2.0 * (ax - sx), 0.0, 1.0))
    if relation == "in_front_of":
        return float(np.clip(0.5 + 2.0 * (sy - ay), 0.0, 1.0))
    if relation == "under":
        return float(np.clip(0.5 + 2.0 * (sy - ay), 0.0, 1.0))
    if relation in {"on", "inside"}:
        return float(np.clip(0.45 * (1.0 - min(distance, 1.0)) + 0.55 * affinity, 0.0, 1.0))
    if relation in {"open", "closed", "articulated"}:
        # Articulated state is represented by subject displacement relative to
        # the stable frame. Direction is task-specific, so only separation is
        # asserted here; the task binding supplies open/closed semantics.
        return float(np.clip(distance, 0.0, 1.0))
    return float(np.clip(1.0 - min(distance, 1.0), 0.0, 1.0))


def _centroid_geometry(subject: np.ndarray, anchor: np.ndarray) -> tuple[float, float, float]:
    """Return distance, soft affinity and minimum evidence quality."""
    subject = np.clip(np.asarray(subject, dtype=np.float64), 0.0, 1.0)
    anchor = np.clip(np.asarray(anchor, dtype=np.float64), 0.0, 1.0)
    if subject.ndim != 2 or anchor.shape != subject.shape:
        raise ValueError("candidate relevance maps must be aligned 2-D arrays")
    if not np.isfinite(subject).all() or not np.isfinite(anchor).all():
        raise ValueError("candidate relevance maps must be finite")
    height, width = subject.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    smass, amass = float(subject.sum()), float(anchor.sum())
    if smass <= 1e-8 or amass <= 1e-8:
        return 1.0, 0.0, 0.0
    sx, sy = float((subject * xx).sum() / smass), float((subject * yy).sum() / smass)
    ax, ay = float((anchor * xx).sum() / amass), float((anchor * yy).sum() / amass)
    distance = float(np.hypot((sx - ax) / max(width - 1, 1), (sy - ay) / max(height - 1, 1)))
    affinity = float(np.minimum(subject, anchor).sum() / max(np.maximum(subject, anchor).sum(), 1e-8))
    def map_quality(array: np.ndarray) -> float:
        q50, q90, q99 = np.quantile(array, (0.50, 0.90, 0.99))
        # CLIPSeg logits are not calibrated probabilities.  Reliability is
        # therefore based on localization contrast, not a task-specific raw
        # score threshold.  Constant/missing evidence remains zero.
        contrast = float(max(0.0, q99 - q50) / max(float(q99), 0.05))
        tail = float(max(0.0, q99 - q90) / max(float(q99), 0.05))
        return float(np.clip(0.75 * contrast + 0.25 * tail, 0.0, 1.0))
    quality = min(map_quality(subject), map_quality(anchor))
    return distance, affinity, float(np.clip(quality, 0.0, 1.0))


def build_candidate_visual_evidence(
    *,
    current_subject_maps: Sequence[np.ndarray],
    predicted_subject_maps: Sequence[np.ndarray],
    current_anchor_maps: Sequence[np.ndarray],
    predicted_anchor_maps: Sequence[np.ndarray],
    current_gripper_maps: Sequence[np.ndarray],
    predicted_gripper_maps: Sequence[np.ndarray],
    relation: str,
) -> CandidateVisualEvidence:
    """Fuse primary/wrist relevance maps into candidate-level effect evidence.

    Inputs are produced by a frozen language-conditioned RGB localizer.  No
    simulator segmentation or object pose is accepted by this interface.
    """
    groups = tuple(map(tuple, (
        current_subject_maps, predicted_subject_maps,
        current_anchor_maps, predicted_anchor_maps,
        current_gripper_maps, predicted_gripper_maps,
    )))
    if not groups[0] or len({len(group) for group in groups}) != 1:
        raise ValueError("all candidate evidence views must have equal non-zero length")
    per_view = []
    for cs, ps, ca, pa, cg, pg in zip(*groups, strict=True):
        before_d, before_a, q1 = _centroid_geometry(cs, ca)
        after_d, after_a, q2 = _centroid_geometry(ps, pa)
        grip_before_d, grip_before_a, q3 = _centroid_geometry(cs, cg)
        grip_after_d, grip_after_a, q4 = _centroid_geometry(ps, pg)
        subject_d, _, qs = _centroid_geometry(cs, ps)
        anchor_d, _, qa = _centroid_geometry(ca, pa)
        csx, csy, _ = _centroid(cs); psx, psy, _ = _centroid(ps)
        cax, cay, _ = _centroid(ca); pax, pay, _ = _centroid(pa)
        relation_before = _relation_score(cs, ca, relation)
        relation_after = _relation_score(ps, pa, relation)
        grasp_before = float(np.clip(max(grip_before_a, 1.0 - 2.0 * grip_before_d), 0.0, 1.0))
        grasp_after = float(np.clip(max(grip_after_a, 1.0 - 2.0 * grip_after_d), 0.0, 1.0))
        relation_name = str(relation).lower()
        if relation_name in {"right_of", "left_of", "under", "in_front_of"}:
            progress = relation_after - relation_before
        elif relation_name in {"on", "inside"}:
            progress = (before_d - after_d) + (after_a - before_a)
        elif relation_name in {"open", "closed", "articulated"}:
            progress = subject_d - 0.5 * anchor_d
        else:
            progress = before_d - after_d
        relation_quality = min(q1, q2, qs, qa)
        contact_quality = min(q3, q4, qs)
        per_view.append(np.asarray([
            subject_d, anchor_d, before_d, after_d, before_a, after_a,
            grip_before_d, grip_after_d, grip_before_a, grip_after_a,
            progress, relation_quality, contact_quality,
            psx-csx, psy-csy, pax-cax, pay-cay,
            relation_before, relation_after, grasp_before, grasp_after,
        ], dtype=np.float64))
    stacked = np.stack(per_view)
    relation_weights = stacked[:, 11] + 1e-8
    contact_weights = stacked[:, 12] + 1e-8
    relation_mean = np.average(stacked, axis=0, weights=relation_weights)
    contact_mean = np.average(stacked, axis=0, weights=contact_weights)
    merged = relation_mean.copy()
    merged[6:10] = contact_mean[6:10]
    merged[19:21] = contact_mean[19:21]
    relation_confidence = float(np.max(stacked[:, 11]))
    contact_confidence = float(np.max(stacked[:, 12]))
    reliable_relation_views = stacked[stacked[:, 11] >= 0.35, 10]
    if len(reliable_relation_views) >= 2:
        agreement = float(np.exp(-6.0 * float(np.std(reliable_relation_views))))
    elif len(reliable_relation_views) == 1:
        # One usable camera is still useful for ranking, but it is not
        # cross-view corroboration and must not trigger a hard rejection.
        agreement = 0.45
    else:
        agreement = 0.0
    visibility = max(relation_confidence, contact_confidence)
    return CandidateVisualEvidence(
        target_motion=float(merged[0]), anchor_motion=float(merged[1]),
        target_anchor_distance_before=float(merged[2]), target_anchor_distance_after=float(merged[3]),
        target_anchor_affinity_before=float(merged[4]), target_anchor_affinity_after=float(merged[5]),
        target_gripper_distance_before=float(merged[6]), target_gripper_distance_after=float(merged[7]),
        target_gripper_affinity_before=float(merged[8]), target_gripper_affinity_after=float(merged[9]),
        relation_progress=float(merged[10]), visibility_confidence=float(np.clip(visibility, 0, 1)),
        cross_view_agreement=agreement, relation_confidence=relation_confidence,
        contact_confidence=contact_confidence,
        target_delta_x=float(merged[13]), target_delta_y=float(merged[14]),
        anchor_delta_x=float(merged[15]), anchor_delta_y=float(merged[16]),
        relation_score_before=float(np.clip(merged[17], 0, 1)),
        relation_score_after=float(np.clip(merged[18], 0, 1)),
        grasp_support_before=float(np.clip(merged[19], 0, 1)),
        grasp_support_after=float(np.clip(merged[20], 0, 1)),
    )


def localize_candidate_visual_evidence(
    *,
    localizer: object,
    current_primary: object,
    current_wrist: object,
    predicted_primary: object,
    predicted_wrist: object,
    target_prompt: str,
    anchor_prompt: str,
    relation: str,
    gripper_prompt: str = "robot gripper",
) -> CandidateVisualEvidence:
    """Extract candidate evidence directly from current/predicted RGB pairs."""
    images = [current_primary, predicted_primary, current_wrist, predicted_wrist]
    subject = np.asarray(localizer.relevance(images, target_prompt), dtype=np.float32)
    anchor = np.asarray(localizer.relevance(images, anchor_prompt), dtype=np.float32)
    gripper = np.asarray(localizer.relevance(images, gripper_prompt), dtype=np.float32)
    if subject.shape[0] != 4 or anchor.shape[0] != 4 or gripper.shape[0] != 4:
        raise RuntimeError("candidate localizer must return four aligned relevance maps")
    return build_candidate_visual_evidence(
        current_subject_maps=(subject[0], subject[2]),
        predicted_subject_maps=(subject[1], subject[3]),
        current_anchor_maps=(anchor[0], anchor[2]),
        predicted_anchor_maps=(anchor[1], anchor[3]),
        current_gripper_maps=(gripper[0], gripper[2]),
        predicted_gripper_maps=(gripper[1], gripper[3]),
        relation=relation,
    )


def parse_candidate_effect(candidate_id: int, actions: np.ndarray, *, visual_support: float = 0.5,
                           visual_evidence: CandidateVisualEvidence | None = None,
                           close_when_negative: bool = True) -> CandidateEffect:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (16, 7) or not np.isfinite(actions).all():
        raise ValueError("candidate actions must be finite with shape (16,7)")
    xyz = actions[:, :3]
    gripper = actions[:, 6]
    movement = float(np.linalg.norm(np.abs(xyz).sum(axis=0)))
    net_xyz = xyz.sum(axis=0)
    close_signal = -gripper if close_when_negative else gripper
    close_strength = float(np.max(close_signal))
    open_strength = float(np.max(-close_signal))
    closed_fraction = float(np.mean(close_signal > 0.25))
    lateral = float(np.linalg.norm(net_xyz[:2]))
    first_close = int(np.argmax(close_signal > 0.25)) if np.any(close_signal > 0.25) else 16
    upward_after_close = float(max(0.0, xyz[first_close:, 2].sum())) if first_close < 16 else 0.0
    step_norm = np.linalg.norm(xyz, axis=1)
    path_length = float(step_norm.sum())
    net_displacement = float(np.linalg.norm(net_xyz))
    path_efficiency = float(np.clip(net_displacement / max(path_length, 1e-8), 0.0, 1.0))
    acceleration = np.diff(xyz, axis=0)
    jerk = np.diff(acceleration, axis=0)
    max_step = float(step_norm.max(initial=0.0))
    max_acceleration = float(np.linalg.norm(acceleration, axis=1).max(initial=0.0))
    max_jerk = float(np.linalg.norm(jerk, axis=1).max(initial=0.0))
    cumulative_excursion = float(np.linalg.norm(np.cumsum(xyz, axis=0), axis=1).max(initial=0.0))
    gripper_state = close_signal > 0.25
    gripper_transitions = int(np.count_nonzero(gripper_state[1:] != gripper_state[:-1]))
    closed_at_entry = bool(gripper_state[0])
    open_after_closed = np.flatnonzero((~gripper_state) & np.maximum.accumulate(gripper_state))
    release_index = int(open_after_closed[0]) if open_after_closed.size else 16
    closed_before_release = float(np.mean(gripper_state[:release_index])) if release_index else 0.0
    trajectory_risk = float(np.clip(
        0.30 * np.clip((max_step - 0.35) / 0.65, 0.0, 1.0)
        + 0.20 * np.clip((max_acceleration - 0.50) / 1.50, 0.0, 1.0)
        + 0.20 * np.clip((max_jerk - 0.75) / 2.25, 0.0, 1.0)
        + 0.20 * (1.0 - path_efficiency)
        + 0.10 * np.clip((gripper_transitions - 2) / 3.0, 0.0, 1.0),
        0.0, 1.0,
    ))
    if open_strength > 0.5 and closed_fraction > 0.2:
        stage = Stage.PLACE
    elif close_strength > 0.5 and upward_after_close > 0.08:
        stage = Stage.LIFT
    elif closed_fraction > 0.5 and lateral > 0.12:
        stage = Stage.TRANSPORT
    elif close_strength > 0.5:
        stage = Stage.GRASP
    elif movement > 0.08:
        stage = Stage.APPROACH
    else:
        stage = Stage.UNCERTAIN
    requirements = {
        Stage.APPROACH: {"target_visible": 0.5, "target_pose_current": 0.6},
        # A newly generated grasp chunk is itself the recovery action after an
        # execution mismatch; it must not require the previous chunk's
        # execution_consistent predicate. Pose and reachability remain hard.
        Stage.GRASP: {"target_pose_current": 0.8, "target_reachable": 0.8},
        # ``execution_consistent`` describes whether the *previous* action
        # block matched its request.  It is evidence for invalidating the old
        # grasp assumption, not a permanent prerequisite of a newly planned
        # lift.  The new lift must instead re-establish ``grasped`` from the
        # current observation/candidate trace before it is allowed through.
        Stage.LIFT: {"grasped": 0.85},
        Stage.TRANSPORT: {"grasped": 0.8, "lifted": 0.75},
        Stage.PLACE: {"lifted": 0.75, "receptacle_visible": 0.6, "place_ready": 0.7},
        Stage.UNCERTAIN: {}, Stage.OBSERVE: {},
    }
    effects = {
        Stage.APPROACH: {"target_reachable": (TriValue.TRUE, 0.55)},
        Stage.GRASP: {"grasped": (TriValue.TRUE, 0.55)},
        Stage.LIFT: {"lifted": (TriValue.TRUE, 0.55)},
        Stage.TRANSPORT: {"place_ready": (TriValue.TRUE, 0.45)},
        Stage.PLACE: {"placed": (TriValue.TRUE, 0.55)},
        Stage.UNCERTAIN: {}, Stage.OBSERVE: {},
    }
    support = float(visual_support if visual_evidence is None else visual_evidence.visibility_confidence)
    confidence = float(np.clip(0.55 + 0.25 * abs(close_strength - open_strength) + 0.2 * support, 0, 1))
    evidence = {
        "movement_l1": movement, "net_upward": float(max(0.0, net_xyz[2])),
        "net_lateral": lateral, "close_strength": close_strength, "open_strength": open_strength,
        "closed_fraction": closed_fraction, "visual_support": support,
        "trajectory_path_length": path_length,
        "trajectory_net_displacement": net_displacement,
        "trajectory_path_efficiency": path_efficiency,
        "trajectory_max_step": max_step,
        "trajectory_max_acceleration": max_acceleration,
        "trajectory_max_jerk": max_jerk,
        "trajectory_cumulative_excursion": cumulative_excursion,
        "trajectory_gripper_transitions": float(gripper_transitions),
        "trajectory_risk": trajectory_risk,
        # Temporal command evidence.  These fields distinguish an entry
        # prerequisite from a predicate established inside the same chunk.
        "closed_at_entry": float(closed_at_entry),
        "release_index": float(release_index),
        "closed_before_release": closed_before_release,
    }
    proposed_effects = dict(effects[stage])
    violations: dict[str, float] = {}
    if visual_evidence is not None:
        evidence.update({
            "target_motion": visual_evidence.target_motion,
            "anchor_motion": visual_evidence.anchor_motion,
            "target_anchor_distance_delta": (
                visual_evidence.target_anchor_distance_after
                - visual_evidence.target_anchor_distance_before
            ),
            "target_anchor_affinity_delta": (
                visual_evidence.target_anchor_affinity_after
                - visual_evidence.target_anchor_affinity_before
            ),
            "target_gripper_distance_after": visual_evidence.target_gripper_distance_after,
            "target_gripper_distance_before": visual_evidence.target_gripper_distance_before,
            "target_gripper_affinity_after": visual_evidence.target_gripper_affinity_after,
            "target_gripper_affinity_before": visual_evidence.target_gripper_affinity_before,
            "relation_progress": visual_evidence.relation_progress,
            "cross_view_agreement": visual_evidence.cross_view_agreement,
            "relation_confidence": visual_evidence.relation_confidence,
            "contact_confidence": visual_evidence.contact_confidence,
            "target_delta_x": visual_evidence.target_delta_x,
            "target_delta_y": visual_evidence.target_delta_y,
            "target_displacement": float(np.hypot(
                visual_evidence.target_delta_x, visual_evidence.target_delta_y
            )),
            "anchor_delta_x": visual_evidence.anchor_delta_x,
            "anchor_delta_y": visual_evidence.anchor_delta_y,
            "relation_score_before": visual_evidence.relation_score_before,
            "relation_score_after": visual_evidence.relation_score_after,
            "relation_score_delta": (
                visual_evidence.relation_score_after - visual_evidence.relation_score_before
            ),
            "grasp_support_before": visual_evidence.grasp_support_before,
            "grasp_support_after": visual_evidence.grasp_support_after,
            "grasp_support_delta": (
                visual_evidence.grasp_support_after - visual_evidence.grasp_support_before
            ),
        })
        relation_reliable = min(
            visual_evidence.relation_confidence, visual_evidence.cross_view_agreement
        )
        contact_reliable = min(
            visual_evidence.contact_confidence,
            max(visual_evidence.cross_view_agreement, 0.65 if visual_evidence.contact_confidence >= 0.55 else 0.0),
        )
        contact_support = max(
            visual_evidence.target_gripper_affinity_after,
            1.0 - min(1.0, 2.0 * visual_evidence.target_gripper_distance_after),
        )
        grasp_prediction = float(np.clip(
            close_strength * visual_evidence.contact_confidence
            * (0.5 * visual_evidence.grasp_support_after
               + 0.5 * max(0.0, visual_evidence.grasp_support_after
                           - visual_evidence.grasp_support_before)),
            0.0, 1.0,
        ))
        release_prediction = float(np.clip(
            open_strength * visual_evidence.relation_confidence
            * (0.5 * visual_evidence.relation_score_after
               + 0.5 * max(0.0, visual_evidence.grasp_support_before
                           - visual_evidence.grasp_support_after)),
            0.0, 1.0,
        ))
        evidence["predicted_grasp_support"] = grasp_prediction
        evidence["predicted_release_support"] = release_prediction
        current_contact_support = float(np.clip(max(
            visual_evidence.grasp_support_before,
            visual_evidence.target_gripper_affinity_before,
            1.0 - min(1.0, 2.0 * visual_evidence.target_gripper_distance_before),
        ), 0.0, 1.0))
        current_contact_reliability = float(min(
            visual_evidence.contact_confidence,
            max(
                visual_evidence.cross_view_agreement,
                0.65 if visual_evidence.contact_confidence >= 0.55 else 0.0,
            ),
        ))
        evidence["current_contact_support"] = current_contact_support
        evidence["current_contact_reliability"] = current_contact_reliability
        evidence["current_grasped_support"] = float(min(
            current_contact_support, current_contact_reliability
        ))
        carrying_sequence = (
            closed_at_entry
            and closed_before_release >= 0.75
            and stage in {Stage.TRANSPORT, Stage.PLACE}
        )
        evidence["current_lifted_support"] = float(
            min(current_contact_support, current_contact_reliability)
            if carrying_sequence else 0.0
        )
        relation_nonregressing = (
            visual_evidence.relation_score_after
            >= visual_evidence.relation_score_before - 0.02
        )
        establishes_place_ready = (
            stage is Stage.PLACE
            and release_index < 16
            and closed_before_release >= 0.75
            and relation_nonregressing
        )
        evidence["establishes_place_ready_before_release"] = float(establishes_place_ready)
        evidence["place_ready_support"] = float(
            min(
                visual_evidence.relation_confidence,
                max(
                    visual_evidence.cross_view_agreement,
                    0.65 if visual_evidence.relation_confidence >= 0.55 else 0.0,
                ),
            ) if establishes_place_ready else 0.0
        )
        if stage is Stage.GRASP:
            proposed_effects["grasped"] = (TriValue.TRUE, max(0.05, grasp_prediction))
        elif stage in {Stage.LIFT, Stage.TRANSPORT}:
            proposed_effects["lifted"] = (
                TriValue.TRUE,
                float(np.clip(contact_support * visual_evidence.contact_confidence, 0.05, 1.0)),
            )
        elif stage is Stage.PLACE:
            proposed_effects["placed"] = (TriValue.TRUE, max(0.05, release_prediction))
        if contact_reliable >= 0.55:
            if stage in {Stage.LIFT, Stage.TRANSPORT} and contact_support < 0.35:
                violations["predicted_target_gripper_contact_missing"] = contact_reliable * (1.0 - contact_support)
            grasp_relation_nonimprovement = stage is Stage.GRASP and (
                visual_evidence.target_gripper_distance_after
                >= visual_evidence.target_gripper_distance_before
                and visual_evidence.target_gripper_affinity_after
                <= visual_evidence.target_gripper_affinity_before
            )
            # Lack of predicted progress is not a demonstrated safety
            # violation: downstream replanning may still recover. Preserve it
            # as an auditable soft risk rather than rejecting the candidate.
            evidence["grasp_relation_nonimprovement_soft"] = float(
                contact_reliable if grasp_relation_nonimprovement else 0.0
            )
        distance_regression = (
            visual_evidence.target_anchor_distance_after
            - visual_evidence.target_anchor_distance_before
        )
        affinity_regression = (
            visual_evidence.target_anchor_affinity_before
            - visual_evidence.target_anchor_affinity_after
        )
        # A centroid moving slightly away is not sufficient evidence that a
        # placement regresses: perspective and partial occlusion often move a
        # centroid while overlap improves.  Hard rejection requires two
        # independent geometric signals, two-view support, and a material
        # margin.  We retain the weaker signal below as a soft risk feature.
        evidence["relation_regression_soft"] = float(np.clip(
            max(0.0, distance_regression) + max(0.0, affinity_regression), 0.0, 1.0
        ))
        if relation_reliable >= 0.75:
            if (
                stage is Stage.PLACE
                and visual_evidence.cross_view_agreement >= 0.75
                and distance_regression >= 0.08
                and affinity_regression >= 0.08
            ):
                violations["predicted_target_anchor_relation_regresses"] = relation_reliable
            if stage is Stage.PLACE and open_strength < 0.5:
                violations["predicted_release_not_commanded"] = relation_reliable
        confidence *= float(0.7 + 0.3 * max(relation_reliable, contact_reliable))
    return CandidateEffect(
        candidate_id, stage, requirements[stage], proposed_effects, confidence,
        evidence, violations,
    )

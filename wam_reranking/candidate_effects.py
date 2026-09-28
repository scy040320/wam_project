"""Auditable action-chunk semantics used before learned effect prediction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .contracts import CandidateEffect, CandidateVisualEvidence, Stage, TriValue


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
        relation_name = str(relation).lower()
        if relation_name == "right_of":
            progress = before_d - after_d
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
        ], dtype=np.float64))
    stacked = np.stack(per_view)
    relation_weights = stacked[:, 11] + 1e-8
    contact_weights = stacked[:, 12] + 1e-8
    relation_mean = np.average(stacked, axis=0, weights=relation_weights)
    contact_mean = np.average(stacked, axis=0, weights=contact_weights)
    merged = relation_mean.copy()
    merged[6:10] = contact_mean[6:10]
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
        *map(float, merged[:11]), float(np.clip(visibility, 0, 1)), agreement,
        relation_confidence, contact_confidence,
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
        Stage.GRASP: {"target_pose_current": 0.8, "target_reachable": 0.8, "execution_consistent": 0.6},
        Stage.LIFT: {"grasped": 0.85, "execution_consistent": 0.8},
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
    }
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
        if contact_reliable >= 0.55:
            if stage in {Stage.LIFT, Stage.TRANSPORT} and contact_support < 0.35:
                violations["predicted_target_gripper_contact_missing"] = contact_reliable * (1.0 - contact_support)
            if stage is Stage.GRASP and (
                visual_evidence.target_gripper_distance_after
                >= visual_evidence.target_gripper_distance_before
                and visual_evidence.target_gripper_affinity_after
                <= visual_evidence.target_gripper_affinity_before
            ):
                violations["predicted_grasp_relation_not_improved"] = contact_reliable
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
        candidate_id, stage, requirements[stage], effects[stage], confidence,
        evidence, violations,
    )

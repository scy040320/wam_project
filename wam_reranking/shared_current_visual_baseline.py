"""Candidate-invariant current geometry for a new forecast feature contract.

The historical parser is deliberately not replaced.  Its camera weights depend
on the forecast and consequently its BEFORE fields vary inside one pool.  This
additive contract fixes camera weights from CURRENT RGB relevance maps only.
All geometry remains an uncalibrated image-space proxy, not a physical fact.
"""
from dataclasses import asdict
import numpy as np

from .candidate_effects import _centroid_geometry, _relation_score, build_candidate_visual_evidence
from .contracts import CandidateVisualEvidence

SCHEMA = "shared_current_camera_fusion_v2_direct_CURRENT"
CONTACT_FIELDS = frozenset((
    "target_gripper_distance_before", "target_gripper_distance_after",
    "target_gripper_affinity_before", "target_gripper_affinity_after",
    "grasp_support_before", "grasp_support_after",
))
BEFORE_FIELDS = (
    "target_anchor_distance_before", "target_anchor_affinity_before",
    "target_gripper_distance_before", "target_gripper_affinity_before",
    "relation_score_before", "grasp_support_before",
)


def _maps(groups):
    arrays = tuple(tuple(np.array(v, dtype=float, copy=True) for v in g) for g in groups)
    if not arrays[0] or len({len(g) for g in arrays}) != 1:
        raise ValueError("Aligned nonempty CURRENT/forecast camera maps required")
    for view in zip(*arrays, strict=True):
        if any(v.ndim != 2 or not np.isfinite(v).all() for v in view):
            raise ValueError("Finite two-dimensional camera maps required")
        if len({v.shape for v in view}) != 1:
            raise ValueError("Same-camera CURRENT and forecast shapes must match")
    return arrays


def build_shared_current_visual_evidence(*, current_subject_maps,
        predicted_subject_maps, current_anchor_maps, predicted_anchor_maps,
        current_gripper_maps, predicted_gripper_maps, relation):
    """Return forecast evidence plus auditable CURRENT-only fusion weights.

    No execution outcome, simulator role map or physical state is an argument.
    Callers must separately authenticate the localizer and actual RGB sources.
    The returned diagnostics do not authenticate map provenance themselves.
    """
    groups = _maps((current_subject_maps, predicted_subject_maps,
        current_anchor_maps, predicted_anchor_maps,
        current_gripper_maps, predicted_gripper_maps))
    per_view, current_before, relation_quality, contact_quality = [], [], [], []
    for cs, ps, ca, pa, cg, pg in zip(*groups, strict=True):
        # CURRENT geometry must never pass through the historical builder's
        # single-view np.average(q*x/q).  Its q depends on FORECAST quality and
        # the mathematical cancellation can leave forecast-dependent ULPs.
        # Derive the six CURRENT-only fields directly, once for this camera.
        relation_distance, relation_affinity, relation_q = _centroid_geometry(cs, ca)
        contact_distance, contact_affinity, contact_q = _centroid_geometry(cs, cg)
        current_before.append(dict(
            target_anchor_distance_before=relation_distance,
            target_anchor_affinity_before=relation_affinity,
            target_gripper_distance_before=contact_distance,
            target_gripper_affinity_before=contact_affinity,
            relation_score_before=_relation_score(cs, ca, relation),
            grasp_support_before=float(np.clip(
                max(contact_affinity, 1.0 - 2.0 * contact_distance), 0.0, 1.0))))
        relation_quality.append(relation_q)
        contact_quality.append(contact_q)
        ev = build_candidate_visual_evidence(current_subject_maps=(cs,),
            predicted_subject_maps=(ps,), current_anchor_maps=(ca,),
            predicted_anchor_maps=(pa,), current_gripper_maps=(cg,),
            predicted_gripper_maps=(pg,), relation=relation)
        per_view.append(asdict(ev))
    # Same epsilon as the historical fusion, but never forecast-conditioned.
    rw, cw = np.array(relation_quality) + 1e-8, np.array(contact_quality) + 1e-8
    rw, cw = rw / rw.sum(), cw / cw.sum()
    merged = {}
    for name in per_view[0]:
        if name in BEFORE_FIELDS:
            continue
        weight = cw if name in CONTACT_FIELDS else rw
        merged[name] = float(weight @ np.array([ev[name] for ev in per_view]))
    for name in BEFORE_FIELDS:
        weight = cw if name in CONTACT_FIELDS else rw
        merged[name] = float(weight @ np.array([before[name] for before in current_before]))
    # Quality remains candidate-specific; it is not a CURRENT certificate.
    merged["relation_confidence"] = max(ev["relation_confidence"] for ev in per_view)
    merged["contact_confidence"] = max(ev["contact_confidence"] for ev in per_view)
    merged["visibility_confidence"] = max(merged["relation_confidence"], merged["contact_confidence"])
    progress = [ev["relation_progress"] for ev in per_view if ev["relation_confidence"] >= .35]
    merged["cross_view_agreement"] = (float(np.exp(-6 * np.std(progress)))
        if len(progress) >= 2 else .45 if progress else 0.)
    evidence = CandidateVisualEvidence(**merged)
    return evidence, dict(schema=SCHEMA, relation_weights=rw.tolist(),
        contact_weights=cw.tolist(), current_relation_quality=relation_quality,
        current_contact_quality=contact_quality,
        before={name: merged[name] for name in BEFORE_FIELDS},
        before_weights_use_forecasts=False, physical_certificate=False,
        before_uses_historical_candidate_fusion=False,
        historical_feature_schema_reinterpreted=False)


def audit_shared_current_pool(evidences, diagnostics):
    if not evidences or len(evidences) != len(diagnostics):
        raise ValueError("Aligned nonempty candidate pool required")
    first = diagnostics[0]
    for ev, diagnostic in zip(evidences, diagnostics, strict=True):
        if diagnostic.get("schema") != SCHEMA:
            raise ValueError("Explicit new fusion schema required")
        for key in ("relation_weights", "contact_weights", "before"):
            if diagnostic[key] != first[key]:
                raise ValueError("Candidates do not share an identical CURRENT baseline")
        if any(getattr(ev, name) != first["before"][name] for name in BEFORE_FIELDS):
            raise ValueError("Evidence and CURRENT baseline disagree")
    return dict(passed=True, candidates=len(evidences),
        before_candidate_invariant=True, current_physical_facts_generated=0,
        old_cached_before_may_be_relabelled=False)

"""Auxiliary observation-availability labels, never physical recovery truth.

Input is offline actual-image certificate data. It is Y-only; no observation
made after executing a probe can be a preexecution candidate feature. A failed
image correspondence means this observer lacks usable evidence, not that the
target is absent. A single recovered view cannot certify resolved dual-view
conflict, grasp, placement, or current target pose.
"""
from __future__ import annotations

import math

SCHEMA = "bounded_reobservation_availability_silver_v1"


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _view_available(frame, view, helpers):
    record = frame["observation_evidence"]["views"][view]
    target = record["roles"]["target"]
    registration = record["global_rgb"]
    local = target["local_rgb"]
    pixels = target["rendered_pixels"]
    # Registration and content thresholds inherit the frozen RGB adapter.
    if type(pixels) is not int or pixels < 0:
        raise ValueError("Literal target pixel count required")
    for metric in (registration, local):
        if type(metric.get("passed")) is not bool:
            raise ValueError("Explicit actual-image registration required")
    if (not registration["passed"] or not helpers.rgb_pass(registration)
            or record.get("fresh_geometry_registered_to_actual_input") is not True
            or pixels < 16 or not local["passed"] or not helpers.rgb_pass(local)):
        return False
    role_noise = record.get("repeat_noise", {}).get("target", {})
    if (not _finite(role_noise.get("centroid_noise_px")) or role_noise["centroid_noise_px"] < 0
            or role_noise.get("rgb_repeat", {}).get("passed") is not True
            or not helpers.rgb_pass(role_noise["rgb_repeat"])):
        return False
    # Independent actual RGB tracks; a segmentation-only witness is not enough.
    for interval in frame["observation_evidence"].get("actual_rgb_temporal_intervals", []):
        if interval["end_step"] != frame["step"] or interval["view"] != view:
            continue
        if interval["start_step"] < 1 or interval["start_step"] >= frame["step"]:
            raise ValueError("Noncausal observation interval")
        tracked = interval["tracks"]["target"]
        noise = interval["same_candidate_repeat_noise"]["target"]
        summary = helpers.track_summary(tracked)
        role = helpers.current_role(summary, target)
        if (_finite(noise) and noise >= 0 and summary is not None
                and role["identity_rgb_unique"] is True):
            return True
    return False


def derive_reobservation_availability(frames, *, identity, arm, certificate_helpers):
    """Same-time observer availability only; no conversion to task predicates.

    The before reference is real frame2 (a no-motion settling prefix); cached
    query frame0 is retained in the trace but cannot retroactively certify it.
    Even an explicit peek need not recover evidence. All negative results stay
    in the denominator. No stage/outcome or simulator physical atom is read.
    """
    if arm not in ("hold", "peek_left", "peek_right", "peek_up"):
        raise ValueError("Unknown predeclared probe arm")
    if len(frames) != 17 or [f.get("step") for f in frames] != list(range(17)):
        raise ValueError("17 aligned frames are required")
    if not all(callable(getattr(certificate_helpers, k, None))
               for k in ("rgb_pass", "track_summary", "current_role")):
        raise ValueError("Frozen certificate helpers required")
    for f in frames:
        qc = f["observation_evidence"]["same_candidate_repeat_qc"]
        if (qc.get("physics_passed") is not True or qc.get("all_contact_atoms_match") is not True
                or not _finite(qc.get("qpos_qvel_max_abs")) or not 0 <= qc["qpos_qvel_max_abs"] <= 1e-3
                or not _finite(qc.get("qpos_qvel_p95")) or not 0 <= qc["qpos_qvel_p95"] <= 1e-4):
            raise ValueError("Same-arm repeat QC is required")
    rows = []
    for f in frames:
        if f["step"] < 2:
            rows.append(dict(step=f["step"], value=None, mask=False,
                             reason="cached_or_unmatched_initial_reference"))
            continue
        views = {v: _view_available(f, v, certificate_helpers) for v in ("primary", "wrist")}
        rows.append(dict(step=f["step"], value=any(views.values()), mask=True,
                         views=views, target_physical_absence_inferred=False))
    before, after = rows[2], rows[16]
    return dict(schema=SCHEMA, role="auxiliary_supervision_only", identity=identity,
                arm=arm, frames=rows, reference_step=2,
                observed_availability_recovered=before["value"] is False and after["value"] is True,
                # No semantic cross-view identity agreement certificate exists
                # in this finite probe; do not invent one from view reliability.
                cross_view_conflict_resolved=dict(value=None, mask=False,
                    reason="separate_cross_view_semantic_agreement_certificate_required"),
                task_predicate_mapping_admitted=False, ranking_training_ready=False,
                no_after_execution_observation_in_preexecution_x=True)

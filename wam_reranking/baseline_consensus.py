"""Candidate-specific evidence is required to override a learned baseline.

Experimental development repair, separate from the frozen source-scoped
policy. This does not guarantee zero success harm. Pixel contact and view
scores are comparative proxies, not physical safety certificates.
"""
from dataclasses import replace

import numpy as np

from .candidate_utility import select_with_utility
from .evidence_residual import select_evidence_residual


def contact_view_support(features):
    """Two comparable deployable axes, with release semantics for place.

    Feature indices follow the frozen candidate-backbone contract. Contact
    support is a soft endpoint prediction, including for approach/articulated
    actions; no successful grasp is inferred from this comparison.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.shape != (27,) or not np.isfinite(x).all():
        raise ValueError('invalid frozen backbone vector')
    contact = float(x[13] if x[20] == 1. else x[6])
    quality = min(float(x[5]), float(x[8]))
    return contact, quality


def select_baseline_consensus(*, decisions, backbone_features, cause_features,
                              backbone, residual):
    """Change only unresolved-but-physically-consistent value routing.

    Keep all witnessed FALSE and candidate hard violations, unreliable-view
    fallback, existing model weights and switch margins intact. No task,
    state, labels, outcomes or rollout costs enter this function.
    """
    accepted = [d for d in decisions if d.accepted]
    if not accepted:
        return select_evidence_residual(decisions=decisions,
            backbone_features=backbone_features, cause_features=cause_features,
            backbone=backbone, residual=residual)
    # Candidate-only control has zero belief terms. Use its exact selector,
    # then require its chosen candidate to survive the full physical gate.
    controls = [replace(d, accepted=True, rejection_reasons=(),
                        components={'dependency_risk': 0., 'uncertainty': 0.}) for d in decisions]
    anchor, _ = select_with_utility(controls, backbone_features, backbone)
    feasible = {d.candidate_id: d for d in accepted}
    value = max(accepted, key=lambda d: (d.official_value, -d.candidate_id))
    eligible = (anchor.candidate_id in feasible
                and all(d.components.get('unresolved_without_supported_physical_recovery', 0.) for d in accepted)
                and any(d.components.get('preserve_official_value', 0.) for d in accepted))
    if eligible and anchor.candidate_id != value.candidate_id:
        base = np.asarray(backbone_features[anchor.candidate_id])
        official = np.asarray(backbone_features[value.candidate_id])
        # Do not compare contact-after-grasp to release-after-place. Unknown
        # or incompatible action phases do not supply override evidence.
        same_phase = np.array_equal(base[16:21], official[16:21]) and bool(np.sum(base[16:21]) == 1)
        a = contact_view_support(official)
        b = contact_view_support(base)
        dominates = same_phase and all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))
        if not dominates:
            chosen = replace(feasible[anchor.candidate_id], components={
                **feasible[anchor.candidate_id].components,
                'candidate_baseline_retained_no_comparative_override_evidence': 1.})
            return chosen, {d.candidate_id: backbone.score(backbone_features[d.candidate_id]) for d in accepted}
    return select_evidence_residual(decisions=decisions,
        backbone_features=backbone_features, cause_features=cause_features,
        backbone=backbone, residual=residual)

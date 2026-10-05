"""Trusted positive evidence, not absence of evidence, permits a route change.

The frozen source-scoped selector remains the reference. This development
repair does not infer a physical cause from unknown, fit new thresholds, or
guarantee zero rollout harm. Contact/release and view scores remain proxies.
"""
from dataclasses import dataclass, replace

import numpy as np

from .candidate_utility import select_with_utility
from .contracts import CoarseCause, ConsistencyFactor, TriValue
from .evidence_residual import select_evidence_residual
from .source_scoped_policy import prepare_source_scoped_decisions


# Inherit the parser's existing contact-evidence threshold; do not calibrate
# this on failure cases. It is not a physical safety certificate.
MIN_CONTACT_EVIDENCE = 0.55


@dataclass(frozen=True)
class EvidenceRoute:
    visual_reliable: bool
    cause_unresolved: bool
    physically_consistent: bool
    attribution_available: bool

    @property
    def comparison_allowed(self):
        return (self.attribution_available and self.visual_reliable
                and self.cause_unresolved and self.physically_consistent)


def evidence_route(attribution):
    if attribution is None:
        return EvidenceRoute(False, True, False, False)
    quality = attribution.evidence_quality
    visual = (attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE)
              is TriValue.TRUE and quality.primary_reliable
              and quality.wrist_reliable and not quality.cross_view_conflict)
    unresolved = (attribution.projected_cause is CoarseCause.UNKNOWN
                  or attribution.factor_state(ConsistencyFactor.CAUSE_RESOLVED)
                  is not TriValue.TRUE)
    physical = all(attribution.factor_state(f) is TriValue.TRUE for f in (
        ConsistencyFactor.WORLD_STATE_CONSISTENT,
        ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT,
        ConsistencyFactor.TASK_STAGE_CONSISTENT))
    return EvidenceRoute(visual, unresolved, physical, True)


def annotate_evidence_routes(decisions, attribution):
    """Keep gates intact and make distinct routing reasons auditable."""
    route = evidence_route(attribution)
    return tuple(replace(d, components={**d.components,
        'visual_evidence_unreliable': float(not route.visual_reliable),
        'cause_unresolved': float(route.cause_unresolved),
        'trusted_candidate_comparison_allowed': float(route.comparison_allowed),
        'routing_attribution_available': float(route.attribution_available),
    }) for d in decisions)


def prepare_arbitrated_decisions(**kwargs):
    """Same frozen belief update and gates, only additional provenance."""
    decisions = prepare_source_scoped_decisions(**kwargs)
    return annotate_evidence_routes(decisions, kwargs['attribution'])


def _vector(features):
    x = np.asarray(features, dtype=np.float64)
    if x.shape != (27,) or not np.isfinite(x).all():
        raise ValueError('invalid frozen backbone vector')
    return x


def affirmative_improvement(candidate, reference):
    """Positive contact/release gain with no evidence-quality/risk tradeoff.

    A different phase, tied support, weaker view/contact evidence or higher
    trajectory risk is not comparable positive evidence. Model preference
    alone is insufficient. No outcomes or task identities are used.
    """
    a, b = _vector(candidate), _vector(reference)
    same_phase = (np.array_equal(a[16:21], b[16:21])
                  and np.count_nonzero(a[16:21]) == 1
                  and np.sum(a[16:21]) == 1.)
    if not same_phase:
        return False
    index = 13 if a[20] == 1. else 6
    qa, qb = min(a[5], a[8]), min(b[5], b[8])
    return bool(min(qa, qb) >= MIN_CONTACT_EVIDENCE
                and a[index] > b[index] and qa >= qb and a[15] <= b[15])


def select_evidence_arbitration(*, decisions, backbone_features,
                                cause_features, backbone, residual,
                                attribution=None):
    """Unreliable/conflicting/missing visual evidence cannot overturn V6.

    Only its unresolved-but-physically-consistent value route is eligible for
    a change. A candidate must survive the existing gate AND positively
    dominate the reference evidence, not merely avoid being dominated.
    """
    reference, scores = select_evidence_residual(decisions=decisions,
        backbone_features=backbone_features, cause_features=cause_features,
        backbone=backbone, residual=residual)
    tagged = annotate_evidence_routes(decisions, attribution)
    if reference is None:
        return None, scores
    feasible = {d.candidate_id: d for d in tagged if d.accepted}
    original = feasible[reference.candidate_id]
    route = evidence_route(attribution)
    eligible = (route.comparison_allowed
                and all(d.components.get('unresolved_without_supported_physical_recovery', 0.)
                        for d in feasible.values())
                and any(d.components.get('preserve_official_value', 0.)
                        for d in feasible.values()))
    if not eligible:
        return replace(original, components={**original.components,
                       'frozen_reference_route_preserved': 1.}), scores
    controls = [replace(d, accepted=True, rejection_reasons=(),
                        components={'dependency_risk': 0., 'uncertainty': 0.})
                for d in decisions]
    anchor, _ = select_with_utility(controls, backbone_features, backbone)
    if (anchor.candidate_id not in feasible
            or anchor.candidate_id == original.candidate_id
            or not affirmative_improvement(backbone_features[anchor.candidate_id],
                                           backbone_features[original.candidate_id])):
        return replace(original, components={**original.components,
                       'frozen_reference_route_preserved': 1.}), scores
    chosen = feasible[anchor.candidate_id]
    return replace(chosen, components={**chosen.components,
                   'trusted_positive_comparative_switch': 1.}), {
        d.candidate_id: backbone.score(backbone_features[d.candidate_id])
        for d in feasible.values()}

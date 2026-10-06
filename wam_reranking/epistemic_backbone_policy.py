"""Preserve a frozen ranker when cause uncertainty supplies no counterevidence.

This is a development repair, not evidence that attribution adds value.
Existing hard gates, observed contradictions and fallback are never bypassed.
Frozen historical arbitration remains a separate, reproducible policy.
"""
from dataclasses import replace

from .candidate_utility import select_with_utility
from .evidence_arbitration import evidence_route, select_evidence_arbitration


def preserve_unresolved_backbone(*, reference, scores, decisions, attribution,
                                backbone_anchor, backbone_scores):
    """Cause unresolved is not evidence against reliable candidate learning.

    A counterfactual anchor is supplied by the same frozen candidate-only
    selector at this observation. It may be used only if the actual hard gate
    accepts it. No task/state IDs or outcomes enter this function.
    """
    route = evidence_route(attribution)
    if reference is None or not route.comparison_allowed or backbone_anchor is None:
        return reference, scores
    feasible = {d.candidate_id: d for d in decisions if d.accepted}
    anchor = feasible.get(backbone_anchor.candidate_id)
    if anchor is None or anchor.rejection_reasons:
        return reference, scores
    if not all(d.components.get('unresolved_without_supported_physical_recovery', 0.)
               for d in feasible.values()):
        return reference, scores
    chosen = replace(anchor, components={**anchor.components,
        'unsupported_cause_preserves_backbone': 1.,
        'historical_value_route_superseded': float(anchor.candidate_id != reference.candidate_id),
        'cause_residual_gain_changed': 0.,
    })
    return chosen, {i: value for i, value in backbone_scores.items() if i in feasible}


def select_epistemic_backbone(**kwargs):
    """Frozen arbitration except reliable cause-unresolved baseline continuity."""
    reference, scores = select_evidence_arbitration(**kwargs)
    controls = [replace(d, accepted=True, rejection_reasons=(),
                        components={'dependency_risk': 0., 'uncertainty': 0.})
                for d in kwargs['decisions']]
    anchor, anchor_scores = select_with_utility(
        controls, kwargs['backbone_features'], kwargs['backbone'])
    return preserve_unresolved_backbone(reference=reference, scores=scores,
        decisions=kwargs['decisions'], attribution=kwargs.get('attribution'),
        backbone_anchor=anchor, backbone_scores=anchor_scores)

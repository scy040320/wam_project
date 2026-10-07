"""Additive utility correction AFTER the frozen complete V8 routing decision.

The complete V8 reference is select_evidence_arbitration, not the old stand-alone
safe residual ranker. Zero correction is an exact identity. No gate or route is
relaxed. No execution outcome enters this inference API.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .evidence_arbitration import (affirmative_improvement, evidence_route,
                                  select_evidence_arbitration)


def select_full_v8_conditioned(*, decisions, backbone_features, cause_features,
                               backbone, frozen_residual, attribution, deltas):
    reference, original_scores = select_evidence_arbitration(decisions=decisions,
        backbone_features=backbone_features,cause_features=cause_features,
        backbone=backbone,residual=frozen_residual,attribution=attribution)
    ids=[d.candidate_id for d in decisions]
    correction=np.asarray(deltas,dtype=float)
    if correction.shape!=(len(ids),) or not np.isfinite(correction).all():
        raise ValueError('Complete aligned finite conditional corrections required')
    if len(set(ids))!=len(ids):raise ValueError('Duplicate candidate identity')
    corrections=dict(zip(ids,correction))
    audit={'frozen_v8_candidate':None if reference is None else reference.candidate_id,
           'frozen_routing_preserved':True,'conditional_switch':False}
    # Do not simulate the reference with a different tie rule or Pareto veto.
    if reference is None or not np.any(correction):
        audit['reason']='zero_residual_or_original_fallback'
        return reference,original_scores,audit
    route=evidence_route(attribution)
    feasible=[d for d in decisions if d.accepted]
    if not route.visual_reliable:
        audit['reason']='frozen_unreliable_visual_route'
        return reference,original_scores,audit
    value_route=any(d.components.get('preserve_official_value',0.) for d in feasible)
    if value_route and not route.comparison_allowed:
        audit['reason']='frozen_conservative_value_route'
        return reference,original_scores,audit
    eligible=[reference]
    for d in feasible:
        if d.candidate_id==reference.candidate_id:continue
        if corrections[d.candidate_id]<=corrections[reference.candidate_id]+1e-12:continue
        # V8 allows a change in unresolved-but-physically-consistent scenes only
        # with positive trusted same-phase contact/release evidence. Keep that.
        if value_route and not affirmative_improvement(backbone_features[d.candidate_id],
                                                       backbone_features[reference.candidate_id]):
            continue
        eligible.append(d)
    scores={d.candidate_id:original_scores[d.candidate_id]+corrections[d.candidate_id]
            for d in feasible}
    margin=max(.01,float(backbone.switch_margin))
    top=max(scores[d.candidate_id] for d in eligible)
    tied=[d for d in eligible if top-scores[d.candidate_id]<=margin]
    chosen=max(tied,key=lambda d:(d.official_value,-d.candidate_id))
    if chosen.candidate_id!=reference.candidate_id:
        # Positive residual evidence must clear the inherited switch margin;
        # otherwise adding tiny corrections can silently undo route arbitration.
        if scores[chosen.candidate_id]-scores[reference.candidate_id]<=margin:
            chosen=reference
    audit.update(reason='frozen_route_plus_conditional_score',
        eligible_candidates=[d.candidate_id for d in eligible],
        conditional_switch=chosen.candidate_id!=reference.candidate_id)
    return replace(chosen,components={**chosen.components,
        'full_v8_conditional_residual':float(corrections[chosen.candidate_id])}),scores,audit

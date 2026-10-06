"""Conservative, pixel/command-only recovery silver labels.

Open-vocabulary map quality is NOT calibrated physical certainty. Labels are
explicit observer silver; grasp, lift, placement and exact pose stay masked.
No predicted frame, attribution output, simulator state or episode success is
accepted by this module.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from .contracts import PREDICATES
from .candidate_effects import _centroid


def _unknown(source='observed_image_support'):
    return dict(before='unknown',after='unknown',before_mask=False,after_mask=False,
        transition_mask=False,before_quality=0.,after_quality=0.,source=source,
        evidence_steps=[],evidence_ids=[])


def map_endpoint(relevance):
    array=np.asarray(relevance,dtype=np.float32)
    if array.ndim!=2 or array.size==0 or not np.isfinite(array).all():
        raise ValueError('Actual observer requires finite 2D relevance')
    # Inherits the existing frozen localization-contrast formula, not a
    # newly tuned confidence threshold. No absence->physical FALSE inference.
    x,y,q=_centroid(array)
    return dict(x=x,y=y,quality=q,usable=bool(q>=.5))


def _visibility(before,after):
    b=max(before,key=lambda d:d['quality']);a=max(after,key=lambda d:d['quality'])
    out=_unknown()
    out.update(before='true' if b['usable'] else 'unknown',after='true' if a['usable'] else 'unknown',
        before_mask=b['usable'],after_mask=a['usable'],transition_mask=b['usable'] and a['usable'],
        before_quality=b['quality'],after_quality=a['quality'],
        evidence_steps=[0,16],evidence_ids=['primary_sequence','wrist_sequence'])
    return out


def _availability(before,after):
    b=max(d['quality'] for d in before);a=max(d['quality'] for d in after)
    # Availability refers to this observer's usable image evidence, NOT object
    # existence. Observer insufficiency is reliably labelled as insufficiency.
    out=_unknown()
    out.update(before='true' if b>=.5 else 'false',after='true' if a>=.5 else 'false',
        before_mask=True,after_mask=True,transition_mask=True,before_quality=1.,after_quality=1.,
        evidence_steps=[0,16],evidence_ids=['primary_sequence','wrist_sequence'])
    return out


def labels_from_actual(*, identity, subject_maps, anchor_maps, requested, applied):
    """Maps order: before-primary, before-wrist, after-primary, after-wrist."""
    sm=np.asarray(subject_maps);am=np.asarray(anchor_maps)
    if sm.ndim!=3 or sm.shape[0]!=4 or am.shape!=sm.shape:
        raise ValueError('Actual endpoint map identity mismatch')
    r=np.asarray(requested);a=np.asarray(applied)
    if r.shape!=(16,7) or a.shape!=r.shape or not np.isfinite(r).all() or not np.isfinite(a).all():
        raise ValueError('Invalid actual command evidence')
    s=[map_endpoint(m) for m in sm];z=[map_endpoint(m) for m in am]
    labels={p:_unknown() for p in PREDICATES}
    labels['target_visible']=_visibility(s[:2],s[2:])
    labels['receptacle_visible']=_visibility(z[:2],z[2:])
    command=_unknown('measured_command_difference')
    # This supervision is a measured command channel only, not contact physics.
    command.update(after='true' if np.array_equal(r,a) else 'false',after_mask=True,
        after_quality=1.,evidence_steps=[16],evidence_ids=['requested_actions','applied_actions'])
    labels['command_consistent']=command
    scores=[];qualities=[]
    for k in (0,2):
        perview=[max(0.,1-float(np.hypot(s[j]['x']-z[j]['x'],s[j]['y']-z[j]['y'])))
                 for j in (k,k+1)]
        q=[min(s[j]['quality'],z[j]['quality']) for j in (k,k+1)]
        reliable=[j for j,v in enumerate(q) if v>=.5]
        scores.append(float(np.mean([perview[j] for j in reliable])) if reliable else None)
        qualities.append(min([q[j] for j in reliable]) if reliable else 0.)
    proxy=_unknown()
    proxy.update(before=scores[0],after=scores[1],before_mask=scores[0] is not None,
        after_mask=scores[1] is not None,transition_mask=all(v is not None for v in scores),
        before_quality=qualities[0],after_quality=qualities[1],evidence_steps=[0,16],
        evidence_ids=['primary_sequence','wrist_sequence'],units='image_space_proxy')
    return dict(schema='candidate_recovery_labels_v2',inherits='candidate_recovery_supervision_v1',
        role='supervision_only',identity=deepcopy(identity),predicates=labels,
        observability={'target_evidence_available':_availability(s[:2],s[2:]),
                       'receptacle_evidence_available':_availability(z[:2],z[2:])},
        relation_support_2d=proxy)

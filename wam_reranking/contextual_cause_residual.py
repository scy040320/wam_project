"""One bounded development repair: cause-specific full candidate context.

No task identity, outcome, intervention or simulator state is accepted by
the feature function. Frozen backbone and hard gates stay separate.
"""
from dataclasses import dataclass

import numpy as np

from .contracts import CoarseCause, ConsistencyFactor, TriValue

FEATURE_COUNT = 70  # Three cause routes x 20 non-value context terms + 10 legacy terms.


def contextual_cause_features(backbone_features, legacy_features, attribution):
    x=np.asarray(backbone_features,dtype=np.float64)
    z=np.asarray(legacy_features,dtype=np.float64)
    if x.shape!=(27,) or z.shape!=(10,) or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError('Invalid candidate context')
    if attribution is None or attribution.projected_cause in (CoarseCause.NORMAL,CoarseCause.UNKNOWN):
        return np.zeros(FEATURE_COUNT)
    quality=attribution.evidence_quality
    if quality.cross_view_conflict or attribution.factor_state(ConsistencyFactor.CAUSE_RESOLVED) is not TriValue.TRUE:
        return np.zeros(FEATURE_COUNT)
    cause=attribution.projected_cause
    visible=attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is TriValue.TRUE
    world=(1-attribution.factor(ConsistencyFactor.WORLD_STATE_CONSISTENT)) if visible and cause is CoarseCause.OBJECT_SHIFT else 0.
    execution=(1-attribution.factor(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT)) if visible and quality.execution_reliable and cause is CoarseCause.EXECUTION_CONTACT_DEVIATION else 0.
    occlusion=(1-attribution.factor(ConsistencyFactor.OBSERVATION_RELIABLE)) if cause is CoarseCause.VISUAL_OCCLUSION else 0.
    # No free value coefficient: the shared backbone already owns it.
    return np.concatenate([world*x[1:21],execution*x[1:21],occlusion*x[1:21],z])


@dataclass(frozen=True)
class ContextResidualModel:
    scale: np.ndarray
    weights: np.ndarray
    train_abs_bound: np.ndarray
    gain: float
    score_bound: float

    def score(self,features):
        features=np.asarray(features,dtype=np.float64)
        if features.shape!=(FEATURE_COUNT,) or not np.isfinite(features).all():
            raise ValueError('Invalid contextual residual')
        if np.any(np.abs(features)>self.train_abs_bound+1e-8):return 0.
        return float(np.clip(self.gain*((features/self.scale)@self.weights),-self.score_bound,self.score_bound))

    def to_record(self):
        return dict(schema='bounded_cause_context_residual_development_v1',
            feature_count=FEATURE_COUNT,scale=self.scale.tolist(),weights=self.weights.tolist(),
            train_abs_bound=self.train_abs_bound.tolist(),gain=self.gain,score_bound=self.score_bound)


def fit_context_residual(pairs,support):
    support=np.asarray(support,dtype=np.float64)
    if support.ndim!=2 or support.shape[1]!=FEATURE_COUNT or not np.isfinite(support).all():
        raise ValueError('Invalid context training support')
    scale=support.std(axis=0);scale[scale<1e-8]=1.
    weights=np.zeros(FEATURE_COUNT)
    if pairs:
        delta=np.stack([(a-b)/scale for a,b,_ in pairs])
        margin=np.asarray([m for _,_,m in pairs])
        for _ in range(2000):
            probability=1/(1+np.exp(-np.clip(margin+delta@weights,-30,30)))
            gradient=-((1-probability)[:,None]*delta).mean(axis=0)+weights
            weights-=.05*gradient
    scores=(support/scale)@weights
    return ContextResidualModel(scale,weights,np.max(np.abs(support),axis=0),1.,float(np.max(np.abs(scores))))

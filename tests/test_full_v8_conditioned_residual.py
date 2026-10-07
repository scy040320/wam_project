from dataclasses import replace
import unittest
import numpy as np

from wam_reranking.contracts import (AttributionOutput,CandidateDecision,CoarseCause,
    ConsistencyFactor,EvidenceQuality,TriValue)
from wam_reranking.candidate_utility import CandidateUtilityModel
from wam_reranking.evidence_residual import EvidenceResidualModel
from wam_reranking.evidence_arbitration import select_evidence_arbitration
from wam_reranking.full_v8_conditioned_residual import select_full_v8_conditioned


def attr(cause=CoarseCause.OBJECT_SHIFT,reliable=True):
    p={f.value:1. for f in ConsistencyFactor}
    if cause is CoarseCause.OBJECT_SHIFT:p['world_state_consistent']=0.
    if cause is CoarseCause.UNKNOWN:p['cause_resolved']=0.
    return AttributionOutput(p,{c.value:float(c is cause) for c in CoarseCause},cause,.9,0.,
        EvidenceQuality(reliable,reliable,True,not reliable),'before',
        {f.value:TriValue.TRUE if p[f.value] else TriValue.FALSE for f in ConsistencyFactor})


class FullV8Tests(unittest.TestCase):
    def fixtures(self):
        backbone=CandidateUtilityModel(np.zeros(27),np.ones(27),np.zeros(27),0.,switch_margin=.03)
        residual=EvidenceResidualModel(np.ones(10),np.zeros(10),np.ones(10),0.,.1)
        ds=[CandidateDecision(i,True,.6-i*.01,None,(),{}) for i in range(3)]
        xs={i:np.zeros(27) for i in range(3)}
        for i in range(3):xs[i][0]=ds[i].official_value;xs[i][16]=1.;xs[i][5]=xs[i][8]=.9
        return dict(decisions=ds,backbone_features=xs,cause_features={i:np.zeros(10) for i in range(3)},
            backbone=backbone,frozen_residual=residual,attribution=attr())

    def test_zero_is_exact_full_arbitration_not_old_ranker(self):
        k=self.fixtures()
        original={name:value for name,value in k.items() if name!='frozen_residual'}
        reference,scores=select_evidence_arbitration(**original,residual=k['frozen_residual'])
        chosen,new,audit=select_full_v8_conditioned(**k,deltas=np.zeros(3))
        self.assertEqual(reference,chosen);self.assertEqual(scores,new);self.assertFalse(audit['conditional_switch'])

    def test_unreliable_visual_keeps_original_value_route(self):
        k=self.fixtures();k['attribution']=attr(CoarseCause.UNKNOWN,False)
        k['decisions']=[replace(d,components={'preserve_official_value':1.}) for d in k['decisions']]
        c,_,a=select_full_v8_conditioned(**k,deltas=[-.1,.1,0])
        self.assertEqual(c.candidate_id,0);self.assertEqual(a['reason'],'frozen_unreliable_visual_route')

    def test_cannot_unlock_false_or_hard_violation(self):
        k=self.fixtures();k['decisions'][1]=replace(k['decisions'][1],accepted=False,rejection_reasons=('hard_false',))
        c,_,_=select_full_v8_conditioned(**k,deltas=[-.1,.1,0])
        self.assertNotEqual(c.candidate_id,1)

    def test_resolved_route_can_use_candidate_specific_correction(self):
        c,_,a=select_full_v8_conditioned(**self.fixtures(),deltas=[-.05,.05,0])
        self.assertEqual(c.candidate_id,1);self.assertTrue(a['conditional_switch'])

    def test_unresolved_needs_affirmative_evidence_not_just_score(self):
        k=self.fixtures();k['attribution']=attr(CoarseCause.UNKNOWN)
        k['decisions']=[replace(d,components={'preserve_official_value':1.,
            'unresolved_without_supported_physical_recovery':1.}) for d in k['decisions']]
        c,_,_=select_full_v8_conditioned(**k,deltas=[-.05,.05,0]);self.assertEqual(c.candidate_id,0)
        k['backbone_features'][1][6]=.9
        c,_,_=select_full_v8_conditioned(**k,deltas=[-.05,.05,0]);self.assertEqual(c.candidate_id,1)

    def test_alignment_and_nonfinite_fail_closed(self):
        for d in ([0,0],[0,np.nan,0]):
            with self.assertRaises(ValueError):select_full_v8_conditioned(**self.fixtures(),deltas=d)


if __name__=='__main__':unittest.main()

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from wam_reranking.belief import initial_belief
from wam_reranking.contracts import (AttributionOutput, CandidateDecision, CandidateEffect,
    CoarseCause, ConsistencyFactor, EvidenceQuality, Stage, TriValue)
from wam_reranking.terminal_conditioned_residual import (CAP, FEATURE_NAMES, TerminalResidual,
    conditioned_features, fit_terminal_pairs, select_residual)


def attr(cause=CoarseCause.OBJECT_SHIFT, reliable=True):
    probs={f.value:1. for f in ConsistencyFactor}
    if cause is CoarseCause.OBJECT_SHIFT:probs['world_state_consistent']=0.
    if cause is CoarseCause.EXECUTION_CONTACT_DEVIATION:probs['execution_contact_consistent']=0.
    if cause is CoarseCause.UNKNOWN:probs['cause_resolved']=0.
    return AttributionOutput(probs,{c.value:float(c is cause) for c in CoarseCause},cause,.9,0.,
        EvidenceQuality(reliable,reliable,True,not reliable),'actual_before',
        {f.value:TriValue.TRUE if probs[f.value] else TriValue.FALSE for f in ConsistencyFactor},
        {f.value:.9 for f in ConsistencyFactor}) if reliable else replace(attr(CoarseCause.UNKNOWN),
            evidence_quality=EvidenceQuality(False,False,True,True))


def effect(cid=0, support=.9):
    return CandidateEffect(cid,Stage.LIFT,{'grasped':.8},{'grasped':(TriValue.TRUE,support)},.9,
        {'cross_view_agreement':.9,'contact_confidence':.9,'grasp_support_after':support,
         'relation_confidence':.9,'relation_score_after':support,'relation_score_delta':support,
         'target_displacement':support,'target_gripper_distance_after':1-support,
         'trajectory_path_efficiency':support,'trajectory_risk':1-support})


class TerminalResidualTests(unittest.TestCase):
    def feature(self,a=None,e=None,no_dag=False):
        plan=np.zeros((16,7));plan[:,6]=1;plan[8:,2]=.1
        return conditioned_features(prior=initial_belief(0),attribution=attr() if a is None else a,
            effect=effect() if e is None else e,actions=plan,no_dag=no_dag)

    def test_candidates_receive_distinct_cause_belief_interactions(self):
        x,_=self.feature(e=effect(support=.9));y,_=self.feature(e=effect(support=.1))
        self.assertEqual(len(x),len(FEATURE_NAMES));self.assertTrue(np.any(x!=y))

    def test_no_dag_removes_descendant_credit_not_direct_shift(self):
        x,p=self.feature();y,q=self.feature(no_dag=True)
        self.assertTrue(np.any(x!=y));self.assertTrue(p['paths']['grasped'])
        self.assertFalse(q['paths']['grasped']);self.assertTrue(q['paths']['target_pose_current'])

    def test_normal_does_not_fabricate_recovery(self):
        x,_=self.feature(a=attr(CoarseCause.NORMAL));self.assertFalse(np.any(x))

    def test_unreliable_visual_has_zero_correction_input(self):
        x,p=self.feature(a=attr(reliable=False));self.assertFalse(np.any(x))
        self.assertFalse(p['sufficient_evidence'])

    def test_unknown_cannot_claim_endpoint_physical_recovery(self):
        x,_=self.feature(a=attr(CoarseCause.UNKNOWN))
        for i,n in enumerate(FEATURE_NAMES):
            if n.endswith('endpoint_forecast'):self.assertEqual(x[i],0)

    def test_initial_unknown_is_not_a_certified_fact(self):
        _,p=self.feature(a=attr(CoarseCause.NORMAL))
        self.assertEqual(p['needs']['grasped'],0.)

    def test_endpoint_never_updates_prior(self):
        b=initial_belief(0);before=b.snapshot()
        conditioned_features(prior=b,attribution=attr(),effect=effect(),actions=np.zeros((16,7)))
        self.assertEqual(before,b.snapshot())

    def test_zero_matches_frozen_tie_margin_and_pareto(self):
        d=[CandidateDecision(0,True,.8,None,(),{'dependency_risk':.1,'uncertainty':.1}),
           CandidateDecision(1,True,.7,None,(),{'dependency_risk':.2,'uncertainty':.2})]
        s,_=select_residual(d,[.8,1.],[0,0],.036)
        self.assertEqual(s.candidate_id,0)  # identical inherited Pareto guard

    def test_cannot_unlock_rejected_candidate(self):
        d=[CandidateDecision(0,True,.8,None,(),{}),CandidateDecision(1,False,.9,None,('hard_false',),{})]
        s,_=select_residual(d,[.8,.9],[-.1,.1],.036)
        self.assertEqual(s.candidate_id,0)

    def test_bounded_centered_and_oos_pool_backoff(self):
        m=TerminalResidual(np.ones(len(FEATURE_NAMES)),np.ones(len(FEATURE_NAMES))*100,
            np.ones(len(FEATURE_NAMES)),1.)
        x=np.zeros((4,len(FEATURE_NAMES)));x[0]=1
        delta=m.deltas(x);self.assertAlmostEqual(float(delta.sum()),0.)
        self.assertTrue(np.all(np.abs(delta)<=CAP));x[0,0]=2
        self.assertFalse(np.any(m.deltas(x)))

    def test_fit_rejects_validation(self):
        with self.assertRaises(ValueError):fit_terminal_pairs([{'split':'val'}])

    def test_fit_uses_terminal_success_and_cost_pairs(self):
        x=np.zeros((3,len(FEATURE_NAMES)));x[:,0]=[.1,.5,.8]
        p={'split':'train','x':x,'base_scores':np.array([.9,.8,.7]),'baseline_id':0,
            'decisions':[CandidateDecision(i,True,1-i*.1,None,(),{}) for i in range(3)],
            'y':[{'success':True,'steps':100,'calls':5},{'success':False,'steps':400,'calls':24},
                 {'success':True,'steps':80,'calls':4}]}
        m,r=fit_terminal_pairs([p],steps=20)
        self.assertEqual(r['success_pairs'],2);self.assertEqual(r['successful_cost_pairs'],1)
        self.assertFalse(r['fit_uses_validation']);self.assertFalse(m.record()['recovery_probability_head_trained'])

    def test_shuffle_donors_same_split_different_group(self):
        path=Path(__file__).resolve().parents[1]/'scripts/train_terminal_conditioned_residual.py'
        spec=importlib.util.spec_from_file_location('terminal_training_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(json.dumps({'count':np.int64(4),'passed':np.bool_(True)},default=module.json_scalar),
                         '{"count": 4, "passed": true}')
        pools=[{'split':s,'key':(0,i,0,c)} for s,ii in [('train',(0,1)),('val',(6,7))]
               for i in ii for c in ('clean','object_shift')]
        donors=module.donor_map(pools)
        for p in pools:
            dk=donors[p['key']];self.assertNotEqual(p['key'][:2],dk[:2])
            self.assertEqual(p['key'][1]<=5,dk[1]<=5);self.assertEqual(p['key'][2:],dk[2:])


if __name__=='__main__':unittest.main()

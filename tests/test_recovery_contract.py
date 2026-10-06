import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from wam_reranking.candidate_utility import CandidateUtilityModel

from wam_reranking.belief import DEFAULT_GRAPH, DependencyGraph, initial_belief
from wam_reranking.contracts import (
    AttributionOutput, BeliefFact, BeliefState, CandidateEffect, CoarseCause,
    ConsistencyFactor, EvidenceQuality, PREDICATES, Stage, TriValue,
)
from wam_reranking.evidence_residual import typed_gate_effect
from wam_reranking.recovery_contract import (
    SCHEMA, FEATURE_NAMES, CurrentFactEvidence, ForecastFactEvent,
    prepare_recovery_context, trace_candidate_recovery, require_recovery_training_ready,
    select_recovery_candidate,
)
from wam_reranking.recovery_supervision import audit_recovery_supervision


def attr(cause, *, anomalous=(), reliable=True):
    probs={f.value:1. for f in ConsistencyFactor}
    states={f.value:TriValue.TRUE for f in ConsistencyFactor}
    for factor in anomalous:
        probs[factor.value]=0.;states[factor.value]=TriValue.FALSE
    if not reliable:
        probs['observation_reliable']=0.;states['observation_reliable']=TriValue.FALSE
    if cause is CoarseCause.UNKNOWN:
        probs['cause_resolved']=0.;states['cause_resolved']=TriValue.FALSE
    return AttributionOutput(probs,{c.value:float(c is cause) for c in CoarseCause},
        cause,.9,0.,EvidenceQuality(reliable,True,True),'recorded_block',states,
        {f.value:.9 for f in ConsistencyFactor})


def prior():
    b=BeliefState(0,'target','anchor',Stage.LIFT)
    for name in PREDICATES:
        b.facts[name]=BeliefFact(TriValue.TRUE,.9,'observation',2,('actual_before:'+name,))
    return b


def effect(stage=Stage.LIFT):
    return CandidateEffect(0,stage,{'grasped':.85},{'grasped':(TriValue.TRUE,.9)},.9,{})


class RecoveryContractTests(unittest.TestCase):
    def context(self,a,b=None,**kwargs):
        return prepare_recovery_context(prior=prior() if b is None else b,
            attribution=a,block_index=3,**kwargs)

    def test_normal_maintains_observed_facts_without_forced_invalidation(self):
        ctx=self.context(attr(CoarseCause.NORMAL))
        self.assertTrue(all(f.value is TriValue.TRUE for f in ctx.belief.facts.values()))

    def test_occlusion_preserves_real_holding(self):
        ctx=self.context(attr(CoarseCause.VISUAL_OCCLUSION,reliable=False))
        self.assertIs(ctx.belief.facts['target_pose_current'].value,TriValue.UNKNOWN)
        self.assertIs(ctx.belief.facts['grasped'].value,TriValue.TRUE)
        t=trace_candidate_recovery(ctx,effect(),np.zeros((16,7)))
        self.assertTrue(np.any(t.features[:]))

    def test_shift_propagates_then_real_current_grasp_can_refresh(self):
        a=attr(CoarseCause.OBJECT_SHIFT,anomalous=(ConsistencyFactor.WORLD_STATE_CONSISTENT,))
        ctx=self.context(a)
        self.assertIs(ctx.belief.facts['grasped'].value,TriValue.UNKNOWN)
        repaired=self.context(a,observations=[CurrentFactEvidence('grasped',TriValue.TRUE,.95,'actual_wrist')])
        self.assertIs(repaired.belief.facts['grasped'].value,TriValue.TRUE)
        self.assertIs(repaired.belief.facts['target_pose_current'].value,TriValue.FALSE)

    def test_execution_and_world_coexist_despite_coarse_projection(self):
        a=attr(CoarseCause.OBJECT_SHIFT,anomalous=(ConsistencyFactor.WORLD_STATE_CONSISTENT,
                                               ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT))
        ctx=self.context(a);t=trace_candidate_recovery(ctx,effect(),np.zeros((16,7)))
        for route in ('object_shift','execution_contact_deviation'):
            idx=[i for i,n in enumerate(FEATURE_NAMES) if n.startswith(route+'.')]
            self.assertTrue(np.any(t.features[idx]))

    def test_execution_record_cannot_certify_grasp(self):
        with self.assertRaises(ValueError):
            self.context(attr(CoarseCause.NORMAL),observations=[CurrentFactEvidence(
                'grasped',TriValue.TRUE,1.,'commands_equal','measured_execution')])

    def test_unknown_is_epistemic_not_physical_false(self):
        ctx=self.context(attr(CoarseCause.UNKNOWN))
        self.assertIs(ctx.belief.facts['grasped'].value,TriValue.UNKNOWN)
        self.assertFalse(any(f.value is TriValue.FALSE for f in ctx.belief.facts.values()))
        t=trace_candidate_recovery(ctx,effect(),np.zeros((16,7)))
        self.assertTrue(any(t.features[i]>0 for i,n in enumerate(FEATURE_NAMES) if n.startswith('unknown.')))

    def test_no_initial_assumption_or_generic_reliability_certifies_fact(self):
        ctx=self.context(attr(CoarseCause.NORMAL),initial_belief(0))
        self.assertTrue(all(f.value is TriValue.UNKNOWN for f in ctx.belief.facts.values()))

    def test_candidate_prediction_is_not_current_observation(self):
        with self.assertRaises(ValueError):
            CurrentFactEvidence('grasped',TriValue.TRUE,.99,'future','candidate_prediction')
        b=prior();b.facts['grasped']=BeliefFact(TriValue.TRUE,.9,'candidate_prediction',2,('future',))
        ctx=self.context(attr(CoarseCause.NORMAL),b)
        self.assertIs(ctx.belief.facts['grasped'].value,TriValue.UNKNOWN)

    def test_terminal_grasp_cannot_satisfy_earlier_lift(self):
        ctx=self.context(attr(CoarseCause.EXECUTION_CONTACT_DEVIATION,
            anomalous=(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT,)))
        actions=np.zeros((16,7));actions[8:,2]=.1
        t=trace_candidate_recovery(ctx,effect(),actions)
        self.assertEqual(t.facts['grasped']['metrics']['terminal_proposal'],.9)
        self.assertEqual(t.facts['grasped']['metrics']['ordered_forecast_support'],0.)
        self.assertIn('grasped',t.unresolved_requirements)

    def test_intermediate_forecast_is_soft_and_does_not_write_belief(self):
        ctx=self.context(attr(CoarseCause.EXECUTION_CONTACT_DEVIATION,
            anomalous=(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT,)))
        actions=np.zeros((16,7));actions[2:,6]=1.;actions[8:,2]=.1
        before=ctx.belief.snapshot()
        t=trace_candidate_recovery(ctx,effect(),actions,forecasts=[ForecastFactEvent(
            'grasped',TriValue.TRUE,.8,4,'pred_frame4','predicted_intermediate')])
        self.assertEqual(t.facts['grasped']['metrics']['ordered_forecast_support'],.8)
        self.assertEqual(before,ctx.belief.snapshot())
        self.assertTrue(t.command_order['closes_before_upward'])
        self.assertIn('grasped',t.unresolved_requirements)

    def test_endpoint_cannot_be_backdated(self):
        with self.assertRaises(ValueError):
            ForecastFactEvent('place_ready',TriValue.TRUE,.9,5,'final_image')

    def test_hard_false_is_not_overridden_by_forecast(self):
        b=prior();b.facts['grasped']=BeliefFact(TriValue.FALSE,.9,'observation',2,('drop_seen',))
        t=trace_candidate_recovery(self.context(attr(CoarseCause.NORMAL),b),effect(),np.zeros((16,7)))
        self.assertTrue(any(n.startswith('grasped=false') for n in t.hard_reasons))

    def test_dependency_ablation_changes_propagation_features(self):
        a=attr(CoarseCause.OBJECT_SHIFT,anomalous=(ConsistencyFactor.WORLD_STATE_CONSISTENT,))
        full=trace_candidate_recovery(self.context(a),effect(),np.zeros((16,7)))
        flat=trace_candidate_recovery(self.context(a,graph=DependencyGraph(())),effect(),np.zeros((16,7)))
        self.assertFalse(np.array_equal(full.features,flat.features))
        self.assertTrue(full.facts['grasped']['paths'])

    def test_articulated_plan_has_no_lift_drawer_requirement(self):
        e=typed_gate_effect(effect(),'open')
        t=trace_candidate_recovery(self.context(attr(CoarseCause.NORMAL)),e,np.zeros((16,7)))
        self.assertIsNone(t.facts['lifted']['required_at_step'])
        self.assertIsNone(t.facts['grasped']['required_at_step'])

    def test_context_does_not_mutate_prior(self):
        b=prior();before=b.snapshot()
        self.context(attr(CoarseCause.OBJECT_SHIFT,anomalous=(ConsistencyFactor.WORLD_STATE_CONSISTENT,)),b)
        self.assertEqual(b.snapshot(),before)

    def test_missing_contract_blocks_training(self):
        with self.assertRaises(RuntimeError):require_recovery_training_ready({'schema':SCHEMA})

    def test_observed_grasp_cannot_refresh_during_cross_view_conflict(self):
        a=attr(CoarseCause.UNKNOWN)
        a=replace(a,evidence_quality=EvidenceQuality(True,True,True,True))
        ctx=self.context(a,observations=[CurrentFactEvidence('grasped',TriValue.TRUE,.9,'contradictory_views')])
        self.assertIs(ctx.belief.facts['grasped'].value,TriValue.UNKNOWN)

    def test_unreliable_primary_cannot_borrow_reliable_wrist_quality(self):
        a=attr(CoarseCause.VISUAL_OCCLUSION,reliable=False)
        observations=[CurrentFactEvidence('target_pose_current',TriValue.TRUE,.9,'primary_image',view='primary')]
        ctx=self.context(a,observations=observations)
        self.assertIs(ctx.belief.facts['target_pose_current'].value,TriValue.UNKNOWN)
        observations=[CurrentFactEvidence('target_pose_current',TriValue.TRUE,.9,'wrist_image',view='wrist')]
        ctx=self.context(a,observations=observations)
        self.assertIs(ctx.belief.facts['target_pose_current'].value,TriValue.TRUE)

    def test_unreliable_record_does_not_claim_execution_false(self):
        a=attr(CoarseCause.EXECUTION_CONTACT_DEVIATION,anomalous=(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT,))
        a=replace(a,evidence_quality=EvidenceQuality(True,True,False))
        ctx=self.context(a)
        self.assertIs(ctx.belief.facts['execution_consistent'].value,TriValue.UNKNOWN)

    def test_unreliable_record_cannot_refresh_command_consistency(self):
        a=replace(attr(CoarseCause.UNKNOWN),evidence_quality=EvidenceQuality(True,True,False))
        ctx=self.context(a,observations=[CurrentFactEvidence('execution_consistent',TriValue.TRUE,.9,
            'corrupt_record','measured_execution','record')])
        self.assertIs(ctx.belief.facts['execution_consistent'].value,TriValue.UNKNOWN)

    def selector(self,**kwargs):
        weights=np.zeros(27);weights[0]=1.
        params=dict(prior=prior(),attribution=attr(CoarseCause.NORMAL),observations=[],block_index=3,
            effects=[effect()],actions=[np.zeros((16,7))],values=[.7],relation='inside',
            backbone=CandidateUtilityModel(np.zeros(27),np.ones(27),weights,0.))
        params.update(kwargs);return select_recovery_candidate(**params)

    def test_single_entry_emits_belief_paths_and_candidate_trace(self):
        a=attr(CoarseCause.OBJECT_SHIFT,anomalous=(ConsistencyFactor.WORLD_STATE_CONSISTENT,))
        result=self.selector(attribution=a)
        self.assertEqual(result.traces[0].facts['grasped']['value'],'unknown')
        self.assertTrue(result.traces[0].facts['grasped']['paths'])
        self.assertIsNone(result.selected_candidate_id)

    def test_untrained_residual_cannot_be_deployed(self):
        class Model:
            def score(self,x):return 0.
        with self.assertRaises(RuntimeError):self.selector(recovery_model=Model())

    def test_single_entry_preserves_hard_false_and_explicit_fallback(self):
        b=prior();b.facts['grasped']=BeliefFact(TriValue.FALSE,.9,'observation',2,('drop',))
        result=self.selector(prior=b)
        self.assertIsNone(result.selected_candidate_id);self.assertEqual(result.fallback,'requery')

    def test_cold_start_control_reproduces_value_weight_backbone(self):
        effects=[effect(),replace(effect(),candidate_id=1)]
        result=self.selector(effects=effects,actions=[np.zeros((16,7))]*2,values=[.7,.8])
        self.assertEqual(result.selected_candidate_id,1)

    def test_unknown_safety_fact_cannot_escape_gate_by_confidence_decay(self):
        b=prior();b.facts['grasped']=BeliefFact(TriValue.UNKNOWN,.001,'attribution',2,('uncertain',))
        result=self.selector(prior=b)
        self.assertIsNone(result.selected_candidate_id)
        self.assertIn('grasped=unknown_requires_current_evidence',result.traces[0].hard_reasons)

    def test_current_observed_holding_allows_after_shift_without_prediction_override(self):
        a=attr(CoarseCause.OBJECT_SHIFT,anomalous=(ConsistencyFactor.WORLD_STATE_CONSISTENT,))
        result=self.selector(attribution=a,observations=[CurrentFactEvidence(
            'grasped',TriValue.TRUE,.9,'current_wrist',view='wrist')])
        self.assertEqual(result.selected_candidate_id,0)


class RecoverySupervisionTests(unittest.TestCase):
    def test_final_success_is_not_recovery_supervision(self):
        a=audit_recovery_supervision({'outcome':{'success':True}},directory=Path('.'))
        self.assertFalse(a['ready']);self.assertFalse(any(a['mask'].values()))

    def test_missing_actual_record_is_masked(self):
        a=audit_recovery_supervision(None,directory=Path('.'))
        self.assertFalse(a['ready']);self.assertEqual(a['missing'],['recovery_execution_record'])

    def fixture(self,folder):
        record=dict(schema=SCHEMA,candidate_id=0,executed_length=16,step_indices=list(range(17)),
            snapshot_sha256='a'*64,query_observation_sha256='b'*64,planned_actions_sha256='c'*64,
            predicate_labels_path='labels.json')
        shapes={'primary_sequence':(17,4,4,3),'wrist_sequence':(17,4,4,3),
                'proprio_sequence':(17,8),'requested_actions':(16,7),'applied_actions':(16,7)}
        for name,shape in shapes.items():
            record[name]=name+'.npy';np.save(folder/record[name],np.zeros(shape))
        labels=dict(schema=SCHEMA,role='supervision_only',candidate_id=0,predicates={})
        for name in PREDICATES:
            labels['predicates'][name]=dict(before='unknown',after='unknown',mask=False,
                quality=0.,source='observed_image_support',evidence_steps=[0,16],
                evidence_ids=['primary_sequence','wrist_sequence'])
        labels['predicates']['grasped'].update(before='false',after='true',mask=True,quality=.9)
        (folder/'labels.json').write_text(json.dumps(labels))
        return record,labels

    def test_valid_actual_block_and_partial_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);record,_=self.fixture(folder)
            a=audit_recovery_supervision(record,directory=folder)
            self.assertTrue(a['ready']);self.assertEqual(a['supervised_predicates'],['grasped'])

    def test_prediction_source_is_not_allowed_for_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);record,labels=self.fixture(folder)
            labels['predicates']['grasped']['source']='candidate_prediction'
            (folder/'labels.json').write_text(json.dumps(labels))
            self.assertFalse(audit_recovery_supervision(record,directory=folder)['ready'])

    def test_unknown_label_cannot_be_unmasked(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);record,labels=self.fixture(folder)
            labels['predicates']['target_visible']['mask']=True
            (folder/'labels.json').write_text(json.dumps(labels))
            self.assertFalse(audit_recovery_supervision(record,directory=folder)['ready'])

    def test_old_proprio_or_short_trajectory_cannot_fill_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);record,_=self.fixture(folder)
            np.save(folder/record['proprio_sequence'],np.zeros((1,8)))
            self.assertFalse(audit_recovery_supervision(record,directory=folder)['ready'])

    def test_command_equality_cannot_be_grasp_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);record,labels=self.fixture(folder)
            labels['predicates']['grasped']['source']='measured_command_difference'
            (folder/'labels.json').write_text(json.dumps(labels))
            self.assertFalse(audit_recovery_supervision(record,directory=folder)['ready'])

    def test_label_candidate_identity_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);record,labels=self.fixture(folder);labels['candidate_id']=1
            (folder/'labels.json').write_text(json.dumps(labels))
            self.assertFalse(audit_recovery_supervision(record,directory=folder)['ready'])

    def test_weak_label_quality_is_masked_not_forced_supervision(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);record,labels=self.fixture(folder)
            labels['predicates']['grasped']['quality']=.1
            (folder/'labels.json').write_text(json.dumps(labels))
            self.assertFalse(audit_recovery_supervision(record,directory=folder)['ready'])


if __name__=='__main__':unittest.main()

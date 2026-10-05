import json
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from wam_reranking.candidate_utility import CandidateUtilityModel
from wam_reranking.baseline_consensus import select_baseline_consensus
from wam_reranking.contracts import AttributionOutput, CandidateDecision, CoarseCause, ConsistencyFactor, EvidenceQuality, TriValue
from wam_reranking.evidence_arbitration import affirmative_improvement, annotate_evidence_routes, evidence_route, select_evidence_arbitration
from wam_reranking.evidence_residual import EvidenceResidualModel


def attribution(record):
    return AttributionOutput(**{**record, 'projected_cause':CoarseCause(record['projected_cause']),
        'evidence_quality':EvidenceQuality(**record['evidence_quality']),
        'factor_states':{k:TriValue(v) for k,v in record.get('factor_states',{}).items()}})


def example():
    weights=np.zeros(27);weights[14]=.2
    model=CandidateUtilityModel(np.zeros(27),np.ones(27),weights,0.)
    residual=EvidenceResidualModel(np.ones(10),np.zeros(10),np.ones(10))
    xs={0:np.zeros(27),1:np.zeros(27)}
    xs[0][0]=1.;xs[1][0]=.9;xs[1][14]=1.
    for x in xs.values():
        x[5]=.8;x[8]=.9;x[6]=.5;x[16]=1.;x[15]=.1
    xs[1][6]=.7
    flags={'preserve_official_value':1.,'unresolved_without_supported_physical_recovery':1.}
    ds=[CandidateDecision(i,True,xs[i][0],xs[i][0],(),dict(flags)) for i in (0,1)]
    probs={f.value:.9 for f in ConsistencyFactor};probs['cause_resolved']=0.
    a=AttributionOutput(probs,{c.value:float(c is CoarseCause.UNKNOWN) for c in CoarseCause},
                        CoarseCause.UNKNOWN,.8,.1,EvidenceQuality(True,True,True),'test')
    return dict(decisions=ds,backbone_features=xs,cause_features={i:np.zeros(10) for i in (0,1)},
                backbone=model,residual=residual,attribution=a)


class ArbitrationTests(unittest.TestCase):
    def chosen(self,c):return select_evidence_arbitration(**c)[0].candidate_id

    def test_positive_trusted_support_can_switch(self):
        self.assertEqual(self.chosen(example()),1)

    def test_unreliable_visual_with_both_legacy_flags_keeps_v6(self):
        c=example();a=c['attribution']
        c['attribution']=replace(a,factor_probs={**a.factor_probs,'observation_reliable':.1})
        self.assertEqual(self.chosen(c),0)

    def test_cross_view_conflict_keeps_v6(self):
        c=example();c['attribution']=replace(c['attribution'],evidence_quality=EvidenceQuality(True,True,True,True))
        self.assertEqual(self.chosen(c),0)

    def test_missing_attribution_is_not_assumed_reliable(self):
        c=example();c.pop('attribution');self.assertEqual(self.chosen(c),0)

    def test_one_unreliable_view_blocks_dual_view_comparison(self):
        c=example();c['attribution']=replace(c['attribution'],evidence_quality=EvidenceQuality(True,False,True))
        self.assertEqual(self.chosen(c),0)

    def test_contact_quality_tradeoff_keeps_v6(self):
        c=example();c['backbone_features'][1][5]=.6
        self.assertEqual(self.chosen(c),0)

    def test_equal_support_is_not_positive_evidence(self):
        c=example();c['backbone_features'][1][6]=.5
        self.assertEqual(self.chosen(c),0)

    def test_phase_mismatch_keeps_v6(self):
        c=example();c['backbone_features'][1][16]=0.;c['backbone_features'][1][20]=1.
        c['backbone_features'][1][13]=1.
        self.assertEqual(self.chosen(c),0)

    def test_place_compares_release_not_grasp(self):
        c=example()
        for x in c['backbone_features'].values():x[16]=0.;x[20]=1.
        c['backbone_features'][0][13]=.8;c['backbone_features'][1][13]=.5
        self.assertEqual(self.chosen(c),0)

    def test_low_contact_quality_is_not_trusted(self):
        c=example()
        for x in c['backbone_features'].values():x[5]=.4
        self.assertEqual(self.chosen(c),0)

    def test_higher_trajectory_risk_is_a_tradeoff(self):
        c=example();c['backbone_features'][1][15]=.2
        self.assertEqual(self.chosen(c),0)

    def test_hard_rejected_anchor_cannot_override(self):
        c=example();c['decisions'][1]=replace(c['decisions'][1],accepted=False,rejection_reasons=('grasped=false',))
        self.assertEqual(self.chosen(c),0)

    def test_all_rejected_remains_explicit_fallback(self):
        c=example();c['decisions']=[replace(d,accepted=False) for d in c['decisions']]
        self.assertIsNone(select_evidence_arbitration(**c)[0])

    def test_physical_inconsistency_does_not_enter_unresolved_normal_route(self):
        c=example();a=c['attribution'];c['attribution']=replace(a,factor_probs={**a.factor_probs,'world_state_consistent':.1})
        self.assertEqual(self.chosen(c),0)

    def test_annotations_do_not_mutate_gates_or_inputs(self):
        c=example();before=[dict(d.components) for d in c['decisions']]
        ds=annotate_evidence_routes(c['decisions'],c['attribution'])
        self.assertEqual(before,[d.components for d in c['decisions']])
        self.assertEqual([d.accepted for d in ds],[d.accepted for d in c['decisions']])
        self.assertTrue(evidence_route(c['attribution']).comparison_allowed)

    def test_order_invariant(self):
        c=example();c['decisions'].reverse();self.assertEqual(self.chosen(c),1)

    def test_nan_cannot_supply_comparison_evidence(self):
        c=example();c['backbone_features'][1][5]=np.nan
        with self.assertRaises(ValueError):affirmative_improvement(c['backbone_features'][1],c['backbone_features'][0])


class RecordedFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((Path(__file__).parent/'fixtures/evidence_arbitration_failures.json').read_text())

    def check_case(self,index):
        f=self.fixture;c=f['cases'][index]
        ds=[CandidateDecision(**{**d,'rejection_reasons':tuple(d['rejection_reasons'])}) for d in c['decisions']]
        args=dict(decisions=ds,
            backbone_features={int(k):np.asarray(v) for k,v in c['backbone_features'].items()},
            cause_features={d.candidate_id:np.zeros(10) for d in ds},
            backbone=CandidateUtilityModel.from_record(f['backbone']),
            residual=EvidenceResidualModel.from_record(f['residual']))
        failed,_=select_baseline_consensus(**args)
        choice,_=select_evidence_arbitration(**args,attribution=attribution(c['attribution']))
        self.assertEqual(failed.candidate_id,c['recorded_v7'])
        self.assertNotEqual(c['recorded_v7'],c['expected_v6'])
        self.assertEqual(choice.candidate_id,c['expected_v6'])

    def test_recorded_unreliable_visual_failure(self):self.check_case(0)
    def test_recorded_quality_tradeoff_failure(self):self.check_case(1)


if __name__=='__main__':unittest.main()

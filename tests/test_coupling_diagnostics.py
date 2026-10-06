from dataclasses import replace
import unittest

import numpy as np

from wam_reranking.candidate_utility import CandidateUtilityModel, FEATURE_NAMES
from wam_reranking.contracts import CandidateDecision
from wam_reranking.coupling_diagnostics import check_membership, choose, feature_audit, fit_diagnostic
from wam_reranking.evidence_residual import EvidenceResidualModel


class CouplingDiagnosisTests(unittest.TestCase):
    def setUp(self):
        n = len(FEATURE_NAMES)
        self.base = CandidateUtilityModel(np.zeros(n), np.ones(n), np.zeros(n), 0.)
        self.residual = EvidenceResidualModel(np.ones(10), np.zeros(10), np.ones(10))
        self.features = {}
        for i, value in enumerate((.7, .6)):
            x = np.zeros(n); x[0] = value
            self.features[i] = dict(backbone=x, cause=np.zeros(10),
                decision=CandidateDecision(i, True, value, value, (), {}))
        self.rows = [dict(key=[0,s,0,'clean'],split=split,features=self.features,
            supervision={0:dict(success=True,steps=10,calls=1),1:dict(success=False,steps=20,calls=2)})
            for s,split in ((0,'train'),(6,'val'))]

    def test_no_cause_ranking_equals_backbone(self):
        self.assertEqual(choose(self.features,self.base,self.residual,panel='ranking_only'),0)

    def test_gate_panel_never_selects_rejected(self):
        f = {i:dict(c) for i,c in self.features.items()}
        f[0]['decision']=replace(f[0]['decision'],accepted=False,rejection_reasons=('witnessed_false',))
        self.assertEqual(choose(f,self.base,self.residual,panel='current_variant_gate'),1)

    def test_all_rejected_is_unobserved_not_failure(self):
        f = {i:{**c,'decision':replace(c['decision'],accepted=False)} for i,c in self.features.items()}
        self.assertIsNone(choose(f,self.base,self.residual,panel='current_variant_gate'))

    def test_group_leakage_is_rejected(self):
        r = [dict(self.rows[0]),{**self.rows[1],'key':[0,0,0,'object_shift']}]
        with self.assertRaisesRegex(ValueError,'leakage'):check_membership(r)

    def test_gt_field_in_features_is_rejected(self):
        r = [{**self.rows[0],'features':{i:{**c,'success':True} for i,c in self.features.items()}}]
        with self.assertRaisesRegex(ValueError,'GT'):check_membership(r)

    def test_constant_nonzero_attribution_does_not_discriminate(self):
        f = {i:{**c,'cause':np.ones(10)} for i,c in self.features.items()}
        audit=feature_audit([{**self.rows[0],'features':f}])
        self.assertEqual(audit['nonzero_cause_pools'],1)
        self.assertEqual(audit['within_pool_discriminative_pools'],0)
        self.assertEqual(audit['nonzero_success_preference_pairs'],0)

    def test_val_outcomes_do_not_affect_fit_or_gain(self):
        first=fit_diagnostic(self.rows,self.base,panel='ranking_only')
        changed=[self.rows[0],{**self.rows[1],'supervision':{0:dict(success=False,steps=30,calls=2),1:dict(success=True,steps=5,calls=1)}}]
        second=fit_diagnostic(changed,self.base,panel='ranking_only')
        self.assertEqual(first['model'].to_record(),second['model'].to_record())
        self.assertEqual(first['report']['train_gain_calibration'],second['report']['train_gain_calibration'])


if __name__=='__main__':unittest.main()

from dataclasses import replace
import unittest

import numpy as np

from wam_reranking.contextual_cause_residual import contextual_cause_features,fit_context_residual
from wam_reranking.contracts import CoarseCause,ConsistencyFactor,EvidenceQuality,TriValue
from test_evidence_arbitration import example


class ContextResidualTests(unittest.TestCase):
    def setUp(self):
        self.attr=example()['attribution']
        self.x=np.arange(27)/27.;self.z=np.zeros(10)
        self.attr=replace(self.attr,projected_cause=CoarseCause.OBJECT_SHIFT,
            class_probs={c.value:float(c is CoarseCause.OBJECT_SHIFT) for c in CoarseCause},
            factor_probs={**self.attr.factor_probs,'world_state_consistent':.1,'cause_resolved':1.},
            factor_states={f.value:TriValue.TRUE for f in ConsistencyFactor})

    def test_same_candidate_different_cause_changes_context(self):
        a=contextual_cause_features(self.x,self.z,self.attr)
        b=contextual_cause_features(self.x,self.z,replace(self.attr,projected_cause=CoarseCause.EXECUTION_CONTACT_DEVIATION,
            class_probs={c.value:float(c is CoarseCause.EXECUTION_CONTACT_DEVIATION) for c in CoarseCause},
            factor_probs={**self.attr.factor_probs,'execution_contact_consistent':.1}))
        self.assertFalse(np.array_equal(a,b))

    def test_normal_and_unknown_zero_preserve_shared_backbone(self):
        for cause in (CoarseCause.NORMAL,CoarseCause.UNKNOWN):
            self.assertFalse(np.any(contextual_cause_features(self.x,self.z,replace(self.attr,projected_cause=cause,
                class_probs={c.value:float(c is cause) for c in CoarseCause}))))

    def test_conflicting_visuals_zero(self):
        attr=replace(self.attr,projected_cause=CoarseCause.UNKNOWN,
            class_probs={c.value:float(c is CoarseCause.UNKNOWN) for c in CoarseCause},
            evidence_quality=EvidenceQuality(True,True,True,True))
        self.assertFalse(np.any(contextual_cause_features(self.x,self.z,attr)))

    def test_value_is_not_a_second_free_feature(self):
        changed=self.x.copy();changed[0]=99
        self.assertTrue(np.array_equal(contextual_cause_features(self.x,self.z,self.attr),
            contextual_cause_features(changed,self.z,self.attr)))

    def test_zero_correction_and_out_of_support(self):
        a=contextual_cause_features(self.x,self.z,self.attr)
        model=fit_context_residual([(a,np.zeros(70),0.)],[a,np.zeros(70)])
        self.assertEqual(model.score(np.zeros(70)),0.)
        self.assertEqual(model.score(np.ones(70)*1e5),0.)

    def test_fit_deterministic(self):
        a=contextual_cause_features(self.x,self.z,self.attr)
        self.assertEqual(fit_context_residual([(a,np.zeros(70),0.)],[a,np.zeros(70)]).to_record(),
            fit_context_residual([(a,np.zeros(70),0.)],[a,np.zeros(70)]).to_record())


if __name__=='__main__':unittest.main()

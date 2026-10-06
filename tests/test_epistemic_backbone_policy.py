import copy
import json
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from wam_reranking.contracts import EvidenceQuality
from wam_reranking.epistemic_backbone_policy import select_epistemic_backbone
from test_evidence_arbitration import example


class EpistemicBackboneTests(unittest.TestCase):
    def setUp(self):
        self.case = example()
        # Deliberately incomparable action phases: no cause evidence licenses
        # replacing the established candidate ranker with official value.
        self.case['backbone_features'][1][16:21] = [0, 0, 0, 1, 0]

    def chosen(self):
        selected, _ = select_epistemic_backbone(**self.case)
        return None if selected is None else selected.candidate_id

    def test_reliable_unknown_preserves_trained_backbone_across_phase_change(self):
        self.assertEqual(self.chosen(), 1)

    def test_rejected_backbone_anchor_never_overridden(self):
        d = self.case['decisions'][1]
        self.case['decisions'][1] = replace(d, accepted=False,
            rejection_reasons=('lifted=false@0.99',))
        self.assertEqual(self.chosen(), 0)

    def test_conflicting_views_keep_historical_route(self):
        self.case['attribution'] = replace(self.case['attribution'],
            evidence_quality=EvidenceQuality(True, True, True, True))
        self.assertEqual(self.chosen(), 0)

    def test_unreliable_primary_keeps_historical_route(self):
        self.case['attribution'] = replace(self.case['attribution'],
            evidence_quality=EvidenceQuality(False, True, True))
        self.assertEqual(self.chosen(), 0)

    def test_missing_attribution_does_not_restore_anchor(self):
        self.case.pop('attribution')
        self.assertEqual(self.chosen(), 0)

    def test_physical_negative_keeps_historical_route(self):
        a = self.case['attribution']
        self.case['attribution'] = replace(a,
            factor_probs={**a.factor_probs, 'world_state_consistent': .1})
        self.assertEqual(self.chosen(), 0)

    def test_all_rejected_stays_fallback(self):
        self.case['decisions'] = [replace(d, accepted=False) for d in self.case['decisions']]
        self.assertIsNone(self.chosen())

    def test_inputs_and_residual_gain_are_unchanged(self):
        before = copy.deepcopy(self.case)
        self.chosen()
        self.assertEqual(self.case['decisions'], before['decisions'])
        self.assertEqual(self.case['attribution'], before['attribution'])
        self.assertEqual(self.case['residual'].gain, before['residual'].gain)

    def test_recorded_first_divergence_keeps_gate_and_restores_backbone(self):
        fixture = Path(__file__).parent / 'fixtures/backbone_preservation_first_divergence.json'
        if not fixture.exists():
            self.skipTest('Recorded cloud fixture not synchronized yet')
        from wam_reranking.candidate_utility import CandidateUtilityModel
        from wam_reranking.contracts import AttributionOutput, CandidateDecision, CoarseCause, TriValue
        from wam_reranking.evidence_arbitration import select_evidence_arbitration
        from wam_reranking.evidence_residual import EvidenceResidualModel
        data = json.loads(fixture.read_text(encoding='utf-8'))
        a = data['attribution']
        attribution = AttributionOutput(**{**a, 'projected_cause': CoarseCause(a['projected_cause']),
            'evidence_quality': EvidenceQuality(**a['evidence_quality']),
            'factor_states': {k: TriValue(v) for k, v in a.get('factor_states', {}).items()}})
        decisions = [CandidateDecision(**{**d, 'rejection_reasons': tuple(d['rejection_reasons'])})
                     for d in data['decisions']]
        features = {int(k): np.asarray(v) for k, v in data['backbone_features'].items()}
        args = dict(decisions=decisions, attribution=attribution, backbone_features=features,
            cause_features={k: np.zeros(10) for k in features},
            backbone=CandidateUtilityModel.from_record(data['backbone']),
            residual=EvidenceResidualModel.from_record(data['residual']))
        old, _ = select_evidence_arbitration(**args)
        new, _ = select_epistemic_backbone(**args)
        self.assertEqual(old.candidate_id, data['recorded_choice'])
        self.assertEqual(new.candidate_id, data['expected_backbone_choice'])
        self.assertTrue(new.accepted)
        self.assertFalse(new.rejection_reasons)
        self.assertEqual(args['residual'].gain, 0.)


if __name__ == '__main__': unittest.main()

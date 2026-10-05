import unittest
from dataclasses import replace

import numpy as np

from wam_reranking.baseline_consensus import contact_view_support, select_baseline_consensus
from wam_reranking.candidate_utility import CandidateUtilityModel
from wam_reranking.contracts import CandidateDecision
from wam_reranking.evidence_residual import EvidenceResidualModel


class BaselineConsensusTests(unittest.TestCase):
    def setup_case(self):
        weights = np.zeros(27); weights[14] = .2
        model = CandidateUtilityModel(np.zeros(27), np.ones(27), weights, 0.)
        residual = EvidenceResidualModel(np.ones(10), np.zeros(10), np.ones(10))
        features = {0: np.zeros(27), 1: np.zeros(27)}
        features[0][0] = 1.; features[1][0] = .9; features[1][14] = 1.
        for x in features.values():
            x[5] = .8; x[6] = .5; x[8] = .9; x[16] = 1.
        components = {'preserve_official_value': 1., 'unresolved_without_supported_physical_recovery': 1.}
        decisions = [CandidateDecision(i, True, features[i][0], features[i][0], (), dict(components)) for i in (0, 1)]
        return dict(decisions=decisions, backbone_features=features,
                    cause_features={i: np.zeros(10) for i in (0, 1)}, backbone=model, residual=residual)

    def test_equal_support_keeps_candidate_baseline(self):
        self.assertEqual(select_baseline_consensus(**self.setup_case())[0].candidate_id, 1)

    def test_lower_official_contact_keeps_candidate(self):
        c = self.setup_case(); c['backbone_features'][0][6] = .1
        self.assertEqual(select_baseline_consensus(**c)[0].candidate_id, 1)

    def test_two_axis_official_dominance_preserves_value(self):
        c = self.setup_case(); c['backbone_features'][0][6] = .7; c['backbone_features'][0][5] = .9
        self.assertEqual(select_baseline_consensus(**c)[0].candidate_id, 0)

    def test_contact_quality_tradeoff_keeps_candidate(self):
        c = self.setup_case(); c['backbone_features'][0][6] = .7; c['backbone_features'][0][5] = .5
        self.assertEqual(select_baseline_consensus(**c)[0].candidate_id, 1)

    def test_phase_mismatch_cannot_supply_comparable_support(self):
        c = self.setup_case(); c['backbone_features'][0][16] = 0.; c['backbone_features'][0][20] = 1.
        c['backbone_features'][0][13] = 1.
        self.assertEqual(select_baseline_consensus(**c)[0].candidate_id, 1)

    def test_place_compares_release_not_grasp(self):
        c = self.setup_case()
        for x in c['backbone_features'].values(): x[16] = 0.; x[20] = 1.
        c['backbone_features'][0][6] = 1.; c['backbone_features'][1][6] = 0.
        c['backbone_features'][0][13] = .1; c['backbone_features'][1][13] = .9
        self.assertEqual(select_baseline_consensus(**c)[0].candidate_id, 1)

    def test_known_physical_false_is_not_overridden(self):
        c = self.setup_case()
        c['decisions'][1] = replace(c['decisions'][1], accepted=False, rejection_reasons=('lifted=false',))
        self.assertEqual(select_baseline_consensus(**c)[0].candidate_id, 0)

    def test_all_rejected_remains_fallback(self):
        c = self.setup_case(); c['decisions'] = [replace(d, accepted=False) for d in c['decisions']]
        self.assertIsNone(select_baseline_consensus(**c)[0])

    def test_unreliable_view_route_is_unchanged(self):
        c = self.setup_case()
        c['decisions'] = [replace(d, components={'preserve_official_value': 1.}) for d in c['decisions']]
        self.assertEqual(select_baseline_consensus(**c)[0].candidate_id, 0)

    def test_no_official_route_leaves_frozen_ranker_unchanged(self):
        c = self.setup_case(); c['decisions'] = [replace(d, components={}) for d in c['decisions']]
        self.assertEqual(select_baseline_consensus(**c)[0].candidate_id, 1)

    def test_no_infinite_prediction_is_accepted(self):
        with self.assertRaises(ValueError): contact_view_support(np.full(27, np.nan))

    def test_candidate_order_does_not_change_selection(self):
        c = self.setup_case(); c['decisions'].reverse()
        self.assertEqual(select_baseline_consensus(**c)[0].candidate_id, 1)


if __name__ == '__main__': unittest.main()

"""Same wrapper for backbone controls and recovery-score interventions."""
from dataclasses import replace
from unittest.mock import patch
import unittest

import numpy as np

from wam_reranking.candidate_utility import CandidateUtilityModel, select_with_utility
from wam_reranking.contracts import BeliefFact, CoarseCause, Stage, TriValue
from wam_reranking.recovery_contract import FEATURE_NAMES, SCHEMA, select_recovery_candidate
from test_recovery_coupling_scores import (
    TestOnlyLinearRecoveryHead, attribution, candidate, recorded_prior,
)


class SharedRecoverySelectionTests(unittest.TestCase):
    def setUp(self):
        self.effects = [candidate(0, .9), replace(candidate(1, .1), confidence=.5)]
        self.actions = [np.zeros((16, 7)), np.zeros((16, 7))]
        self.values = [.6, .7]
        weights = np.zeros(27); weights[1] = .2625
        self.backbone = CandidateUtilityModel(np.zeros(27), np.ones(27), weights, 0.)
        self.audit = dict(schema=SCHEMA, recovery_training_ready=True,
                          group_leakage=False, deployment_gt_leakage=False,
                          role="unit_test_fixture_not_dataset_evidence")

    def select(self, **kwargs):
        parameters = dict(
            prior=recorded_prior(), attribution=attribution(CoarseCause.NORMAL),
            observations=(), block_index=3, effects=self.effects,
            actions=self.actions, values=self.values, relation="inside",
            backbone=self.backbone, training_audit=self.audit)
        parameters.update(kwargs)
        return select_recovery_candidate(**parameters)

    def zero_head(self):
        return TestOnlyLinearRecoveryHead("normal.grasped.terminal_proposal", 0.)

    def test_zero_head_keeps_equivalence_tie_break_identical_to_no_model(self):
        control = self.select()
        zero = self.select(recovery_model=self.zero_head())
        # Score0 is 0.005 greater, but the frozen 0.01 tie tolerance keeps the
        # higher official value candidate. Direct argmax would incorrectly differ.
        self.assertGreater(control.scores[0], control.scores[1])
        self.assertLess(control.scores[0] - control.scores[1], .01)
        self.assertEqual(control.selected_candidate_id, 1)
        self.assertEqual(zero.selected_candidate_id, control.selected_candidate_id)
        self.assertEqual(zero.scores, control.scores)
        self.assertEqual(zero.fallback, control.fallback)

    def test_small_nonzero_correction_cannot_bypass_same_frozen_tie_margin(self):
        result = self.select(recovery_model=TestOnlyLinearRecoveryHead(
            "normal.grasped.terminal_proposal", .002))
        self.assertGreater(result.scores[0], result.scores[1])
        self.assertLess(result.scores[0] - result.scores[1], .01)
        self.assertEqual(result.selected_candidate_id, 1)

    def test_nonzero_correction_can_change_choice_once_same_margin_is_exceeded(self):
        result = self.select(recovery_model=TestOnlyLinearRecoveryHead(
            "normal.grasped.terminal_proposal", .05))
        self.assertGreater(result.scores[0] - result.scores[1], .01)
        self.assertEqual(result.selected_candidate_id, 0)

    def test_larger_frozen_backbone_switch_margin_is_shared_unchanged(self):
        backbone = replace(self.backbone, switch_margin=.1)
        control = self.select(backbone=backbone)
        intervention = self.select(backbone=backbone,
            recovery_model=TestOnlyLinearRecoveryHead("normal.grasped.terminal_proposal", .05))
        self.assertEqual(control.selected_candidate_id, 1)
        self.assertEqual(intervention.selected_candidate_id, 1)
        self.assertEqual(backbone.switch_margin, .1)

    def test_shared_wrapper_is_called_in_both_arms(self):
        with patch("wam_reranking.recovery_contract.select_with_utility",
                   wraps=select_with_utility) as shared:
            self.select()
            self.select(recovery_model=self.zero_head())
        self.assertEqual(shared.call_count, 2)
        for call in shared.call_args_list:
            self.assertEqual(call.args[2].switch_margin, self.backbone.switch_margin)

    def test_shared_pareto_guard_cannot_be_bypassed_by_recovery_branch(self):
        weights = np.zeros(27); weights[1] = .5
        backbone = replace(self.backbone, weights=weights)
        effects = [self.effects[0], replace(self.effects[1], confidence=.1)]

        def audit_axis_fixture(decisions, features, model):
            # Inject audited-axis fixture at the common selection boundary.
            # It is not a claim these risks have been measured in real data.
            decisions = [replace(decision, components={**decision.components,
                "dependency_risk": .8 if decision.candidate_id == 0 else .1,
                "uncertainty": .8 if decision.candidate_id == 0 else .1})
                for decision in decisions]
            return select_with_utility(decisions, features, model)

        with patch("wam_reranking.recovery_contract.select_with_utility",
                   side_effect=audit_axis_fixture):
            control = self.select(backbone=backbone, effects=effects)
            intervention = self.select(backbone=backbone, effects=effects,
                recovery_model=TestOnlyLinearRecoveryHead("normal.grasped.terminal_proposal", .05))
        self.assertGreater(control.scores[0], control.scores[1])
        self.assertGreater(intervention.scores[0], intervention.scores[1])
        self.assertEqual(control.selected_candidate_id, 1)
        self.assertEqual(intervention.selected_candidate_id, 1)

    def test_identical_score_and_value_tie_uses_candidate_identity_in_both_arms(self):
        effects = [replace(candidate(4, .9), candidate_id=4), candidate(2, .9)]
        control = self.select(effects=effects, values=[.7, .7])
        zero = self.select(effects=effects, values=[.7, .7], recovery_model=self.zero_head())
        self.assertEqual(control.selected_candidate_id, 2)
        self.assertEqual(zero.selected_candidate_id, 2)

    def test_direct_false_and_explicit_fallback_are_identical_across_arms(self):
        prior = recorded_prior()
        prior.facts["grasped"] = BeliefFact(TriValue.FALSE, .9,
            "observation", 2, ("actual_drop",))
        effects = [replace(effect, stage=Stage.LIFT) for effect in self.effects]
        control = self.select(prior=prior, effects=effects)
        intervention = self.select(prior=prior, effects=effects,
            recovery_model=TestOnlyLinearRecoveryHead("normal.grasped.terminal_proposal", 100.))
        self.assertIsNone(control.selected_candidate_id)
        self.assertIsNone(intervention.selected_candidate_id)
        self.assertEqual(control.fallback, "requery")
        self.assertEqual(intervention.fallback, control.fallback)
        self.assertEqual(intervention.scores, control.scores)
        self.assertEqual([trace.hard_reasons for trace in intervention.traces],
                         [trace.hard_reasons for trace in control.traces])

    def test_unreliable_visual_all_rejected_route_stays_reobserve_in_both_arms(self):
        prior = recorded_prior()
        prior.facts["grasped"] = BeliefFact(TriValue.UNKNOWN, .001,
            "attribution", 2, ("uncertain_grasp",))
        effects = [replace(effect, stage=Stage.LIFT) for effect in self.effects]
        attr = attribution(CoarseCause.VISUAL_OCCLUSION)
        control = self.select(prior=prior, effects=effects, attribution=attr)
        zero = self.select(prior=prior, effects=effects, attribution=attr,
                           recovery_model=self.zero_head())
        self.assertIsNone(control.selected_candidate_id)
        self.assertIsNone(zero.selected_candidate_id)
        self.assertEqual(control.fallback, "reobserve")
        self.assertEqual(zero.fallback, control.fallback)

    def test_nonfinite_composite_score_is_rejected_not_selected_in_either_arm(self):
        class NonfiniteBackbone:
            switch_margin = .01
            def score(self, features):
                return float("nan")
        for model in (None, self.zero_head()):
            with self.subTest(model=model), self.assertRaises(ValueError):
                self.select(backbone=NonfiniteBackbone(), recovery_model=model)


if __name__ == "__main__":
    unittest.main()

"""Synthetic control equality tests; no fitting or task performance claims."""
from dataclasses import replace
import unittest

from tests.test_direct_recovery import fixture_head, pool_fixture
from wam_reranking.direct_recovery import (
    DirectRecoveryModel, assert_equal_candidate_information, masked_attribution_input,
)


class EffectOnlyControlTests(unittest.TestCase):
    def test_masking_needs_does_not_erase_ordinary_effect_comparator(self):
        pool = pool_fixture()
        masked = tuple(masked_attribution_input(c) for c in pool)
        model = DirectRecoveryModel((fixture_head(),))
        self.assertTrue(assert_equal_candidate_information(pool, masked))
        first = model.score_effect_only_pool(pool, reference_candidate_id=1)
        second = model.score_effect_only_pool(masked, reference_candidate_id=1)
        self.assertEqual(first, second)
        self.assertGreater(first[0]["effect_only_residual"], 0.)
        self.assertEqual(model.score_pool(masked, reference_candidate_id=1)[0]["cause_recovery_residual"], 0.)

    def test_common_head_support_guard_and_reference_are_preserved(self):
        pool = pool_fixture()
        model = DirectRecoveryModel((fixture_head(valid=False),))
        result = model.score_effect_only_pool(pool, reference_candidate_id=1)
        self.assertEqual(result[0]["effect_only_residual"], 0.)
        self.assertEqual(result[1]["effect_only_residual"], 0.)
        self.assertEqual(result[0]["effect_contributions"][0]["inactive_reason"], "supervision_admission_not_passed")

    def test_no_cause_budget_can_leak_into_ordinary_comparator(self):
        pool = pool_fixture()
        empty = tuple(replace(c, needs=()) for c in pool)
        model = DirectRecoveryModel((fixture_head(),))
        self.assertEqual(model.score_effect_only_pool(pool, reference_candidate_id=1),
                         model.score_effect_only_pool(empty, reference_candidate_id=1))
        for r in model.score_effect_only_pool(empty, reference_candidate_id=1).values():
            self.assertLessEqual(abs(r["effect_only_residual"]), .1)
            self.assertFalse(r["attribution_or_belief_needs_used"])
            self.assertTrue(r["same_recovery_heads_and_candidate_information"])
            self.assertFalse(r["observed_execution_labels_used_as_score_input"])

    def test_both_arms_reject_unknown_reference(self):
        model = DirectRecoveryModel((fixture_head(),))
        for method in (model.score_pool, model.score_effect_only_pool):
            with self.assertRaisesRegex(ValueError, "reference"):
                method(pool_fixture(), reference_candidate_id=9)


if __name__ == "__main__":
    unittest.main()

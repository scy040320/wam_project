"""Synthetic optimizer/contract checks; none establishes real V9 task benefit."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from wam_reranking import recovery_contract
from wam_reranking.recovery_residual import (
    AUXILIARY_LOSS_WEIGHT, L2_WEIGHT, RESIDUAL_CAP, RecoveryResidualModel,
    fit_recovery_residual,
)


def synthetic_rows():
    names = recovery_contract.FEATURE_NAMES
    index = names.index("object_shift.grasped.terminal_proposal")
    rows, auxiliary = [], []
    for pool in range(4):
        for candidate_id in (0, 1):
            features = np.zeros(len(names))
            features[index] = float(candidate_id == 0)
            rows.append(dict(
                pool_id=f"synthetic_fixture:pool{pool}", candidate_id=candidate_id,
                split="train", features=features,
                backbone_score=.5 if candidate_id == 0 else .6,
                success=candidate_id == 0))
            auxiliary.append(dict(
                row_id=f"synthetic_fixture:aux{pool}:{candidate_id}",
                split="train", features=features,
                targets=np.asarray([float(candidate_id == 0)]),
                masks=np.asarray([True])))
    return rows, auxiliary


class RecoveryResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.auxiliary = synthetic_rows()
        cls.model = fit_recovery_residual(
            rank_rows=cls.rows, auxiliary_rows=cls.auxiliary,
            feature_names=recovery_contract.FEATURE_NAMES,
            target_names=("grasped",), seed=0, epochs=80)

    def fit(self, **kwargs):
        parameters = dict(
            rank_rows=deepcopy(self.rows), auxiliary_rows=deepcopy(self.auxiliary),
            feature_names=recovery_contract.FEATURE_NAMES,
            target_names=("grasped",), seed=0, epochs=10)
        parameters.update(kwargs)
        return fit_recovery_residual(**parameters)

    def test_score_decomposes_frozen_backbone_plus_bounded_residual(self):
        row = self.rows[0]
        parts = self.model.score_decomposition(row["features"], row["backbone_score"])
        self.assertAlmostEqual(parts["total_score"],
            row["backbone_score"] + self.model.score(row["features"]))
        self.assertLessEqual(abs(parts["cause_recovery_residual"]), RESIDUAL_CAP)
        self.assertFalse(parts["outside_train_support"])
        self.assertTrue(parts["auxiliary_predictions_are_not_current_belief_evidence"])

    def test_synthetic_success_preference_can_change_relative_score(self):
        good, bad = self.rows[:2]
        before = good["backbone_score"] - bad["backbone_score"]
        after = before + self.model.score(good["features"]) - self.model.score(bad["features"])
        self.assertGreater(after, before)
        # This is an optimizer wiring fixture, not a real task-performance gate.
        self.assertGreater(after, 0.)

    def test_model_does_not_mutate_frozen_backbone_scores_or_training_arrays(self):
        self.assertEqual([row["backbone_score"] for row in self.rows], [.5, .6] * 4)
        self.assertEqual(self.rows[0]["features"].max(), 1.)
        self.assertFalse(self.model.shared_projection.flags.writeable)

    def test_zero_residual_control_keeps_projection_and_returns_exact_zero(self):
        zero = self.model.zero_residual_control()
        np.testing.assert_array_equal(zero.shared_projection, self.model.shared_projection)
        self.assertEqual(zero.score(self.rows[0]["features"]), 0.)
        self.assertEqual(zero.score(self.rows[1]["features"]), 0.)

    def test_outside_training_support_is_explicit_zero_not_extrapolated_gain(self):
        features = self.rows[0]["features"].copy()
        features[0] = 9.
        parts = self.model.score_decomposition(features, .75)
        self.assertTrue(parts["outside_train_support"])
        self.assertEqual(parts["cause_recovery_residual"], 0.)
        self.assertEqual(parts["total_score"], .75)

    def test_training_metadata_does_not_claim_independent_coupling_benefit(self):
        metadata = self.model.training_metadata
        self.assertFalse(metadata["independent_coupling_benefit_established"])
        self.assertFalse(metadata["forced_nonzero_residual"])
        self.assertEqual(metadata["input_roles"], ["train"])
        self.assertEqual(metadata["success_first_pair_count"], 4)
        self.assertEqual(metadata["observed_auxiliary_targets"], 8)
        self.assertEqual(metadata["auxiliary_loss_weight"], AUXILIARY_LOSS_WEIGHT)
        self.assertEqual(metadata["l2_weight"], L2_WEIGHT)
        self.assertTrue(metadata["shared_projection_receives_both_losses"])
        self.assertEqual(metadata["masks_by_target"], {"grasped": 8})
        self.assertTrue(all(len(value) == 64 for value in metadata["training_fingerprints"].values()))

    def test_auxiliary_loss_changes_the_projection_used_by_ranking(self):
        masked = deepcopy(self.auxiliary)
        for row in masked:
            row["masks"][:] = False
        with_aux = self.fit(epochs=1)
        without_aux = self.fit(auxiliary_rows=masked, epochs=1)
        self.assertFalse(np.array_equal(with_aux.shared_projection, without_aux.shared_projection))
        self.assertFalse(np.array_equal(with_aux.auxiliary_weights, without_aux.auxiliary_weights))
        # Same object/array is used by both forward branches, not two encoders.
        self.assertEqual(with_aux.shared_projection.shape[1], 8)

    def test_masked_unknown_labels_are_not_forced_to_positive_or_negative(self):
        masked_nan, masked_arbitrary = deepcopy(self.auxiliary), deepcopy(self.auxiliary)
        for row in masked_nan:
            row["masks"][:] = False; row["targets"][:] = np.nan
        for row in masked_arbitrary:
            row["masks"][:] = False; row["targets"][:] = 777.
        first = self.fit(auxiliary_rows=masked_nan)
        second = self.fit(auxiliary_rows=masked_arbitrary)
        np.testing.assert_array_equal(first.shared_projection, second.shared_projection)
        np.testing.assert_array_equal(first.ranking_weights, second.ranking_weights)
        self.assertEqual(first.training_metadata["observed_auxiliary_targets"], 0)
        self.assertEqual(first.score_decomposition(self.rows[0]["features"], .5)[
            "auxiliary_probabilities"], {"grasped": .5})

    def test_no_mixed_success_pool_leaves_rank_head_zero_even_with_auxiliary_training(self):
        rows = deepcopy(self.rows)
        for row in rows:
            row["success"] = True
        model = self.fit(rank_rows=rows)
        self.assertEqual(model.training_metadata["success_first_pair_count"], 0)
        self.assertFalse(model.training_metadata["ranking_supervision_available"])
        np.testing.assert_array_equal(model.ranking_weights, np.zeros(8))
        self.assertEqual(model.score(rows[0]["features"]), 0.)

    def test_val_test_confirmation_and_unmarked_rank_rows_are_rejected(self):
        for split in ("val", "test", "confirmation", "qualification", None):
            with self.subTest(split=split), self.assertRaises(ValueError):
                rows = deepcopy(self.rows); rows[0]["split"] = split
                self.fit(rank_rows=rows)

    def test_val_or_unmarked_auxiliary_rows_are_rejected(self):
        for split in ("val", "confirmation", None):
            with self.subTest(split=split), self.assertRaises(ValueError):
                rows = deepcopy(self.auxiliary); rows[0]["split"] = split
                self.fit(auxiliary_rows=rows)

    def test_qc_or_confirmation_role_cannot_be_smuggled_by_setting_train_split(self):
        for role in ("qc", "clean_b", "confirmation", "qualification"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                rows = deepcopy(self.rows); rows[0]["role"] = role
                self.fit(rank_rows=rows)
            with self.subTest(auxiliary_role=role), self.assertRaises(ValueError):
                rows = deepcopy(self.auxiliary); rows[0]["dataset_role"] = role
                self.fit(auxiliary_rows=rows)

    def test_duplicate_candidate_or_auxiliary_identity_is_rejected(self):
        rows = deepcopy(self.rows); rows[1]["candidate_id"] = rows[0]["candidate_id"]
        with self.assertRaises(ValueError):
            self.fit(rank_rows=rows)
        auxiliary = deepcopy(self.auxiliary)
        auxiliary[1]["row_id"] = auxiliary[0]["row_id"]
        with self.assertRaises(ValueError):
            self.fit(auxiliary_rows=auxiliary)

    def test_unmasked_nan_or_out_of_range_target_is_rejected(self):
        for invalid in (np.nan, -1., 2.):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                rows = deepcopy(self.auxiliary); rows[0]["targets"][:] = invalid
                self.fit(auxiliary_rows=rows)

    def test_nonfinite_features_or_backbone_scores_are_rejected(self):
        rows = deepcopy(self.rows); rows[0]["features"][0] = np.nan
        with self.assertRaises(ValueError):
            self.fit(rank_rows=rows)
        rows = deepcopy(self.rows); rows[0]["backbone_score"] = float("inf")
        with self.assertRaises(ValueError):
            self.fit(rank_rows=rows)
        with self.assertRaises(ValueError):
            self.model.score_decomposition(self.rows[0]["features"], float("nan"))

    def test_feature_reordering_and_legacy_360_contract_are_rejected(self):
        wrong_order = tuple(reversed(recovery_contract.FEATURE_NAMES))
        with self.assertRaises(ValueError):
            self.fit(feature_names=wrong_order)
        with self.assertRaises(ValueError):
            self.fit(feature_names=recovery_contract.FEATURE_NAMES[:360])
        with self.assertRaises(ValueError):
            replace(self.model, feature_schema="legacy_recovery_features_v1")

    def test_score_does_not_accept_outcome_or_label_arguments(self):
        with self.assertRaises(TypeError):
            self.model.score(self.rows[0]["features"], success=True)
        with self.assertRaises(TypeError):
            self.model.score(self.rows[0]["features"], labels=[1.])

    def test_serialization_round_trip_preserves_scores_and_fingerprints(self):
        record = self.model.to_record()
        restored = RecoveryResidualModel.from_record(record)
        np.testing.assert_array_equal(restored.shared_projection, self.model.shared_projection)
        self.assertEqual(restored.score(self.rows[0]["features"]), self.model.score(self.rows[0]["features"]))
        self.assertEqual(restored.training_metadata["training_fingerprints"],
                         self.model.training_metadata["training_fingerprints"])

    def test_serialized_weight_or_feature_version_tampering_is_rejected(self):
        for field in ("ranking_weights", "feature_schema"):
            record = deepcopy(self.model.to_record())
            if field == "ranking_weights":
                record[field][0] += .01
            else:
                record[field] = "legacy360"
            with self.subTest(field=field), self.assertRaises(ValueError):
                RecoveryResidualModel.from_record(record)

    def test_training_is_deterministic_and_curve_starts_with_zero_rank_correction(self):
        first = self.fit(epochs=10)
        second = self.fit(epochs=10)
        np.testing.assert_array_equal(first.shared_projection, second.shared_projection)
        np.testing.assert_array_equal(first.ranking_weights, second.ranking_weights)
        self.assertEqual(first.training_curve[0]["max_abs_train_correction"], 0.)
        self.assertEqual(first.training_curve[-1]["epoch"], 10)
        self.assertLess(first.training_curve[-1]["total_loss"], first.training_curve[0]["total_loss"])

    def test_fixed_small_cap_cannot_guarantee_success_or_zero_harm(self):
        # A baseline score margin >2*cap cannot be overcome by this contract.
        good, bad = self.rows[:2]
        corrected_margin = -1. + self.model.score(good["features"]) - self.model.score(bad["features"])
        self.assertLess(corrected_margin, 0.)
        self.assertEqual(self.model.residual_cap, .1)


if __name__ == "__main__":
    unittest.main()

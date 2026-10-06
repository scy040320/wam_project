"""Synthetic direct-score contracts, NOT trained task-performance evidence.

All nonzero weights and SHA strings below are TEST FIXTURES.  No source
manifest was approved by these tests and no model is trained or deployed.
"""
from dataclasses import replace
import unittest

import numpy as np

from wam_reranking.direct_recovery import (
    CandidateRecoveryInput, CandidateRequirement, DirectRecoveryModel, HeadAdmission, LABEL_SCHEMA,
    LinearRecoveryHead, NEED_SOURCE, RAW_FEATURE_NAMES, RawCandidateFeatures,
    RecoveryNeed, assert_equal_candidate_information, extract_raw_candidate_features,
    masked_attribution_input, no_dependency_input, replace_control_needs,
)
from wam_reranking.recovery_contract import (
    COMMAND_FEATURE_NAMES, FEATURE_NAMES, ROUTES, VISUAL_EVIDENCE_KEYS,
)


def historical_fixture(visual=.6, routes=(.2, .3, .1, .4, 0.)):
    values = np.zeros(len(FEATURE_NAMES))
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    for i, name in enumerate(COMMAND_FEATURE_NAMES):
        values[index["command." + name]] = i / 16.
    for route, weight in zip(ROUTES, routes):
        for key in VISUAL_EVIDENCE_KEYS:
            values[index[f"{route}.visual.{key}.value"]] = weight * visual
            values[index[f"{route}.visual.{key}.available"]] = weight
    return values


def raw_fixture(value):
    return extract_raw_candidate_features(historical_fixture(value))


def fixture_admission(predicate="grasped", *, valid=True, deadline=4):
    """Self-declared metadata fixture, not an actual passed data audit."""
    positive, negative, paired = [0] * 17, [0] * 17, [0] * 17
    positive[deadline], negative[deadline], paired[deadline] = 2, 2, 1
    return HeadAdmission(predicate, predicate + ".valid_at_use_time",
        "a" * 64, "b" * 64, "c" * 64,
        tuple(positive), tuple(negative), tuple(paired),
        label_integrity_passed=valid, prefix_causal=valid,
        masked_is_not_negative=valid, private_teacher_excluded_from_x=valid)


def fixture_head(predicate="grasped", **admission_kwargs):
    weights = np.zeros(len(RAW_FEATURE_NAMES))
    weights[RAW_FEATURE_NAMES.index("visual.grasp_support_after.value")] = 4.
    return LinearRecoveryHead(fixture_admission(predicate, **admission_kwargs),
        weights, -2., .2, np.full(len(weights), -100.), np.full(len(weights), 100.))


def fixture_need(predicate="grasped", *, cause="object_shift", propagated=True,
                 weight=.9, deadline=4):
    return RecoveryNeed(cause, predicate, weight, deadline, propagated,
                        evidence_ids=("actual_before_block", "previous_attribution"))


def pool_fixture(*, predicate="grasped", deadline=4):
    return (
        CandidateRecoveryInput(0, .5, raw_fixture(.9),
                               (fixture_need(predicate, deadline=deadline),),
                               (CandidateRequirement(predicate, deadline),)),
        CandidateRecoveryInput(1, .55, raw_fixture(.1),
                               (fixture_need(predicate, deadline=deadline),),
                               (CandidateRequirement(predicate, deadline),)),
    )


class DirectRecoveryContractTests(unittest.TestCase):
    def test_raw_features_have_same_14_command_and_16_visual_value_mask_fields(self):
        raw = raw_fixture(.6)
        self.assertEqual(len(RAW_FEATURE_NAMES), 46)
        self.assertEqual(raw.values.shape, (46,))
        self.assertTrue(raw.visual_reconstruction_complete)
        np.testing.assert_allclose(raw.values[:14], np.arange(14) / 16.)
        np.testing.assert_allclose(raw.values[14::2], .6)
        np.testing.assert_array_equal(raw.values[15::2], 1.)
        self.assertFalse(raw.values.flags.writeable)

    def test_raw_information_is_identical_for_different_attribution_route_weights(self):
        a = extract_raw_candidate_features(historical_fixture(.3))
        b = extract_raw_candidate_features(historical_fixture(.3, (0., 0., 1., 0., 0.)))
        np.testing.assert_array_equal(a.values, b.values)
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_contradictory_route_visual_ratio_is_rejected_not_averaged(self):
        values = historical_fixture(.3)
        values[FEATURE_NAMES.index("normal.visual.grasp_support_after.value")] = .19
        with self.assertRaisesRegex(ValueError, "contradictory"):
            extract_raw_candidate_features(values)

    def test_value_without_available_is_not_accepted_as_a_visual_measurement(self):
        values = historical_fixture(.3)
        values[FEATURE_NAMES.index("normal.visual.grasp_support_after.available")] = 0.
        with self.assertRaisesRegex(ValueError, "without route availability"):
            extract_raw_candidate_features(values)

    def test_all_zero_routes_do_not_create_false_absence_or_supervised_scores(self):
        raw = extract_raw_candidate_features(historical_fixture(.3, (0.,) * 5))
        self.assertFalse(raw.visual_reconstruction_complete)
        probability, reason = fixture_head().predict(raw, 4)
        self.assertIsNone(probability)
        self.assertEqual(reason, "raw_candidate_visual_not_reconstructable")

    def test_original_preexecution_visual_can_disambiguate_zero_route_mass(self):
        visual = {key: .4 for key in VISUAL_EVIDENCE_KEYS}
        raw = extract_raw_candidate_features(historical_fixture(.3, (0.,) * 5), original_visual=visual)
        self.assertTrue(raw.visual_reconstruction_complete)
        np.testing.assert_allclose(raw.values[14::2], .4)

    def test_extra_gt_outcome_key_cannot_enter_original_visual_interface(self):
        with self.assertRaisesRegex(ValueError, "only pre-execution"):
            extract_raw_candidate_features(historical_fixture(), original_visual={"success": 1.})

    def test_original_visual_disagreement_with_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            extract_raw_candidate_features(historical_fixture(.3), original_visual={key: .4 for key in VISUAL_EVIDENCE_KEYS})

    def test_schema_drift_and_nonfinite_vectors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            extract_raw_candidate_features(historical_fixture(), feature_schema="legacy360")
        values = historical_fixture(); values[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            extract_raw_candidate_features(values)

    def test_missing_supervision_keeps_nonzero_head_at_exact_zero_score(self):
        pool = pool_fixture()
        result = DirectRecoveryModel((fixture_head(valid=False),)).score_pool(pool, reference_candidate_id=1)
        self.assertEqual(result[0]["cause_recovery_residual"], 0.)
        self.assertEqual(result[0]["total_score"], .5)
        self.assertEqual(result[0]["recovery_contributions"][0]["inactive_reason"], "supervision_admission_not_passed")

    def test_legal_fixture_recovery_probability_directly_changes_candidate_score(self):
        # Synthetic heads establish wiring only, not an approved trained model.
        pool = pool_fixture()
        result = DirectRecoveryModel((fixture_head(),)).score_pool(pool, reference_candidate_id=1)
        contribution = result[0]["recovery_contributions"][0]
        expected = .1 * .9 * (contribution["predicted_recovery_probability"]
                              - contribution["reference_predicted_recovery_probability"])
        self.assertAlmostEqual(result[0]["cause_recovery_residual"], expected)
        self.assertGreater(expected, 0.)
        self.assertGreater(result[0]["total_score"], result[1]["total_score"])
        self.assertEqual(result[1]["cause_recovery_residual"], 0.)
        self.assertTrue(result[0]["auxiliary_predictions_do_not_restore_belief"])
        self.assertTrue(result[0]["no_hard_gate_override"])
        self.assertFalse(result[0]["observed_execution_labels_used_as_score_input"])

    def test_masked_and_no_dag_keep_visual_and_commands_exactly_equal(self):
        pool = pool_fixture()
        masked = tuple(masked_attribution_input(c) for c in pool)
        nodag = tuple(no_dependency_input(c) for c in pool)
        self.assertTrue(assert_equal_candidate_information(pool, masked, nodag))
        model = DirectRecoveryModel((fixture_head(),))
        full = model.score_pool(pool, reference_candidate_id=1)
        for control in (masked, nodag):
            result = model.score_pool(control, reference_candidate_id=1)
            self.assertEqual(result[0]["cause_recovery_residual"], 0.)
            self.assertGreater(full[0]["cause_recovery_residual"], result[0]["cause_recovery_residual"])
            self.assertEqual(result[0]["raw_candidate_sha256"], full[0]["raw_candidate_sha256"])

    def test_shuffled_unknown_does_not_guess_physical_need_or_change_candidate_information(self):
        pool = pool_fixture()
        control = tuple(replace_control_needs(c, (fixture_need(cause="unknown"),)) for c in pool)
        self.assertTrue(assert_equal_candidate_information(pool, control))
        result = DirectRecoveryModel((fixture_head(),)).score_pool(control, reference_candidate_id=1)
        self.assertEqual(result[0]["cause_recovery_residual"], 0.)
        self.assertEqual(result[0]["recovery_contributions"][0]["inactive_reason"], "no_active_cause_specific_missing_prerequisite")

    def test_no_dag_keeps_direct_root_need(self):
        pool = tuple(replace(c, needs=(fixture_need("target_pose_current", propagated=False),),
            requirements=(CandidateRequirement("target_pose_current", 4),)) for c in pool_fixture())
        control = tuple(no_dependency_input(c) for c in pool)
        self.assertEqual(pool, control)

    def test_no_dag_does_not_amplify_direct_term_when_zero_gain_descendant_is_removed(self):
        needs = (fixture_need("target_pose_current", propagated=False), fixture_need("grasped"))
        requirements = (CandidateRequirement("target_pose_current", 4), CandidateRequirement("grasped", 4))
        pool = tuple(replace(c, needs=needs, requirements=requirements) for c in pool_fixture())
        constant = replace(fixture_head("grasped"), weights=np.zeros(len(RAW_FEATURE_NAMES)))
        model = DirectRecoveryModel((fixture_head("target_pose_current"), constant))
        full = model.score_pool(pool, reference_candidate_id=1)
        nodag = model.score_pool(tuple(no_dependency_input(c) for c in pool), reference_candidate_id=1)
        self.assertAlmostEqual(full[0]["cause_recovery_residual"], nodag[0]["cause_recovery_residual"])
        self.assertEqual(full[0]["common_requirement_normalization"], nodag[0]["common_requirement_normalization"])

    def test_multiple_causes_do_not_multiply_same_predicate_credit(self):
        needs = (fixture_need(weight=1.), fixture_need(cause="execution_contact_deviation", weight=1.))
        pool = tuple(replace(c, needs=needs) for c in pool_fixture())
        model = DirectRecoveryModel((fixture_head(),))
        combined = model.score_pool(pool, reference_candidate_id=1)
        single = model.score_pool(tuple(replace(c, needs=(needs[0],)) for c in pool), reference_candidate_id=1)
        self.assertAlmostEqual(combined[0]["cause_recovery_residual"], single[0]["cause_recovery_residual"])

    def test_normal_and_unknown_physical_guess_do_not_get_recovery_bonus(self):
        model = DirectRecoveryModel((fixture_head(),))
        for cause in ("normal", "unknown", "visual_occlusion"):
            pool = tuple(replace(c, needs=(fixture_need(cause=cause),)) for c in pool_fixture())
            with self.subTest(cause=cause):
                result = model.score_pool(pool, reference_candidate_id=1)
                self.assertEqual(result[0]["cause_recovery_residual"], 0.)

    def test_entry_prerequisite_cannot_be_established_by_future_prediction(self):
        pool = pool_fixture(deadline=0)
        model = DirectRecoveryModel((fixture_head(deadline=0),))
        result = model.score_pool(pool, reference_candidate_id=1)
        self.assertEqual(result[0]["cause_recovery_residual"], 0.)
        self.assertEqual(result[0]["recovery_contributions"][0]["inactive_reason"], "future_prediction_cannot_establish_entry_fact")

    def test_endpoint_event_cannot_prove_earlier_prerequisite(self):
        head = replace(fixture_head(), prediction_kind="predicted_endpoint_only")
        result = DirectRecoveryModel((head,)).score_pool(pool_fixture(), reference_candidate_id=1)
        self.assertEqual(result[0]["cause_recovery_residual"], 0.)
        self.assertEqual(result[0]["recovery_contributions"][0]["inactive_reason"], "endpoint_cannot_certify_earlier_prerequisite")

    def test_endpoint_head_can_predict_endpoint_only_when_its_own_labels_are_certified(self):
        head = replace(fixture_head(deadline=16), prediction_kind="predicted_endpoint_only")
        probability, reason = head.predict(raw_fixture(.9), 16)
        self.assertIsNone(reason)
        self.assertGreater(probability, .5)

    def test_masked_or_only_positive_target_cannot_enable_head(self):
        for field in ("negative_by_time", "positive_by_time", "paired_difference_by_time"):
            head = fixture_head()
            head = replace(head, admission=replace(head.admission, **{field: (0,) * 17}))
            with self.subTest(field=field):
                result = DirectRecoveryModel((head,)).score_pool(pool_fixture(), reference_candidate_id=1)
                self.assertEqual(result[0]["cause_recovery_residual"], 0.)

    def test_head_must_have_supervision_at_actual_requirement_time_not_only_frame16(self):
        model = DirectRecoveryModel((fixture_head(deadline=16),))
        result = model.score_pool(pool_fixture(deadline=4), reference_candidate_id=1)
        self.assertEqual(result[0]["cause_recovery_residual"], 0.)

    def test_raw_contact_joint_and_release_atoms_cannot_be_silently_remapped(self):
        for predicate, atom in (("grasped", "left_finger_target_contact"),
                                ("target_pose_current", "joint_state_changed"),
                                ("placed", "released_sufficient_evidence")):
            with self.subTest(atom=atom), self.assertRaisesRegex(ValueError, "silently"):
                replace(fixture_admission(predicate), target_name=atom)

    def test_teacher_need_source_is_rejected_instead_of_using_gt_missing_fact(self):
        with self.assertRaisesRegex(ValueError, "teacher, GT"):
            replace(fixture_need(), source="offline_simulator_measurement")
        self.assertEqual(fixture_need().source, NEED_SOURCE)

    def test_unbound_hash_or_confirmation_fitting_cannot_enable_head(self):
        head = fixture_head()
        head = replace(head, admission=replace(head.admission, label_manifest_sha256="not_a_hash"))
        self.assertEqual(DirectRecoveryModel((head,)).score_pool(pool_fixture(), reference_candidate_id=1)[0]["cause_recovery_residual"], 0.)
        with self.assertRaisesRegex(ValueError, "train-only"):
            replace(fixture_admission(), input_roles=("train", "val"))

    def test_cause_controls_have_identical_raw_ood_status(self):
        pool = pool_fixture()
        head = fixture_head()
        head = replace(head, support_max=np.full(len(RAW_FEATURE_NAMES), .01))
        model = DirectRecoveryModel((head,))
        full = model.score_pool(pool, reference_candidate_id=1)
        self.assertEqual(full[0]["cause_recovery_residual"], 0.)
        self.assertEqual(full[0]["recovery_contributions"][0]["inactive_reason"], "outside_train_raw_candidate_support")
        control = tuple(replace_control_needs(c, (replace(c.needs[0], weight=.5),)) for c in pool)
        shuffled = model.score_pool(control, reference_candidate_id=1)
        self.assertEqual(shuffled[0]["recovery_contributions"][0]["inactive_reason"], full[0]["recovery_contributions"][0]["inactive_reason"])

    def test_common_pool_offset_cancels_and_total_many_head_gain_is_bounded(self):
        same = tuple(replace(c, raw=raw_fixture(.5)) for c in pool_fixture())
        model = DirectRecoveryModel((fixture_head(),))
        self.assertEqual(model.score_pool(same, reference_candidate_id=1)[0]["cause_recovery_residual"], 0.)
        predicates = ("grasped", "lifted", "place_ready", "placed", "target_reachable")
        pool = tuple(replace(c, needs=tuple(fixture_need(p, weight=1.) for p in predicates),
            requirements=tuple(CandidateRequirement(p, 4) for p in predicates)) for c in pool_fixture())
        model = DirectRecoveryModel(tuple(fixture_head(p) for p in predicates))
        result = model.score_pool(pool, reference_candidate_id=1)
        self.assertLessEqual(abs(result[0]["cause_recovery_residual"]), .1)
        self.assertGreater(result[0]["cause_recovery_residual"], 0.)

    def test_raw_changed_control_or_duplicate_candidate_is_rejected(self):
        pool = pool_fixture()
        changed = (replace(pool[0], raw=raw_fixture(.2)), pool[1])
        with self.assertRaisesRegex(ValueError, "raw candidate"):
            assert_equal_candidate_information(pool, changed)
        with self.assertRaisesRegex(ValueError, "unique"):
            DirectRecoveryModel(()).score_pool((pool[0], pool[0]), reference_candidate_id=0)

    def test_shuffled_control_cannot_invent_candidate_requirement_or_use_time(self):
        candidate = pool_fixture()[0]
        for need in (fixture_need("lifted"), fixture_need(deadline=9)):
            with self.subTest(need=need), self.assertRaisesRegex(ValueError, "cannot invent/change"):
                replace_control_needs(candidate, (need,))
        changed_requirement = replace(candidate,
            needs=(fixture_need(deadline=9),), requirements=(CandidateRequirement("grasped", 9),))
        with self.assertRaisesRegex(ValueError, "raw candidate information"):
            assert_equal_candidate_information((candidate,), (changed_requirement,))

    def test_first_event_or_cumulative_label_does_not_admit_at_use_validity_head(self):
        with self.assertRaisesRegex(ValueError, "silently"):
            replace(fixture_admission(), target_name="grasped.established_by_time")
        with self.assertRaisesRegex(ValueError, "schema"):
            replace(fixture_admission(), label_schema="first_certified_event_time_v1")
        head = replace(fixture_head(), time_weight=-2.)
        # At-use validity may decrease after a release; no monotonic fiction.
        self.assertLess(head.time_weight, 0.)

    def test_no_admitted_heads_is_legal_zero_control_and_no_training_trigger(self):
        pool = pool_fixture()
        result = DirectRecoveryModel(()).score_pool(pool, reference_candidate_id=1)
        self.assertEqual([v["cause_recovery_residual"] for v in result.values()], [0., 0.])
        self.assertFalse(hasattr(DirectRecoveryModel, "fit"))

    def test_model_round_trip_is_hashed_and_does_not_mutate_backbone(self):
        model = DirectRecoveryModel((fixture_head(),))
        record = model.to_record()
        self.assertTrue(record["no_training_in_module"])
        self.assertEqual(record["label_schema"], LABEL_SCHEMA)
        restored = DirectRecoveryModel.from_record(record)
        pool = pool_fixture()
        self.assertEqual(model.score_pool(pool, reference_candidate_id=1), restored.score_pool(pool, reference_candidate_id=1))
        self.assertEqual([c.backbone_score for c in pool], [.5, .55])
        record["heads"][0]["bias"] = 9.
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            DirectRecoveryModel.from_record(record)

    def test_score_input_has_no_slot_for_true_success_or_future_event(self):
        with self.assertRaises(TypeError):
            CandidateRecoveryInput(0, .5, raw_fixture(.5), (), success=True)
        with self.assertRaises(TypeError):
            fixture_head().predict(raw_fixture(.5), 4, actual_event=True)


if __name__ == "__main__":
    unittest.main()

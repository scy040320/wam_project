"""Synthetic contract tests; no empirical mechanism or safety claim."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from wam_reranking.contracts import (
    AttributionOutput, CandidateDecision, CandidateEffect, CoarseCause,
    ConsistencyFactor, EvidenceQuality, Stage, TriValue,
)
from wam_reranking.soft_attribution_utility import (
    CAP, FEATURE_NAMES, NoFitError, SoftUtilityModel, fast_selection, fit_once,
    select_soft_utility, soft_features,
)


def attribution(world=.15, quality=None):
    factors = {f.value: .95 for f in ConsistencyFactor}
    factors[ConsistencyFactor.WORLD_STATE_CONSISTENT.value] = 1. - world
    probabilities = {c.value: .025 for c in CoarseCause}
    probabilities[CoarseCause.UNKNOWN.value] = .9
    quality = quality or EvidenceQuality(True, True, True)
    cause = CoarseCause.UNKNOWN if quality.cross_view_conflict else CoarseCause.NORMAL
    return AttributionOutput(factors, probabilities, cause, .025, .7,
        quality, "actual:block2",
        {k: TriValue.TRUE if v >= .5 else TriValue.FALSE for k, v in factors.items()},
        {k: max(v, 1.-v) for k, v in factors.items()})


def effects():
    return [CandidateEffect(i, Stage.LIFT, {"grasped": .55}, {}, .8,
        dict(relation_score_after=.2+.15*i, relation_confidence=.8,
             cross_view_agreement=.45, contact_confidence=.8,
             grasp_support_after=.4+.1*i, predicted_release_support=.1*i,
             target_gripper_distance_after=.4-.05*i, visual_support=.8,
             trajectory_risk=.1+.05*i), {}) for i in range(4)]


def model(gain=1.):
    weight = np.zeros(len(FEATURE_NAMES)); weight[0] = 2.
    return SoftUtilityModel(np.ones(len(weight)), weight, np.ones(len(weight)), gain)


def decision_list(accepted=(True, True, True, True)):
    return [CandidateDecision(i, yes, .5+i*.01, .5, (), {}) for i, yes in enumerate(accepted)]


def train_pool(key=(0, 0, 0, "clean"), reference_success=False):
    x = np.zeros((4, len(FEATURE_NAMES))); x[1, 0] = 1.
    return dict(key=key, split="train", x=x, base_scores=[.03, 0., -.02, -.03],
        reference_id=0, accepted=[True]*4, official_values=[.5, .4, .3, .2], value_id=0,
        outcomes=[dict(success=reference_success, steps=80, calls=5),
                  dict(success=True, steps=40, calls=3),
                  dict(success=False, steps=352, calls=21),
                  dict(success=False, steps=352, calls=21)], fallback_outcome=None)


class SoftFeatureTests(unittest.TestCase):
    def test_low_probability_is_soft_not_all_or_nothing(self):
        es = effects(); attr = attribution(.15); original = deepcopy(attr)
        matrix, audit = soft_features(attr, es, "inside")
        self.assertGreater(float(np.ptp(matrix[:, 0])), 0.)
        self.assertTrue(attr.factor_states[ConsistencyFactor.WORLD_STATE_CONSISTENT.value] is TriValue.TRUE)
        self.assertEqual(attr, original)
        self.assertFalse(audit["live_belief_mutated"])
        self.assertFalse(audit["physical_recovery_probability"])

    def test_before_delta_motion_and_after_outcome_are_ignored(self):
        es = effects(); baseline, _ = soft_features(attribution(), es, "inside")
        changed = [replace(e, evidence={**e.evidence, "relation_score_before": 999,
            "relation_score_delta": -999, "target_displacement": 999,
            "target_delta_x": 999, "current_grasped_support": 1.,
            "actual_success": True, "future_truth_grasped": True}) for e in es]
        actual, audit = soft_features(attribution(), changed, "inside")
        np.testing.assert_array_equal(actual, baseline)
        self.assertFalse(audit["before_fields_used"])

    def test_open_closed_are_opposite_and_articulated_not_directional(self):
        opened, _ = soft_features(attribution(), effects(), "open")
        closed, _ = soft_features(attribution(), effects(), "closed")
        unknown, audit = soft_features(attribution(), effects(), "articulated")
        self.assertGreater(opened[3, 0], opened[0, 0])
        self.assertLess(closed[3, 0], closed[0, 0])
        np.testing.assert_array_equal(unknown[:, 0], np.zeros(4))
        self.assertFalse(audit["articulated_direction_known"])

    def test_unreliable_or_conflicting_vision_closes_visual_only(self):
        for quality in (EvidenceQuality(False, False, True), EvidenceQuality(True, True, True, True)):
            matrix, audit = soft_features(attribution(quality=quality), effects(), "inside")
            self.assertFalse(matrix.any())
            self.assertFalse(audit["command_channel_changed"])

    def test_masked_preserves_main_effect_and_no_dag_changes_required_relevance(self):
        matrix, audit = soft_features(attribution(), effects(), "inside", masked=True)
        self.assertFalse(matrix[:, :17].any())
        self.assertTrue(matrix[:, 17:19].any())
        changed, _ = soft_features(attribution(.8), effects(), "inside", masked=True)
        np.testing.assert_array_equal(matrix, changed)
        zero, _ = soft_features(attribution(), effects(), "inside", ablate_all=True)
        self.assertFalse(zero.any())
        full, _ = soft_features(attribution(), effects(), "inside")
        no_dag, no_audit = soft_features(attribution(), effects(), "inside", no_dag=True)
        self.assertGreater(full[0, 4], no_dag[0, 4])
        self.assertTrue(no_audit["no_dag"])

    def test_geometric_distance_can_exceed_one_without_claiming_probability(self):
        es = effects(); es[0] = replace(es[0], evidence={**es[0].evidence, "target_gripper_distance_after": 1.3})
        actual, _ = soft_features(attribution(), es, "inside")
        self.assertEqual(actual[0, 1], 0.)
        bad = effects(); bad[0] = replace(bad[0], evidence={**bad[0].evidence, "target_gripper_distance_after": -.1})
        with self.assertRaises(ValueError):
            soft_features(attribution(), bad, "inside")

    def test_only_fresh_actual_certificate_removes_missing_required_relevance(self):
        snapshot = dict(facts={"grasped": dict(value="true", confidence=.9, source="observation",
            updated_at_block=3, evidence_ids=["actual-runtime-contact"])},
            history=[dict(predicate="grasped", new_value="true", reason="current_observation",
                direct=True, propagation_path=["grasped"], evidence_id="actual-runtime-contact")])
        original = deepcopy(snapshot)
        certified, audit = soft_features(attribution(), effects(), "inside", belief_snapshot=snapshot)
        initial = deepcopy(snapshot); initial["facts"]["grasped"]["evidence_ids"]=["initial_observation"]
        assumed, _ = soft_features(attribution(), effects(), "inside", belief_snapshot=initial)
        self.assertEqual(certified[0, 4], 0.)
        self.assertGreater(assumed[0, 4], 0.)
        self.assertEqual(audit["certified_current_required_facts"], ["grasped"])
        self.assertEqual(snapshot, original)
        for field, value in (("source","candidate_prediction"), ("updated_at_block",2), ("confidence",.1), ("value","unknown")):
            unverified=deepcopy(snapshot);unverified["facts"]["grasped"][field]=value
            matrix, _ = soft_features(attribution(), effects(), "inside", belief_snapshot=unverified)
            self.assertGreater(matrix[0, 4], 0.)
        source_only = deepcopy(snapshot); source_only["history"] = []
        matrix, _ = soft_features(attribution(), effects(), "inside", belief_snapshot=source_only)
        self.assertGreater(matrix[0, 4], 0.)

    def test_no_dag_cannot_borrow_full_history(self):
        snapshot = dict(facts={}, history=[dict(predicate="grasped", new_value="unknown",
            reason="world_state_consistent", propagation_path=["target_pose_current","target_reachable","grasped"])])
        with self.assertRaisesRegex(ValueError,"own unpropagated history"):
            soft_features(attribution(), effects(), "inside", no_dag=True, belief_snapshot=snapshot)

    def test_candidate_order_and_nonfinite_after_fail(self):
        with self.assertRaises(ValueError):
            soft_features(attribution(), effects()[::-1], "inside")
        bad = effects(); bad[0] = replace(bad[0], evidence={**bad[0].evidence, "relation_score_after": float("nan")})
        with self.assertRaises(ValueError):
            soft_features(attribution(), bad, "inside")


class SoftSelectionTests(unittest.TestCase):
    def test_zero_and_missing_source_preserve_reference(self):
        matrix, _ = soft_features(attribution(), effects(), "inside")
        out = select_soft_utility(decision_list(), [0., 0., 0., 0.], [.2,.3,.4,.5], 0, matrix, model(0.))
        self.assertEqual(out["selected_id"], 0)
        matrix[:] = 0.
        out = select_soft_utility(decision_list(), [0., 0., 0., 0.], [.2,.3,.4,.5], 0, matrix, model())
        self.assertEqual(out["selected_id"], 0)

    def test_reference_fallback_cannot_become_failed_candidate(self):
        matrix, _ = soft_features(attribution(), effects(), "inside")
        out = select_soft_utility(decision_list(), [0., 0., 0., 0.], [.2,.3,.4,.5], None, matrix, model())
        self.assertIsNone(out["selected_id"])

    def test_acceptance_margin_and_cap_are_hard(self):
        self.assertEqual(fast_selection([[0.,0.]], [[-.02,.02]], [0], [[True,False]], [[.2,.3]], [.01]).tolist(), [0])
        self.assertEqual(fast_selection([[0.,0.]], [[-.002,.002]], [0], [[True,True]], [[.2,.3]], [.01]).tolist(), [0])
        with self.assertRaises(ValueError):
            fast_selection([[0.,0.]], [[-.051,.051]], [0], [[True,True]], [[.2,.3]], [.01])
        with self.assertRaises(ValueError):
            fast_selection([[0.,0.]], [[0.,0.]], [0], [[False,True]], [[.2,.3]], [.01])

    def test_fast_and_deploy_equivalence(self):
        rng = np.random.default_rng(91)
        for _ in range(40):
            matrix = rng.uniform(0.,1.,size=(4,len(FEATURE_NAMES)))
            base, official = rng.normal(0.,.02,4), rng.uniform(0.,1.,4)
            accepted = rng.uniform(0.,1.,4)>.3; accepted[0]=True
            actual = select_soft_utility(decision_list(accepted), base, official, 0, matrix, model())
            fast = fast_selection(base[None,:], model().deltas(matrix)[None,:], [0], accepted[None,:], official[None,:], [.01])
            self.assertEqual(actual["selected_id"], int(fast[0]))
            self.assertLessEqual(np.ptp(actual["deltas"]), CAP+1e-12)

    def test_record_roundtrip_and_frozen_gain(self):
        restored = SoftUtilityModel.from_record(model().to_record())
        np.testing.assert_array_equal(restored.weights, model().weights)
        with self.assertRaises(ValueError):
            SoftUtilityModel(np.ones(len(FEATURE_NAMES)), np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)), .875)


class SoftTrainingTests(unittest.TestCase):
    def test_single_fit_success_before_cost_deployment_same(self):
        pool = train_pool()
        fitted, audit = fit_once([pool])
        out = select_soft_utility(decision_list(), pool["base_scores"], pool["official_values"], 0, pool["x"], fitted)
        self.assertEqual(out["selected_id"], 1)
        self.assertEqual(audit["steps"], 1200)
        self.assertEqual(audit["gain"], 1.)
        self.assertFalse(audit["validation_used"])
        self.assertTrue(audit["training_deployment_delta_equivalence"])
        self.assertEqual(audit["pairs"]["corrective"], 3)
        self.assertTrue(audit["training_deployment_top1_equivalence"])
        self.assertEqual([r["candidate_id"] for r in audit["corrective_competition"][0]["competitors"]], [0,2,3])

    def test_success_must_beat_third_failure_not_only_reference(self):
        pool = train_pool()
        pool["reference_id"] = 1
        pool["value_id"] = 1
        pool["base_scores"] = [.739868,.754243,.715393,.759027]
        pool["official_values"] = [.64264,.64341,.64249,.64309]
        pool["outcomes"] = [dict(success=True,steps=339,calls=21)] + [dict(success=False,steps=352,calls=21)]*3
        pool["x"][:] = 0.; pool["x"][0,0] = 1.; pool["x"][3,1] = 1.
        fitted, audit = fit_once([pool])
        actual = select_soft_utility(decision_list(), pool["base_scores"], pool["official_values"], 1, pool["x"], fitted)
        self.assertEqual(actual["selected_id"], 0)
        self.assertGreaterEqual(actual["scores"][0]-actual["scores"][3], .02-1e-12)
        self.assertEqual(audit["pairs"]["corrective"], 3)
        self.assertTrue(audit["complete_hard_accepted_failed_competitor_supervision"])

    def test_unreachable_third_competitor_excludes_complete_target_not_easy_pair(self):
        pool = train_pool()
        pool["base_scores"][2] = .2
        with self.assertRaises(NoFitError) as caught:
            fit_once([pool])
        audit = caught.exception.audit
        self.assertEqual(audit["skipped"]["rejected_corrective_candidates"], 1)
        row = audit["corrective_competition"][0]
        self.assertFalse(row["admitted"])
        self.assertTrue(row["whole_candidate_target_excluded"])
        failed = next(r for r in row["competitors"] if r["candidate_id"] == 2)
        self.assertEqual(failed["exclusion_reason"], "outside_fixed_cap")

    def test_unknown_competitor_is_not_negative_label(self):
        pool = train_pool(); pool["outcomes"][2] = None
        with self.assertRaises(NoFitError) as caught:
            fit_once([pool])
        self.assertEqual(caught.exception.audit["skipped"]["unknown_competitor_candidates"], 1)

    def test_unknown_fallback_and_all_failed_remain(self):
        correction = train_pool()
        fallback = train_pool((0,1,0,"clean")); fallback["reference_id"] = None
        fallback["outcomes"] = [None]*4
        failed = train_pool((0,2,0,"clean")); failed["outcomes"] = [dict(success=False,steps=352,calls=21)]*4
        _, audit = fit_once([correction, fallback, failed])
        self.assertEqual(audit["train_pools"], 3)
        self.assertEqual(audit["all_failed_pools_retained"], 1)
        self.assertEqual(audit["skipped"]["unknown_reference"], 1)
        self.assertFalse(audit["unknown_fallback_is_failure"])

    def test_insufficient_or_validation_or_changed_configuration_no_fit(self):
        with self.assertRaises(NoFitError):
            fit_once([train_pool(reference_success=True)])
        bad = train_pool(); bad["split"]="val"
        with self.assertRaises(NoFitError):
            fit_once([bad])
        with self.assertRaises(NoFitError):
            fit_once([train_pool()], steps=100)
        inactive = train_pool(); inactive["x"][:]=0.
        with self.assertRaises(NoFitError):
            fit_once([inactive])

    def test_existing_feasible_value_harm_is_not_erased(self):
        bad = train_pool(); bad["value_id"]=1
        with self.assertRaisesRegex(NoFitError,"zero reference already harms"):
            fit_once([bad])

    def test_fixed_cap_unreachable_is_not_hidden(self):
        bad = train_pool(); bad["base_scores"][0]=.2
        with self.assertRaisesRegex(NoFitError,"no admitted corrective"):
            fit_once([bad])

    def test_duplicate_pool_identity_no_fit(self):
        with self.assertRaisesRegex(NoFitError,"duplicate"):
            fit_once([train_pool(),train_pool()])


if __name__ == "__main__":
    unittest.main()

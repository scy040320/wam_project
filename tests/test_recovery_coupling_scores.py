"""Structural coupling checks, not experimental evidence of learned benefit.

The nonzero linear heads below are explicitly TEST DOUBLES.  Passing these
checks means the outcome-free contract can alter candidate-specific scores;
it does not mean trained V9 weights are nonzero or improve a real rollout.
"""
from dataclasses import replace
import unittest

import numpy as np

from wam_reranking.belief import DEFAULT_GRAPH, DependencyGraph
from wam_reranking.candidate_utility import CandidateUtilityModel
from wam_reranking.contracts import (
    AttributionOutput, BeliefFact, BeliefState, CandidateEffect, CoarseCause,
    ConsistencyFactor, EvidenceQuality, PREDICATES, Stage, TriValue,
)
from wam_reranking.evidence_residual import (
    CAUSE_FEATURE_NAMES, EvidenceResidualModel, candidate_backbone_features,
)
from wam_reranking.recovery_contract import (
    FEATURE_NAMES, SCHEMA, CurrentFactEvidence, ForecastFactEvent, prepare_recovery_context,
    select_recovery_candidate, trace_candidate_recovery,
)


class TestOnlyLinearRecoveryHead:
    """Untrained fixture proving wiring only; never serialize/deploy this head."""

    def __init__(self, feature_name, weight=1.0):
        self.weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        self.weights[FEATURE_NAMES.index(feature_name)] = weight

    def score(self, features):
        return float(np.asarray(features) @ self.weights)


def recorded_prior():
    belief = BeliefState(0, "fixture_target", "fixture_anchor", Stage.LIFT)
    for name in PREDICATES:
        belief.facts[name] = BeliefFact(
            TriValue.TRUE, .9, "observation", 2, ("actual_frame:" + name,))
    return belief


def attribution(cause):
    probabilities = {f.value: 1.0 for f in ConsistencyFactor}
    if cause is CoarseCause.OBJECT_SHIFT:
        probabilities[ConsistencyFactor.WORLD_STATE_CONSISTENT.value] = 0.
    if cause is CoarseCause.EXECUTION_CONTACT_DEVIATION:
        probabilities[ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value] = 0.
    if cause is CoarseCause.VISUAL_OCCLUSION:
        probabilities[ConsistencyFactor.OBSERVATION_RELIABLE.value] = 0.
    if cause is CoarseCause.UNKNOWN:
        probabilities[ConsistencyFactor.CAUSE_RESOLVED.value] = 0.
    return AttributionOutput(
        probabilities, {c.value: float(c is cause) for c in CoarseCause},
        cause, .9, 0., EvidenceQuality(
            cause is not CoarseCause.VISUAL_OCCLUSION, True, True),
        "fixture_prior_block",
        {name: TriValue.TRUE if probability else TriValue.FALSE
         for name, probability in probabilities.items()},
        {f.value: .9 for f in ConsistencyFactor})


def candidate(candidate_id, proposal, predicate="grasped", stage=Stage.APPROACH):
    # APPROACH deliberately avoids critical lift entry gating in score tests.
    # Neither a terminal proposal nor this fixture certifies physical recovery.
    return CandidateEffect(
        candidate_id, stage, {predicate: .85},
        {predicate: (TriValue.TRUE, proposal)}, .9, {})


class RecoveryCouplingScoreTests(unittest.TestCase):
    def setUp(self):
        weights = np.zeros(27, dtype=np.float64)
        # CandidateUtilityModel already adds official value to its residual.
        # Zero residual reproduces the exact value-only control here.
        self.backbone = CandidateUtilityModel(
            np.zeros(27), np.ones(27), weights, 0.)
        self.actions = [np.zeros((16, 7)), np.zeros((16, 7))]
        self.effects = [candidate(0, .9), candidate(1, .1)]
        self.values = [.5, .6]
        # This is a TEST-ONLY admission fixture, not a real passed data audit.
        self.test_only_audit = dict(
            schema=SCHEMA, recovery_training_ready=True,
            group_leakage=False, deployment_gt_leakage=False,
            role="unit_test_fixture_not_dataset_evidence")

    def select(self, cause, **kwargs):
        parameters = dict(
            prior=recorded_prior(), attribution=attribution(cause),
            observations=(), block_index=3, effects=self.effects,
            actions=self.actions, values=self.values, relation="inside",
            backbone=self.backbone,
            recovery_model=TestOnlyLinearRecoveryHead(
                "object_shift.grasped.terminal_proposal"),
            training_audit=self.test_only_audit,
        )
        parameters.update(kwargs)
        return select_recovery_candidate(**parameters)

    def test_all_five_routes_have_candidate_specific_not_set_constant_features(self):
        for cause in CoarseCause:
            with self.subTest(cause=cause.value):
                predicate = "grasped" if cause in (
                    CoarseCause.OBJECT_SHIFT,
                    CoarseCause.EXECUTION_CONTACT_DEVIATION) else "target_visible"
                context = prepare_recovery_context(
                    prior=recorded_prior(), attribution=attribution(cause),
                    block_index=3)
                traces = [trace_candidate_recovery(
                    context, candidate(i, proposal, predicate), self.actions[i])
                    for i, proposal in enumerate((.9, .1))]
                index = FEATURE_NAMES.index(
                    f"{cause.value}.{predicate}.terminal_proposal")
                self.assertGreater(traces[0].features[index], traces[1].features[index])
                self.assertFalse(np.array_equal(traces[0].features, traces[1].features))

    def test_recovery_proposals_do_not_secretly_change_same_candidate_backbone(self):
        original = candidate_backbone_features(self.effects[0], .5)
        changed = candidate_backbone_features(replace(
            self.effects[0], proposed_effects={"grasped": (TriValue.TRUE, .1)}), .5)
        np.testing.assert_array_equal(original, changed)
        self.assertEqual(self.backbone.score(original), self.backbone.score(changed))

    def test_test_double_score_is_exactly_shared_backbone_plus_recovery_term(self):
        result = self.select(CoarseCause.OBJECT_SHIFT)
        fixture = TestOnlyLinearRecoveryHead("object_shift.grasped.terminal_proposal")
        for effect, value, trace in zip(self.effects, self.values, result.traces):
            backbone_score = self.backbone.score(candidate_backbone_features(effect, value))
            correction = fixture.score(trace.features)
            self.assertAlmostEqual(result.scores[effect.candidate_id], backbone_score + correction)
        self.assertAlmostEqual(result.scores[0], 1.4)
        self.assertAlmostEqual(result.scores[1], .7)
        self.assertEqual(result.selected_candidate_id, 0)

    def test_masking_attribution_restores_same_backbone_scores_in_test_double(self):
        shifted = self.select(CoarseCause.OBJECT_SHIFT)
        masked = self.select(CoarseCause.NORMAL)
        self.assertEqual(shifted.selected_candidate_id, 0)
        self.assertEqual(masked.selected_candidate_id, 1)
        self.assertAlmostEqual(masked.scores[0], .5)
        self.assertAlmostEqual(masked.scores[1], .6)

    def test_no_dependency_graph_removes_descendant_recovery_score_in_test_double(self):
        full = self.select(CoarseCause.OBJECT_SHIFT, graph=DEFAULT_GRAPH)
        flat = self.select(CoarseCause.OBJECT_SHIFT, graph=DependencyGraph(()))
        self.assertEqual(full.selected_candidate_id, 0)
        self.assertEqual(flat.selected_candidate_id, 1)
        self.assertAlmostEqual(flat.scores[0], .5)
        self.assertAlmostEqual(flat.scores[1], .6)
        propagated = FEATURE_NAMES.index("object_shift.grasped.propagated_need")
        self.assertGreater(full.traces[0].features[propagated], 0.)
        self.assertEqual(flat.traces[0].features[propagated], 0.)

    def test_wrong_cause_does_not_activate_world_test_double(self):
        # A cause permutation affects interaction features, not backbone input.
        wrong = self.select(CoarseCause.EXECUTION_CONTACT_DEVIATION)
        self.assertEqual(wrong.selected_candidate_id, 1)
        self.assertAlmostEqual(wrong.scores[0], .5)
        self.assertAlmostEqual(wrong.scores[1], .6)

    def test_zero_recovery_weights_are_allowed_and_do_not_prove_usefulness(self):
        zero = TestOnlyLinearRecoveryHead("object_shift.grasped.terminal_proposal", 0.)
        result = self.select(CoarseCause.OBJECT_SHIFT, recovery_model=zero)
        self.assertEqual(result.selected_candidate_id, 1)
        self.assertAlmostEqual(result.scores[0], .5)
        self.assertAlmostEqual(result.scores[1], .6)

    def test_existing_bounded_residual_gain_zero_can_erase_nonzero_features(self):
        width = len(CAUSE_FEATURE_NAMES)
        residual = EvidenceResidualModel(
            np.ones(width), np.ones(width), np.ones(width), gain=0., score_bound=1.)
        self.assertEqual(residual.score(np.ones(width)), 0.)

    def test_existing_bounded_residual_outside_support_can_erase_nonzero_weights(self):
        width = len(CAUSE_FEATURE_NAMES)
        residual = EvidenceResidualModel(
            np.ones(width), np.ones(width), np.ones(width), gain=1., score_bound=1.)
        self.assertEqual(residual.score(np.full(width, 2.)), 0.)

    def test_test_double_cannot_override_directly_observed_false_grasp(self):
        prior = recorded_prior()
        prior.facts["grasped"] = BeliefFact(
            TriValue.FALSE, .9, "observation", 2, ("actual_drop_seen",))
        result = self.select(
            CoarseCause.NORMAL, prior=prior,
            effects=[candidate(0, .9, stage=Stage.LIFT), candidate(1, .1, stage=Stage.LIFT)],
            recovery_model=TestOnlyLinearRecoveryHead("normal.grasped.terminal_proposal", 100.))
        self.assertIsNone(result.selected_candidate_id)
        self.assertFalse(result.scores)
        self.assertTrue(all(t.hard_reasons for t in result.traces))

    def test_unknown_confidence_decay_is_not_permission_to_bypass_current_evidence(self):
        prior = recorded_prior()
        prior.facts["grasped"] = BeliefFact(
            TriValue.UNKNOWN, .001, "attribution", 2, ("uncertain_grasp",))
        result = self.select(
            CoarseCause.UNKNOWN, prior=prior,
            effects=[candidate(0, .9, stage=Stage.LIFT), candidate(1, .1, stage=Stage.LIFT)],
            recovery_model=TestOnlyLinearRecoveryHead("unknown.grasped.terminal_proposal", 100.))
        self.assertIsNone(result.selected_candidate_id)
        self.assertTrue(all("grasped=unknown_requires_current_evidence" in t.hard_reasons
                            for t in result.traces))

    def test_unreliable_primary_cannot_refresh_fact_using_other_sensor_quality(self):
        context = prepare_recovery_context(
            prior=recorded_prior(), attribution=attribution(CoarseCause.VISUAL_OCCLUSION),
            block_index=3, observations=[CurrentFactEvidence(
                "target_pose_current", TriValue.TRUE, .99,
                "unreliable_primary", view="primary")])
        self.assertIs(context.belief.facts["target_pose_current"].value, TriValue.UNKNOWN)
        self.assertNotIn("unreliable_primary", context.belief.facts["target_pose_current"].evidence_ids)

    def test_candidate_prediction_cannot_be_imported_as_current_observation(self):
        with self.assertRaises(ValueError):
            CurrentFactEvidence("grasped", TriValue.TRUE, .99,
                                "candidate_terminal", source="candidate_prediction")

    def test_articulated_same_stage_candidates_keep_relation_specific_coupling(self):
        effects = [replace(candidate(i, .9), evidence={"relation_score_delta": delta})
                   for i, delta in enumerate((.4, .05))]
        result = self.select(
            CoarseCause.OBJECT_SHIFT, effects=effects, relation="open",
            observations=[CurrentFactEvidence(
                "target_pose_current", TriValue.TRUE, .9, "current_joint_image", view="primary")],
            recovery_model=TestOnlyLinearRecoveryHead(
                "object_shift.visual.relation_score_delta.value"))
        self.assertGreater(result.traces[0].features[FEATURE_NAMES.index(
            "object_shift.visual.relation_score_delta.value")], result.traces[1].features[
                FEATURE_NAMES.index("object_shift.visual.relation_score_delta.value")])
        # Frozen value-only fixture has no learned visual weights: isolate the
        # recovery correction rather than credit ordinary candidate ranking.
        self.assertAlmostEqual(result.scores[0], .9)
        self.assertAlmostEqual(result.scores[1], .65)
        self.assertEqual(result.selected_candidate_id, 0)
        self.assertIsNone(result.traces[0].facts["grasped"]["required_at_step"])

    def test_command_order_affects_cause_score_without_certifying_physical_grasp(self):
        first, second = [action.copy() for action in self.actions]
        first[2:, 6] = 1.; first[8:, 2] = .1
        second[8:, 6] = 1.; second[2:, 2] = .1
        result = self.select(
            CoarseCause.OBJECT_SHIFT, actions=[first, second],
            recovery_model=TestOnlyLinearRecoveryHead("object_shift.command.close_before_upward"))
        self.assertEqual(result.selected_candidate_id, 0)
        self.assertAlmostEqual(result.scores[0], 1.5)
        self.assertAlmostEqual(result.scores[1], .6)
        self.assertEqual(result.traces[0].facts["grasped"]["value"], "unknown")
        self.assertTrue(result.traces[0].command_order["commanded_close_is_not_grasp_evidence"])

    def test_single_entry_receives_per_candidate_intermediate_soft_forecasts(self):
        actions = [action.copy() for action in self.actions]
        for action in actions:
            action[8:, 2] = .1
        result = self.select(
            CoarseCause.NORMAL,
            effects=[candidate(0, .9, stage=Stage.LIFT), candidate(1, .1, stage=Stage.LIFT)],
            actions=actions,
            observations=[CurrentFactEvidence(
                "grasped", TriValue.TRUE, .9, "current_held_object", view="wrist")],
            forecasts=[
                [ForecastFactEvent("grasped", TriValue.TRUE, .9, 4,
                                   "candidate0_predicted_frame4", "predicted_intermediate")],
                [ForecastFactEvent("grasped", TriValue.TRUE, .1, 4,
                                   "candidate1_predicted_frame4", "predicted_intermediate")],
            ],
            recovery_model=TestOnlyLinearRecoveryHead("normal.grasped.ordered_forecast_support"))
        self.assertAlmostEqual(result.scores[0], 1.4)
        self.assertAlmostEqual(result.scores[1], .7)
        self.assertEqual(result.selected_candidate_id, 0)
        self.assertEqual(tuple(result.belief_snapshot["facts"]["grasped"]["evidence_ids"]),
                         ("current_held_object",))


if __name__ == "__main__":
    unittest.main()

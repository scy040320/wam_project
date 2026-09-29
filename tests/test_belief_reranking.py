from __future__ import annotations

import unittest
import numpy as np

from wam_reranking import (
    ActionRefinementRequest, ActionRefinementResult, AttributionOutput, BeliefFact,
    CoarseCause, ConsistencyFactor, DEFAULT_GRAPH, DependencyGraph, EvidenceQuality,
    HeadThreshold, ScoreWeights, Stage, TriValue, build_attribution_output, evaluate_candidate, initial_belief,
    parse_candidate_effect, select_candidate, update_belief, evaluate_clean_pair,
    EvidenceRoute, LEARNED_FACTOR_NAMES, canonical_target_prompt,
    evaluate_command_execution, pool_target_residual, route_evidence,
    structural_residual_grid, target_semantic_features,
    cross_task_factor_contrastive_loss,
    CandidateVisualEvidence, SelectorMode, build_candidate_visual_evidence,
    candidate_effect_record,
    select_hard_gate_value_tiebreak, select_value_only,
)
from wam_reranking.refiner import apply_bounded_residual


def attribution(*, observation=0.9, world=0.9, execution=0.9, stage=0.9,
                resolved=0.9, cause=CoarseCause.NORMAL, conflict=False):
    classes = {item.value: 0.025 for item in CoarseCause}
    classes[cause.value] = 0.9
    return AttributionOutput({
        ConsistencyFactor.OBSERVATION_RELIABLE.value: observation,
        ConsistencyFactor.WORLD_STATE_CONSISTENT.value: world,
        ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value: execution,
        ConsistencyFactor.TASK_STAGE_CONSISTENT.value: stage,
        ConsistencyFactor.CAUSE_RESOLVED.value: resolved,
    }, classes, cause, 0.9, 0.4, EvidenceQuality(True, True, True, conflict), "block_002")


def actions_for(stage: Stage) -> np.ndarray:
    actions = np.zeros((16, 7), dtype=np.float32)
    if stage is Stage.APPROACH:
        actions[:, 0] = 0.02
    elif stage is Stage.GRASP:
        actions[8:, 6] = -1.0
    elif stage is Stage.LIFT:
        actions[4:, 6] = -1.0
        actions[6:, 2] = 0.02
    elif stage is Stage.TRANSPORT:
        actions[:, 6] = -1.0
        actions[:, 0] = 0.02
    elif stage is Stage.PLACE:
        actions[:8, 6] = -1.0
        actions[8:, 6] = 1.0
    return actions


WEIGHTS = ScoreWeights(0.3, 0.2, 0.5, 0.1, calibrated=False)


def tri(label: str, confidence: float = 0.9):
    rest = (1.0 - confidence) / 2.0
    return {name: confidence if name == label else rest for name in ("no", "yes", "uncertain")}


def calibrated_output(*, world="yes", observation="yes", execution="yes", stage="yes", resolved="yes"):
    factors = {
        "primary_view_reliable": tri("yes"), "wrist_view_reliable": tri("yes"),
        "action_record_reliable": tri("yes"), "observation_sufficient": tri(observation),
        "world_state_consistent": tri(world), "execution_contact_consistent": tri(execution),
        "task_progress_consistent": tri(stage), "cause_resolved": tri(resolved),
    }
    thresholds = {name: HeadThreshold(0.6) for name in factors}
    thresholds["coarse"] = HeadThreshold(0.6)
    return build_attribution_output(
        coarse_probs={"normal": 0.02, "visual_occlusion": 0.02, "object_shift": 0.92,
                      "execution_contact_deviation": 0.02, "unknown": 0.02},
        factor_distributions=factors, thresholds=thresholds, source_block_id="block_002",
    )


class AttributionContractTests(unittest.TestCase):
    def test_unresolved_projects_to_unknown(self):
        with self.assertRaises(ValueError):
            attribution(resolved=0.2, cause=CoarseCause.OBJECT_SHIFT)

    def test_cross_view_conflict_projects_to_unknown(self):
        with self.assertRaises(ValueError):
            attribution(conflict=True, cause=CoarseCause.OBJECT_SHIFT)

    def test_calibrated_tri_state_adapter_preserves_unknown(self):
        result = calibrated_output(world="uncertain", resolved="uncertain")
        self.assertEqual(result.factor_state(ConsistencyFactor.WORLD_STATE_CONSISTENT), TriValue.UNKNOWN)
        self.assertEqual(result.projected_cause, CoarseCause.UNKNOWN)


class BeliefTests(unittest.TestCase):
    def test_graph_is_acyclic(self):
        self.assertIn("placed", DEFAULT_GRAPH.descendants_with_paths("target_pose_current"))

    def test_cycle_is_rejected(self):
        with self.assertRaises(ValueError):
            DependencyGraph((("grasped", "lifted"), ("lifted", "grasped")))

    def test_consistent_block_preserves_values(self):
        belief = initial_belief(0)
        before = belief.facts["target_pose_current"]
        update_belief(belief, attribution(), 2)
        after = belief.facts["target_pose_current"]
        self.assertEqual(before.value, after.value)
        self.assertGreater(after.confidence, before.confidence)

    def test_observation_failure_does_not_revoke_grasp(self):
        belief = initial_belief(0)
        belief.facts["grasped"] = BeliefFact(TriValue.TRUE, 0.8, "observation", 1, ("contact",))
        update_belief(belief, attribution(observation=0.1, cause=CoarseCause.VISUAL_OCCLUSION), 2)
        self.assertEqual(belief.facts["target_visible"].value, TriValue.UNKNOWN)
        self.assertEqual(belief.facts["grasped"].value, TriValue.TRUE)

    def test_world_shift_minimally_invalidates_pose(self):
        belief = initial_belief(0)
        update_belief(belief, attribution(world=0.1, cause=CoarseCause.OBJECT_SHIFT), 2)
        self.assertEqual(belief.facts["target_visible"].value, TriValue.TRUE)
        self.assertEqual(belief.facts["target_pose_current"].value, TriValue.FALSE)
        self.assertEqual(belief.facts["target_reachable"].value, TriValue.UNKNOWN)

    def test_multiple_inconsistencies_can_coexist(self):
        belief = initial_belief(1)
        update_belief(belief, attribution(world=0.1, execution=0.1, cause=CoarseCause.OBJECT_SHIFT), 2)
        self.assertEqual(belief.facts["target_pose_current"].value, TriValue.FALSE)
        self.assertEqual(belief.facts["execution_consistent"].value, TriValue.FALSE)

    def test_stage_inconsistency_marks_stage_uncertain(self):
        belief = initial_belief(2)
        belief.task_stage = Stage.LIFT
        update_belief(belief, attribution(stage=0.1, cause=CoarseCause.UNKNOWN, resolved=0.1), 3)
        self.assertEqual(belief.task_stage, Stage.UNCERTAIN)

    def test_uncertain_world_state_does_not_become_false(self):
        belief = initial_belief(0)
        update_belief(belief, calibrated_output(world="uncertain", resolved="uncertain"), 2)
        self.assertEqual(belief.facts["target_pose_current"].value, TriValue.UNKNOWN)

    def test_calibrated_world_inconsistency_invalidates_pose(self):
        belief = initial_belief(0)
        update_belief(belief, calibrated_output(world="no"), 2)
        self.assertEqual(belief.facts["target_pose_current"].value, TriValue.FALSE)


class CandidateTests(unittest.TestCase):
    def test_value_only_matches_official_argmax(self):
        selected = select_value_only([0.2, 0.8, 0.5])
        self.assertEqual(selected.selected_candidate_id, 1)
        self.assertIsNone(selected.fallback)

    def test_hard_gate_uses_value_only_among_feasible_candidates(self):
        attr = attribution(world=0.1, cause=CoarseCause.OBJECT_SHIFT)
        result = select_hard_gate_value_tiebreak(
            mode=SelectorMode.ORACLE_HARD_GATE,
            belief=initial_belief(0),
            attribution=attr,
            block_index=2,
            candidate_actions=[actions_for(Stage.GRASP), actions_for(Stage.APPROACH)],
            official_values=[0.99, 0.1],
        )
        self.assertFalse(result.decisions[0].accepted)
        self.assertTrue(result.decisions[1].accepted)
        self.assertEqual(result.selected_candidate_id, 1)
        self.assertEqual(result.belief_snapshot["facts"]["target_pose_current"]["value"], "true")
        self.assertEqual(result.belief_snapshot["facts"]["target_reachable"]["value"], "unknown")

    def test_hard_gate_all_rejected_exposes_fallback(self):
        attr = attribution(observation=0.1, world=0.1, cause=CoarseCause.VISUAL_OCCLUSION)
        result = select_hard_gate_value_tiebreak(
            mode=SelectorMode.LEARNED_HARD_GATE,
            belief=initial_belief(0),
            attribution=attr,
            block_index=2,
            candidate_actions=[actions_for(Stage.GRASP)],
            official_values=[0.9],
        )
        self.assertIsNone(result.selected_candidate_id)
        self.assertEqual(result.fallback, "reobserve")

    def test_rule_parser_stages(self):
        for stage in (Stage.APPROACH, Stage.GRASP, Stage.LIFT, Stage.TRANSPORT, Stage.PLACE):
            self.assertEqual(parse_candidate_effect(0, actions_for(stage)).stage, stage)

    def test_value_orders_equally_feasible_candidates(self):
        belief = initial_belief(0)
        effect = parse_candidate_effect(0, actions_for(Stage.APPROACH))
        low = evaluate_candidate(belief, attribution(), effect, 0.3, WEIGHTS, allow_uncalibrated=True)
        high = evaluate_candidate(belief, attribution(), effect, 0.8, WEIGHTS, allow_uncalibrated=True)
        self.assertGreater(high.total_score, low.total_score)

    def test_shift_rejects_old_pose_candidate(self):
        belief = initial_belief(0)
        attr = attribution(world=0.1, cause=CoarseCause.OBJECT_SHIFT)
        update_belief(belief, attr, 2)
        effect = parse_candidate_effect(0, actions_for(Stage.GRASP))
        decision = evaluate_candidate(belief, attr, effect, 0.99, WEIGHTS, allow_uncalibrated=True)
        self.assertFalse(decision.accepted)

    def test_attribution_source_survives_propagation_confidence_decay(self):
        attr = attribution(world=0.1, cause=CoarseCause.OBJECT_SHIFT)
        result = select_hard_gate_value_tiebreak(
            mode=SelectorMode.ORACLE_HARD_GATE,
            belief=initial_belief(0), attribution=attr, block_index=2,
            candidate_actions=[actions_for(Stage.TRANSPORT)], official_values=[0.99],
        )
        self.assertFalse(result.decisions[0].accepted)
        self.assertTrue(any("unknown_from_world_state_consistent" in reason
                            for reason in result.decisions[0].rejection_reasons))

    def test_visual_candidate_violation_rejects_unsupported_transport(self):
        visual = CandidateVisualEvidence(
            target_motion=0.1, anchor_motion=0.0,
            target_anchor_distance_before=0.5, target_anchor_distance_after=0.4,
            target_anchor_affinity_before=0.0, target_anchor_affinity_after=0.1,
            target_gripper_distance_before=0.6, target_gripper_distance_after=0.6,
            target_gripper_affinity_before=0.0, target_gripper_affinity_after=0.0,
            relation_progress=0.1, visibility_confidence=0.9, cross_view_agreement=0.9,
            relation_confidence=0.9, contact_confidence=0.9,
        )
        effect = parse_candidate_effect(0, actions_for(Stage.TRANSPORT), visual_evidence=visual)
        decision = evaluate_candidate(
            initial_belief(0), attribution(), effect, 0.99, WEIGHTS, allow_uncalibrated=True
        )
        self.assertFalse(decision.accepted)
        self.assertTrue(any("predicted_target_gripper_contact_missing" in reason
                            for reason in decision.rejection_reasons))

    def test_execution_deviation_can_preserve_derived_contact_chain(self):
        visual = CandidateVisualEvidence(
            target_motion=0.1, anchor_motion=0.0,
            target_anchor_distance_before=0.4, target_anchor_distance_after=0.43,
            target_anchor_affinity_before=0.2, target_anchor_affinity_after=0.18,
            target_gripper_distance_before=0.2, target_gripper_distance_after=0.16,
            target_gripper_affinity_before=0.2, target_gripper_affinity_after=0.18,
            relation_progress=-0.03, visibility_confidence=0.9, cross_view_agreement=0.9,
            relation_confidence=0.9, contact_confidence=0.9,
        )
        attr = attribution(execution=0.1, cause=CoarseCause.EXECUTION_CONTACT_DEVIATION)
        belief = initial_belief(0)
        update_belief(belief, attr, 2)
        effect = parse_candidate_effect(0, actions_for(Stage.PLACE), visual_evidence=visual)
        decision = evaluate_candidate(belief, attr, effect, 0.8, WEIGHTS, allow_uncalibrated=True)
        self.assertTrue(decision.accepted)

    def test_world_shift_still_blocks_contact_chain_override(self):
        visual = CandidateVisualEvidence(
            target_motion=0.1, anchor_motion=0.0,
            target_anchor_distance_before=0.4, target_anchor_distance_after=0.35,
            target_anchor_affinity_before=0.2, target_anchor_affinity_after=0.25,
            target_gripper_distance_before=0.2, target_gripper_distance_after=0.16,
            target_gripper_affinity_before=0.2, target_gripper_affinity_after=0.2,
            relation_progress=0.05, visibility_confidence=0.9, cross_view_agreement=0.9,
            relation_confidence=0.9, contact_confidence=0.9,
        )
        attr = attribution(world=0.1, cause=CoarseCause.OBJECT_SHIFT)
        belief = initial_belief(0)
        update_belief(belief, attr, 2)
        effect = parse_candidate_effect(0, actions_for(Stage.PLACE), visual_evidence=visual)
        decision = evaluate_candidate(belief, attr, effect, 0.8, WEIGHTS, allow_uncalibrated=True)
        self.assertFalse(decision.accepted)

    def test_small_or_single_view_relation_regression_is_not_hard_rejection(self):
        visual = CandidateVisualEvidence(
            target_motion=0.1, anchor_motion=0.0,
            target_anchor_distance_before=0.4, target_anchor_distance_after=0.44,
            target_anchor_affinity_before=0.2, target_anchor_affinity_after=0.15,
            target_gripper_distance_before=0.2, target_gripper_distance_after=0.18,
            target_gripper_affinity_before=0.2, target_gripper_affinity_after=0.18,
            relation_progress=-0.04, visibility_confidence=0.9, cross_view_agreement=0.45,
            relation_confidence=0.9, contact_confidence=0.9,
        )
        effect = parse_candidate_effect(0, actions_for(Stage.PLACE), visual_evidence=visual)
        self.assertNotIn("predicted_target_anchor_relation_regresses", effect.hard_violations)

    def test_visual_relation_maps_distinguish_candidate_progress(self):
        def blob(x, y):
            out = np.zeros((32, 32), np.float32)
            out[y-1:y+2, x-1:x+2] = 1.0
            return out
        current_subject = blob(6, 16); current_anchor = blob(25, 16); gripper = blob(7, 16)
        improved = build_candidate_visual_evidence(
            current_subject_maps=(current_subject, current_subject),
            predicted_subject_maps=(blob(20, 16), blob(20, 16)),
            current_anchor_maps=(current_anchor, current_anchor),
            predicted_anchor_maps=(current_anchor, current_anchor),
            current_gripper_maps=(gripper, gripper),
            predicted_gripper_maps=(blob(20, 16), blob(20, 16)), relation="toward",
        )
        regressed = build_candidate_visual_evidence(
            current_subject_maps=(current_subject, current_subject),
            predicted_subject_maps=(blob(3, 16), blob(3, 16)),
            current_anchor_maps=(current_anchor, current_anchor),
            predicted_anchor_maps=(current_anchor, current_anchor),
            current_gripper_maps=(gripper, gripper),
            predicted_gripper_maps=(blob(3, 16), blob(3, 16)), relation="toward",
        )
        self.assertGreater(improved.relation_progress, 0.0)
        self.assertLess(regressed.relation_progress, 0.0)
        self.assertGreater(improved.relation_score_after, improved.relation_score_before)
        self.assertGreater(improved.target_delta_x, 0.0)

    def test_directional_relations_use_signed_candidate_motion(self):
        def blob(x, y):
            out = np.zeros((32, 32), np.float32)
            out[y-1:y+2, x-1:x+2] = 1.0
            return out
        anchor = blob(16, 16)
        subject = blob(16, 16)
        gripper = blob(16, 16)
        left = build_candidate_visual_evidence(
            current_subject_maps=(subject, subject),
            predicted_subject_maps=(blob(8, 16), blob(8, 16)),
            current_anchor_maps=(anchor, anchor), predicted_anchor_maps=(anchor, anchor),
            current_gripper_maps=(gripper, gripper), predicted_gripper_maps=(gripper, gripper),
            relation="left_of",
        )
        under = build_candidate_visual_evidence(
            current_subject_maps=(subject, subject),
            predicted_subject_maps=(blob(16, 24), blob(16, 24)),
            current_anchor_maps=(anchor, anchor), predicted_anchor_maps=(anchor, anchor),
            current_gripper_maps=(gripper, gripper), predicted_gripper_maps=(gripper, gripper),
            relation="under",
        )
        self.assertGreater(left.relation_score_after, left.relation_score_before)
        self.assertGreater(left.relation_progress, 0.0)
        self.assertGreater(under.relation_score_after, under.relation_score_before)
        self.assertGreater(under.relation_progress, 0.0)

    def test_candidate_specific_relation_changes_affect_soft_ranking(self):
        common = dict(
            target_motion=0.2, anchor_motion=0.0,
            target_anchor_distance_before=0.6, target_anchor_distance_after=0.3,
            target_anchor_affinity_before=0.0, target_anchor_affinity_after=0.2,
            target_gripper_distance_before=0.3, target_gripper_distance_after=0.2,
            target_gripper_affinity_before=0.1, target_gripper_affinity_after=0.2,
            relation_progress=0.3, visibility_confidence=0.9, cross_view_agreement=0.9,
            relation_confidence=0.9, contact_confidence=0.9,
            relation_score_before=0.2, grasp_support_before=0.2, grasp_support_after=0.5,
        )
        improved = CandidateVisualEvidence(**common, relation_score_after=0.8, target_delta_x=0.3)
        regressed = CandidateVisualEvidence(**common, relation_score_after=0.1, target_delta_x=-0.2)
        attr = attribution(world=0.1, cause=CoarseCause.OBJECT_SHIFT)
        weights = ScoreWeights(0.8, 0.6, 0.4, 0.1, calibrated=True)
        good = evaluate_candidate(
            initial_belief(0), attr,
            parse_candidate_effect(0, actions_for(Stage.APPROACH), visual_evidence=improved),
            0.5, weights,
        )
        bad = evaluate_candidate(
            initial_belief(0), attr,
            parse_candidate_effect(1, actions_for(Stage.APPROACH), visual_evidence=regressed),
            0.5, weights,
        )
        self.assertGreater(good.components["attribution_compatibility"], bad.components["attribution_compatibility"])
        self.assertGreater(good.components["progress"], bad.components["progress"])
        self.assertGreater(good.total_score, bad.total_score)

    def test_trajectory_risk_distinguishes_smooth_and_oscillatory_actions(self):
        smooth = actions_for(Stage.APPROACH)
        oscillatory = smooth.copy()
        oscillatory[:, 0] = np.where(np.arange(16) % 2 == 0, 0.8, -0.8)
        smooth_effect = parse_candidate_effect(0, smooth)
        risky_effect = parse_candidate_effect(1, oscillatory)
        self.assertGreater(risky_effect.evidence["trajectory_risk"], smooth_effect.evidence["trajectory_risk"])
        self.assertGreater(risky_effect.evidence["trajectory_max_jerk"], smooth_effect.evidence["trajectory_max_jerk"])

    def test_candidate_effect_record_has_four_explicit_evidence_groups(self):
        visual = CandidateVisualEvidence(
            target_motion=0.2, anchor_motion=0.01,
            target_anchor_distance_before=0.5, target_anchor_distance_after=0.2,
            target_anchor_affinity_before=0.1, target_anchor_affinity_after=0.4,
            target_gripper_distance_before=0.4, target_gripper_distance_after=0.1,
            target_gripper_affinity_before=0.1, target_gripper_affinity_after=0.5,
            relation_progress=0.4, visibility_confidence=0.9,
            cross_view_agreement=0.8, relation_confidence=0.85,
            contact_confidence=0.75, target_delta_x=0.1, target_delta_y=-0.2,
            relation_score_before=0.2, relation_score_after=0.8,
            grasp_support_before=0.1, grasp_support_after=0.7,
        )
        record = candidate_effect_record(
            parse_candidate_effect(3, actions_for(Stage.GRASP), visual_evidence=visual)
        )
        self.assertEqual(record["candidate_id"], 3)
        self.assertEqual(
            set(record) & {"target_displacement", "target_anchor_relation", "grasp_release", "trajectory"},
            {"target_displacement", "target_anchor_relation", "grasp_release", "trajectory"},
        )
        self.assertAlmostEqual(record["target_displacement"]["magnitude"], np.hypot(0.1, -0.2))
        self.assertAlmostEqual(record["target_anchor_relation"]["score_delta"], 0.6)
        self.assertGreaterEqual(record["grasp_release"]["predicted_grasp_support"], 0.0)
        self.assertGreaterEqual(record["trajectory"]["risk"], 0.0)

    def test_value_cannot_override_hard_violation(self):
        belief = initial_belief(0)
        belief.facts["grasped"] = BeliefFact(TriValue.FALSE, 1.0, "attribution", 2, ("failure",))
        attr = attribution(execution=0.1, cause=CoarseCause.EXECUTION_CONTACT_DEVIATION)
        lift = parse_candidate_effect(0, actions_for(Stage.LIFT))
        approach = parse_candidate_effect(1, actions_for(Stage.APPROACH))
        lift_d = evaluate_candidate(belief, attr, lift, 1.0, WEIGHTS, allow_uncalibrated=True)
        app_d = evaluate_candidate(belief, attr, approach, 0.1, WEIGHTS, allow_uncalibrated=True)
        chosen, fallback = select_candidate([lift_d, app_d], attr)
        self.assertIsNone(fallback)
        self.assertEqual(chosen.candidate_id, 1)

    def test_uncalibrated_weights_block_deployment(self):
        with self.assertRaises(RuntimeError):
            evaluate_candidate(initial_belief(0), attribution(), parse_candidate_effect(0, actions_for(Stage.APPROACH)), 0.5, WEIGHTS)

    def test_all_rejected_returns_reobserve_for_bad_observation(self):
        chosen, fallback = select_candidate([], attribution(observation=0.1, cause=CoarseCause.VISUAL_OCCLUSION))
        self.assertIsNone(chosen)
        self.assertEqual(fallback, "reobserve")

    def test_query_cost_is_auditable_score_component(self):
        belief = initial_belief(0)
        effect = parse_candidate_effect(0, actions_for(Stage.APPROACH))
        weights = ScoreWeights(0.3, 0.2, 0.5, 0.1, calibrated=True, requery_cost=0.4)
        cheap = evaluate_candidate(belief, attribution(), effect, 0.5, weights, query_cost=0.0)
        costly = evaluate_candidate(belief, attribution(), effect, 0.5, weights, query_cost=1.0)
        self.assertAlmostEqual(cheap.total_score - costly.total_score, 0.4)
        self.assertEqual(costly.components["query_cost"], 1.0)


class RefinerContractTests(unittest.TestCase):
    def test_bounded_residual_changes_actions(self):
        request = ActionRefinementRequest(np.zeros((16, 7), dtype=np.float32), {}, attribution())
        result = ActionRefinementResult(np.full((16, 7), 0.01, dtype=np.float32), 0.8, 0.02)
        self.assertTrue(np.allclose(apply_bounded_residual(request, result), 0.01))

    def test_residual_bound_is_enforced(self):
        with self.assertRaises(ValueError):
            ActionRefinementResult(np.ones((16, 7)), 0.8, 0.1)


def exact_clean_pair():
    image = {"mean_abs": 0.0, "p95_abs": 0.0, "fraction_gt5": 0.0, "psnr_db": 99.0}
    return {
        "sim_state": {
            "qpos_max_abs": 0.0, "qpos_p95_abs": 0.0,
            "robot_gripper_qvel_max_abs": 0.0, "target_qvel_max_abs": 0.0,
            "other_qvel_max_abs": 0.0, "other_qvel_p95_abs": 0.0,
        },
        "primary": image, "wrist": image, "proprio_max_abs": 0.0,
    }


class PairedCounterfactualAuditTests(unittest.TestCase):
    def test_cached_query_difference_is_diagnostic_only(self):
        result = evaluate_clean_pair(
            exact_clean_pair(),
            cached_observation_diagnostics={"proprio_max_abs": 0.001368},
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.cached_observation_diagnostics["proprio_max_abs"], 0.001368)

    def test_clean_pair_proprio_threshold_remains_hard(self):
        pair = exact_clean_pair()
        pair["proprio_max_abs"] = 0.001001
        result = evaluate_clean_pair(pair)
        self.assertFalse(result.passed)
        self.assertIn("proprio_max_abs", result.failures)

    def test_clean_pair_image_threshold_remains_hard(self):
        pair = exact_clean_pair()
        pair["wrist"] = dict(pair["wrist"], mean_abs=2.01)
        result = evaluate_clean_pair(pair)
        self.assertFalse(result.passed)
        self.assertIn("wrist.mean_abs", result.failures)


class EvidenceRoutingTests(unittest.TestCase):
    def factors(self, **overrides):
        values = {name: 0.1 for name in LEARNED_FACTOR_NAMES}
        values.update(overrides)
        return values

    def test_missing_action_record_is_a_hard_unknown_rule(self):
        result = route_evidence(
            action_record_available=False,
            projected_cause=CoarseCause.NORMAL,
            learned_factor_probs=self.factors(),
        )
        self.assertEqual(result.route, EvidenceRoute.ACTION_RECORD_UNRELIABLE)
        self.assertEqual(result.projected_cause, CoarseCause.UNKNOWN)
        self.assertTrue(result.hard_rule_applied)

    def test_unknown_without_supported_factor_abstains(self):
        result = route_evidence(
            action_record_available=True,
            projected_cause=CoarseCause.UNKNOWN,
            learned_factor_probs=self.factors(),
        )
        self.assertEqual(result.route, EvidenceRoute.CAUSE_UNRESOLVED)
        self.assertFalse(result.hard_rule_applied)

    def test_supported_factor_preserves_resolved_cause(self):
        result = route_evidence(
            action_record_available=True,
            projected_cause=CoarseCause.OBJECT_SHIFT,
            learned_factor_probs=self.factors(object_or_environment_state_changed=0.8),
        )
        self.assertEqual(result.route, EvidenceRoute.RESOLVED)
        self.assertEqual(result.projected_cause, CoarseCause.OBJECT_SHIFT)

    def test_requested_applied_mismatch_is_auditable_hard_route(self):
        requested = np.zeros((16, 7), dtype=np.float32)
        applied = requested.copy()
        applied[4:8, 0] = 0.12
        command = evaluate_command_execution(requested, applied)
        result = route_evidence(
            action_record_available=True,
            projected_cause=CoarseCause.NORMAL,
            learned_factor_probs=self.factors(),
            command_execution=command,
        )
        self.assertTrue(command.deviated)
        self.assertEqual(result.route, EvidenceRoute.COMMAND_EXECUTION_DEVIATION)
        self.assertEqual(result.projected_cause, CoarseCause.EXECUTION_CONTACT_DEVIATION)
        self.assertTrue(result.hard_rule_applied)

    def test_equal_requested_applied_actions_do_not_claim_contact_failure(self):
        actions = np.zeros((16, 7), dtype=np.float32)
        command = evaluate_command_execution(actions, actions)
        result = route_evidence(
            action_record_available=True,
            projected_cause=CoarseCause.NORMAL,
            learned_factor_probs=self.factors(),
            command_execution=command,
        )
        self.assertFalse(command.deviated)
        self.assertEqual(result.route, EvidenceRoute.RESOLVED)
        self.assertEqual(result.projected_cause, CoarseCause.NORMAL)

    def test_missing_action_record_dominates_command_route(self):
        requested = np.zeros((16, 7), dtype=np.float32)
        applied = requested.copy()
        applied[:, 1] = 0.1
        result = route_evidence(
            action_record_available=False,
            projected_cause=CoarseCause.NORMAL,
            learned_factor_probs=self.factors(),
            command_execution=evaluate_command_execution(requested, applied),
        )
        self.assertEqual(result.route, EvidenceRoute.ACTION_RECORD_UNRELIABLE)
        self.assertEqual(result.projected_cause, CoarseCause.UNKNOWN)


class TargetConditionedResidualTests(unittest.TestCase):
    def test_target_pooling_separates_target_change_from_background(self):
        residual = np.zeros((4, 4), dtype=np.float32)
        residual[1:3, 1:3] = 0.8
        relevance = np.zeros((4, 4), dtype=np.float32)
        relevance[1:3, 1:3] = 1.0
        result = pool_target_residual(
            residual, relevance, relevance,
            prefix="visual.target_primary", prompt="ketchup_1",
        )
        values = dict(zip(result.names, result.values.tolist()))
        self.assertGreater(values["visual.target_primary.weighted_residual_mean"], 0.79)
        self.assertGreater(values["visual.target_primary.target_minus_background"], 0.79)
        self.assertEqual(result.prompt, "ketchup bottle")

    def test_target_pooling_is_deterministic_and_resizes_relevance(self):
        residual = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
        relevance = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        first = pool_target_residual(
            residual, relevance, relevance,
            prefix="visual.target_wrist", prompt="white_cabinet_1_bottom_level",
        )
        second = pool_target_residual(
            residual, relevance, relevance,
            prefix="visual.target_wrist", prompt="white_cabinet_1_bottom_level",
        )
        np.testing.assert_array_equal(first.values, second.values)
        self.assertEqual(first.prompt, "white cabinet bottom drawer")

    def test_target_pooling_exposes_predicted_to_actual_displacement(self):
        residual = np.ones((5, 5), dtype=np.float32)
        predicted = np.zeros((5, 5), dtype=np.float32)
        actual = np.zeros((5, 5), dtype=np.float32)
        predicted[2, 1] = 1.0
        actual[2, 3] = 1.0
        result = pool_target_residual(
            residual, predicted, actual,
            prefix="visual.target_primary", prompt="ketchup_1",
        )
        values = dict(zip(result.names, result.values.tolist()))
        self.assertAlmostEqual(values["visual.target_primary.centroid_delta_x"], 0.5)
        self.assertAlmostEqual(values["visual.target_primary.centroid_delta_y"], 0.0)
        self.assertAlmostEqual(values["visual.target_primary.centroid_delta_l2"], 0.5)

    def test_structural_residual_detects_internal_boundary_change(self):
        from PIL import Image

        predicted = np.zeros((64, 64, 3), dtype=np.uint8)
        actual = predicted.copy()
        predicted[24:27, 8:56] = 255
        actual[36:39, 8:56] = 255
        residual = structural_residual_grid(Image.fromarray(predicted), Image.fromarray(actual))
        self.assertEqual(residual.shape, (64, 64))
        self.assertGreater(float(residual.max()), 0.0)
        self.assertGreater(float(residual.sum()), 0.0)

    def test_target_semantics_are_language_derived(self):
        drawer = target_semantic_features("white_cabinet_1_bottom_level")
        bottle = target_semantic_features("ketchup_1")
        self.assertEqual(drawer.values.tolist(), [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(bottle.values.tolist(), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        stacked = target_semantic_features("akita black bowl stack right bowl on left bowl")
        self.assertEqual(stacked.values[3:].tolist(), [1.0, 1.0, 1.0, 1.0])


class CrossTaskContrastiveTests(unittest.TestCase):
    def test_cross_task_positive_pairs_reduce_loss_when_aligned(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is tested in the cloud training environment")
        factors = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
        tasks = np.asarray([0, 1, 0, 1], dtype=np.int64)
        aligned = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
        crossed = np.asarray([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
        aligned_loss = cross_task_factor_contrastive_loss(
            torch.from_numpy(aligned), torch.from_numpy(factors), torch.from_numpy(tasks)
        )
        crossed_loss = cross_task_factor_contrastive_loss(
            torch.from_numpy(crossed), torch.from_numpy(factors), torch.from_numpy(tasks)
        )
        self.assertLess(float(aligned_loss), float(crossed_loss))

    def test_no_cross_task_positive_returns_differentiable_zero(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is tested in the cloud training environment")
        embeddings = torch.randn(3, 4, requires_grad=True)
        factors = torch.eye(3)
        tasks = torch.zeros(3, dtype=torch.long)
        loss = cross_task_factor_contrastive_loss(embeddings, factors, tasks)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(embeddings.grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)

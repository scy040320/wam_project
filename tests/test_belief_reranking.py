from __future__ import annotations

import unittest
import numpy as np

from wam_reranking import (
    ActionRefinementRequest, ActionRefinementResult, AttributionOutput, BeliefFact,
    CoarseCause, ConsistencyFactor, DEFAULT_GRAPH, DependencyGraph, EvidenceQuality,
    HeadThreshold, ScoreWeights, Stage, TriValue, build_attribution_output, evaluate_candidate, initial_belief,
    parse_candidate_effect, select_candidate, update_belief,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

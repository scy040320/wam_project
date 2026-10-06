"""Synthetic bridge tests: code coupling/provenance, not task-level benefits."""
from dataclasses import replace
import unittest

import numpy as np

from wam_reranking.belief import DependencyGraph
from wam_reranking.contracts import (
    AttributionOutput, BeliefState, CoarseCause, ConsistencyFactor,
    CandidateEffect, EvidenceQuality, Stage, TriValue,
)
from wam_reranking.direct_recovery import (
    DirectRecoveryModel, masked_attribution_input, no_dependency_input,
    assert_equal_candidate_information,
)
from wam_reranking.direct_recovery_context import build_direct_recovery_inputs
from wam_reranking.recovery_contract import (
    CurrentFactEvidence, prepare_recovery_context,
)


def attribution(cause):
    factors = {f.value: .95 for f in ConsistencyFactor}
    if cause is CoarseCause.OBJECT_SHIFT:
        factors[ConsistencyFactor.WORLD_STATE_CONSISTENT.value] = .1
    elif cause is CoarseCause.EXECUTION_CONTACT_DEVIATION:
        factors[ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value] = .1
    elif cause is CoarseCause.VISUAL_OCCLUSION:
        factors[ConsistencyFactor.OBSERVATION_RELIABLE.value] = .1
    elif cause is CoarseCause.UNKNOWN:
        factors[ConsistencyFactor.CAUSE_RESOLVED.value] = .1
    return AttributionOutput(factors, {c.value: float(c is cause) for c in CoarseCause},
        cause, .9, 0., EvidenceQuality(True, True, True), "predicted:block:3")


def context(cause=CoarseCause.OBJECT_SHIFT, *, graph=None, observations=()):
    prior = BeliefState(46, "target", "container", task_stage=Stage.LIFT)
    for name, fact in prior.facts.items():
        fact.value = TriValue.TRUE
        fact.confidence = .9
        fact.updated_at_block = 2
        fact.evidence_ids = ("actual:before:" + name,)
    kwargs = {} if graph is None else dict(graph=graph)
    return prepare_recovery_context(prior=prior, attribution=attribution(cause),
        observations=observations, block_index=3, **kwargs)


def candidate(cid=0, predicate="grasped", stage=Stage.LIFT):
    effect = CandidateEffect(cid, stage, {predicate: .8},
        {predicate: (TriValue.TRUE, .9)}, .9,
        {"grasp_support_after": .7, "target_displacement": .2})
    plan = np.zeros((16, 7))
    plan[:, 6] = 1.
    plan[4:, 2] = .2
    return effect, plan


class DirectRecoveryContextTests(unittest.TestCase):
    def build(self, ctx=None, *, predicate="grasped", stage=Stage.LIFT):
        e, a = candidate(predicate=predicate, stage=stage)
        return build_direct_recovery_inputs(context() if ctx is None else ctx, [e], [a], [.4])

    def test_object_shift_grasp_need_comes_from_actual_dag_path(self):
        bridge = self.build()
        self.assertEqual(len(bridge.inputs[0].needs), 1)
        need = bridge.inputs[0].needs[0]
        self.assertEqual((need.cause, need.predicate, need.requirement_time),
                         ("object_shift", "grasped", 4))
        self.assertTrue(need.propagated)
        self.assertEqual(bridge.need_provenance[0]["propagation_path"],
                         ["target_pose_current", "target_reachable", "grasped"])
        self.assertAlmostEqual(need.weight, .9 * .85**2)

    def test_requirement_confidence_is_applied_once_by_existing_scorer(self):
        bridge = self.build()
        scored = DirectRecoveryModel(()).score_pool(bridge.inputs, reference_candidate_id=0)
        contribution = scored[0]["recovery_contributions"][0]
        expected = .9 * .85**2 * .8
        self.assertAlmostEqual(contribution["effective_need_weight"], expected)
        self.assertAlmostEqual(bridge.need_provenance[0]["scorer_effective_need_weight"], expected)

    def test_execution_deviation_uses_execution_root(self):
        bridge = self.build(context(CoarseCause.EXECUTION_CONTACT_DEVIATION))
        need = bridge.inputs[0].needs[0]
        self.assertEqual(need.cause, "execution_contact_deviation")
        self.assertEqual(bridge.need_provenance[0]["propagation_path"],
                         ["execution_consistent", "grasped"])

    def test_occlusion_has_visibility_need_not_physical_grasp_need(self):
        ctx = context(CoarseCause.VISUAL_OCCLUSION)
        self.assertEqual(self.build(ctx).inputs[0].needs, ())
        bridge = self.build(ctx, predicate="target_visible", stage=Stage.APPROACH)
        self.assertEqual(bridge.inputs[0].needs[0].cause, "visual_occlusion")
        self.assertFalse(bridge.inputs[0].needs[0].propagated)

    def test_unknown_does_not_guess_physical_grasp_recovery(self):
        ctx = context(CoarseCause.UNKNOWN)
        self.assertEqual(self.build(ctx).inputs[0].needs, ())

    def test_unknown_visibility_scope_is_direct_epistemic_need(self):
        ctx = context(CoarseCause.UNKNOWN)
        # Frozen unknown update on LIFT marks physical needs only: no invented
        # target-visible requirement is backfilled from a generic unknown.
        bridge = self.build(ctx, predicate="target_visible", stage=Stage.APPROACH)
        self.assertEqual(bridge.inputs[0].needs, ())

    def test_normal_consistency_does_not_create_need(self):
        self.assertEqual(self.build(context(CoarseCause.NORMAL)).inputs[0].needs, ())

    def test_current_observation_refresh_removes_attribution_need(self):
        observed = CurrentFactEvidence("grasped", TriValue.TRUE, .9, "actual:wrist:held", view="wrist")
        self.assertEqual(self.build(context(observations=(observed,))).inputs[0].needs, ())

    def test_false_observation_is_not_relabelled_attribution_created(self):
        observed = CurrentFactEvidence("grasped", TriValue.FALSE, .9, "actual:wrist:separated", view="wrist")
        self.assertEqual(self.build(context(observations=(observed,))).inputs[0].needs, ())

    def test_generic_unknown_without_this_block_invalidation_stays_zero(self):
        ctx = context()
        ctx = replace(ctx, changes=())
        self.assertEqual(self.build(ctx).inputs[0].needs, ())

    def test_initial_unknown_propagated_again_does_not_become_cause_created_need(self):
        prior = BeliefState(46, "target", "container", task_stage=Stage.LIFT)
        ctx = prepare_recovery_context(prior=prior, attribution=attribution(CoarseCause.OBJECT_SHIFT),
                                       block_index=3)
        self.assertTrue(any(c.predicate == "grasped" and c.old_value is TriValue.UNKNOWN
                            for c in ctx.changes))
        self.assertEqual(self.build(ctx).inputs[0].needs, ())

    def test_prior_prediction_and_unverified_initial_assumption_never_create_credit(self):
        for source, eid in (("candidate_prediction", "predicted:grasp"),
                            ("observation", "initial_observation")):
            prior = BeliefState(46, "target", "container", task_stage=Stage.LIFT)
            fact = prior.facts["grasped"]
            fact.value, fact.confidence, fact.source = TriValue.TRUE, .9, source
            fact.evidence_ids = (eid,)
            ctx = prepare_recovery_context(prior=prior, attribution=attribution(CoarseCause.OBJECT_SHIFT),
                                           block_index=3)
            self.assertEqual(self.build(ctx).inputs[0].needs, ())

    def test_preexisting_false_not_claimed_as_this_cause_new_failure(self):
        ctx = context()
        ctx = replace(ctx, changes=tuple(replace(c, old_value=TriValue.FALSE)
            if c.predicate == "grasped" else c for c in ctx.changes))
        self.assertEqual(self.build(ctx).inputs[0].needs, ())

    def test_wrong_reason_root_or_impossible_dag_edge_is_not_scored(self):
        for updates in (dict(reason="teacher_missing_grasp"),
                        dict(propagation_path=("execution_consistent", "grasped")),
                        dict(propagation_path=("target_pose_current", "grasped"))):
            ctx = context()
            ctx = replace(ctx, changes=tuple(replace(c, **updates)
                if c.predicate == "grasped" else c for c in ctx.changes))
            self.assertEqual(self.build(ctx).inputs[0].needs, ())

    def test_no_dag_has_no_fabricated_descendant_need(self):
        self.assertEqual(self.build(context(graph=DependencyGraph(()))).inputs[0].needs, ())

    def test_candidate_forecast_cannot_become_before_need(self):
        ctx = context()
        ctx.belief.facts["grasped"].source = "candidate_prediction"
        self.assertEqual(self.build(ctx).inputs[0].needs, ())

    def test_stale_block_or_different_evidence_id_stays_zero(self):
        for field, value in (("updated_at_block", 1), ("evidence_ids", ("other:block",))):
            ctx = context()
            setattr(ctx.belief.facts["grasped"], field, value)
            self.assertEqual(self.build(ctx).inputs[0].needs, ())

    def test_same_b_and_requirements_survive_masked_and_no_dependency_controls(self):
        e0, a0 = candidate(0)
        e1, a1 = candidate(1)
        bridge = build_direct_recovery_inputs(context(), [e0, e1], [a0, a1], [.4, .5])
        masked = tuple(masked_attribution_input(c) for c in bridge.inputs)
        nodag = tuple(no_dependency_input(c) for c in bridge.inputs)
        self.assertTrue(assert_equal_candidate_information(bridge.inputs, masked, nodag))
        self.assertEqual([c.requirements for c in bridge.inputs], [c.requirements for c in masked])

    def test_factory_without_admitted_heads_is_exact_zero_no_fit(self):
        bridge = self.build()
        scores = DirectRecoveryModel(()).score_pool(bridge.inputs, reference_candidate_id=0)
        self.assertEqual(scores[0]["cause_recovery_residual"], 0.)
        self.assertEqual(scores[0]["total_score"], .4)

    def test_requirement_time_zero_not_shifted_to_future_to_unlock_head(self):
        bridge = self.build(predicate="target_pose_current", stage=Stage.APPROACH)
        self.assertEqual(bridge.inputs[0].needs[0].requirement_time, 0)
        self.assertEqual(bridge.inputs[0].requirements[0].requirement_time, 0)

    def test_no_mutation_of_belief_or_predicted_effects(self):
        ctx = context()
        before = ctx.belief.snapshot()
        self.build(ctx)
        self.assertEqual(ctx.belief.snapshot(), before)

    def test_raw_features_do_not_include_extra_gt_fields(self):
        e, a = candidate()
        e = replace(e, evidence={**e.evidence, "actual_success": 1., "sim_contact": 1.})
        bridge = build_direct_recovery_inputs(context(), [e], [a], [.4])
        self.assertFalse(any("success" in n or "sim_contact" in n
                             for n in bridge.inputs[0].raw.feature_names))

    def test_duplicate_or_unaligned_candidate_inputs_rejected(self):
        e, a = candidate()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_direct_recovery_inputs(context(), [e, e], [a, a], [.4, .4])
        with self.assertRaisesRegex(ValueError, "aligned"):
            build_direct_recovery_inputs(context(), [e], [], [.4])


if __name__ == "__main__":
    unittest.main()

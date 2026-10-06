"""Synthetic interface tests, explicitly not evidence of learned benefit."""
from dataclasses import replace
import unittest
import numpy as np

from wam_reranking.belief import DependencyGraph
from wam_reranking.contracts import (AttributionOutput, BeliefState, CandidateEffect,
    CoarseCause, ConsistencyFactor, EvidenceQuality, Stage, TriValue)
from wam_reranking.joint_recovery_context import build_whole_recovery_tables
from wam_reranking.joint_recovery_contract import score_pool
from wam_reranking.recovery_contract import prepare_recovery_context, CurrentFactEvidence


def context(cause, *, graph=None, empty=False, observations=()):
    prior = BeliefState(46, "target", "anchor", task_stage=Stage.LIFT)
    if not empty:
        for fact in prior.facts.values():
            fact.value, fact.confidence, fact.evidence_ids = TriValue.TRUE, .9, ("actual:before",)
    factors = {f.value: .95 for f in ConsistencyFactor}
    changed = {CoarseCause.OBJECT_SHIFT: ConsistencyFactor.WORLD_STATE_CONSISTENT,
        CoarseCause.EXECUTION_CONTACT_DEVIATION: ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT,
        CoarseCause.VISUAL_OCCLUSION: ConsistencyFactor.OBSERVATION_RELIABLE,
        CoarseCause.UNKNOWN: ConsistencyFactor.CAUSE_RESOLVED}.get(cause)
    if changed:
        factors[changed.value] = .1
    attr = AttributionOutput(factors, {c.value: float(c is cause) for c in CoarseCause},
        cause, .9, 0., EvidenceQuality(True, True, True), "predicted:block:3")
    kw = {} if graph is None else dict(graph=graph)
    return prepare_recovery_context(prior=prior, attribution=attr, block_index=3,
        observations=observations, **kw)


def candidate(cid=0, stage=Stage.LIFT):
    plan = np.zeros((16, 7)); plan[:, 6] = 1.; plan[4:, 2] = .2
    req = {"grasped": .85} if stage is Stage.LIFT else {}
    effect = CandidateEffect(cid, stage, req, {}, .9, {})
    return effect, plan


class WholeRecoveryContextTests(unittest.TestCase):
    def build(self, cause=CoarseCause.OBJECT_SHIFT, **kwargs):
        e, a = candidate()
        return build_whole_recovery_tables(context(cause), [e], [a], **kwargs)[0]

    def test_source_dag_boundary_goal_does_not_move_required_use(self):
        row = self.build(frozen_boundary_goals=("grasped",))
        self.assertEqual(row["requirements"][0]["deadline"], 4)
        self.assertEqual(row["goals"][0]["deadline"], 16)
        self.assertEqual(row["needs"][0]["path"], ["target_pose_current", "target_reachable", "grasped"])
        self.assertEqual({n["deadline"] for n in row["needs"]}, {4, 16})

    def test_normal_has_no_restoration_need_or_goal(self):
        row = self.build(CoarseCause.NORMAL, frozen_boundary_goals=("grasped",))
        self.assertEqual(row["needs"], [])
        self.assertEqual(row["goals"], [])

    def test_execution_uses_its_real_root(self):
        row = self.build(CoarseCause.EXECUTION_CONTACT_DEVIATION, frozen_boundary_goals=("grasped",))
        self.assertEqual(row["needs"][0]["path"], ["execution_consistent", "grasped"])

    def test_occlusion_and_unknown_never_guess_physical_goal(self):
        for cause in (CoarseCause.VISUAL_OCCLUSION, CoarseCause.UNKNOWN):
            with self.assertRaisesRegex(ValueError, "guessed physical"):
                self.build(cause, frozen_boundary_goals=("grasped",))
            self.assertEqual(self.build(cause)["needs"], [])

    def test_epistemic_goal_separate_from_physical_visibility(self):
        e, a = candidate(stage=Stage.OBSERVE)
        for cause in (CoarseCause.UNKNOWN, CoarseCause.VISUAL_OCCLUSION):
            row = build_whole_recovery_tables(context(cause), [e], [a],
                information_acquisition=True, before_observer_available=False)[0]
            self.assertEqual(row["requirements"], [])
            self.assertEqual(row["goals"][0]["predicate"], "observer_evidence_available")
            self.assertEqual(row["needs"][0]["origin"], "before_epistemic_insufficiency")

    def test_secondary_goal_keeps_primary_stage_and_original_requirements(self):
        e, a = candidate()
        row = build_whole_recovery_tables(context(CoarseCause.UNKNOWN), [e], [a],
            information_acquisition=True, before_observer_available=False)[0]
        self.assertEqual(row["stage"], "lift")
        self.assertEqual(row["requirements"][0]["deadline"], 4)
        self.assertEqual(row["purpose"], "task_with_observation_recovery")
        self.assertEqual(row["goals"][0]["predicate"], "observer_evidence_available")

    def test_no_secondary_goal_unless_requested_before(self):
        e, a = candidate()
        row = build_whole_recovery_tables(context(CoarseCause.UNKNOWN), [e], [a],
            information_acquisition=False, before_observer_available=False)[0]
        self.assertEqual(row["goals"], [])

    def test_before_available_unknown_or_none_does_not_create_goal(self):
        e, a = candidate(stage=Stage.OBSERVE)
        for value in (None, True):
            row = build_whole_recovery_tables(context(CoarseCause.UNKNOWN), [e], [a],
                information_acquisition=True, before_observer_available=value)[0]
            self.assertEqual(row["goals"], [])

    def test_initial_unknown_not_false_or_caused_loss(self):
        e, a = candidate()
        row = build_whole_recovery_tables(context(CoarseCause.OBJECT_SHIFT, empty=True), [e], [a],
            frozen_boundary_goals=("grasped",))[0]
        self.assertEqual(row["needs"], [])
        self.assertEqual(row["goals"], [])

    def test_real_current_refresh_erases_caused_goal(self):
        e, a = candidate()
        obs = CurrentFactEvidence("grasped", TriValue.TRUE, .9, "actual:wrist", view="wrist")
        row = build_whole_recovery_tables(context(CoarseCause.OBJECT_SHIFT, observations=(obs,)), [e], [a],
            frozen_boundary_goals=("grasped",))[0]
        self.assertEqual(row["goals"], [])
        self.assertEqual(row["needs"], [])

    def test_absent_dependency_no_descendant_credit(self):
        e, a = candidate()
        row = build_whole_recovery_tables(context(CoarseCause.OBJECT_SHIFT, graph=DependencyGraph(())), [e], [a],
            frozen_boundary_goals=("grasped",))[0]
        self.assertEqual(row["needs"], [])

    def test_pipeline_changes_soft_score_without_updating_belief(self):
        ctx = context(CoarseCause.OBJECT_SHIFT)
        before = ctx.belief.snapshot()
        pairs = [candidate(0), candidate(1)]
        tables = build_whole_recovery_tables(ctx, [p[0] for p in pairs], [p[1] for p in pairs],
            frozen_boundary_goals=("grasped",))
        rows = [dict(t, identity={"candidate_id": t["candidate_id"]}, backbone_score=.4,
            entry_route="safe_fallback") for t in tables]
        probs = {0: {("grasped", 4, "required_use"): .2, ("grasped", 16, "next_boundary_goal"): .2},
                 1: {("grasped", 4, "required_use"): .8, ("grasped", 16, "next_boundary_goal"): .8}}
        scored = score_pool(rows, probs, mode="full", reference_candidate_id=0)
        masked = score_pool(rows, probs, mode="masked", reference_candidate_id=0)
        nodag = score_pool(rows, probs, mode="no_dag", reference_candidate_id=0)
        self.assertGreater(scored[1]["recovery_residual"], 0.)
        self.assertEqual(masked[1]["recovery_residual"], 0.)
        self.assertEqual(nodag[1]["recovery_residual"], 0.)
        self.assertEqual(scored[1]["entry_route"], "safe_fallback")
        self.assertEqual(ctx.belief.snapshot(), before)

    def test_outcome_not_in_api(self):
        import inspect
        names = inspect.signature(build_whole_recovery_tables).parameters
        self.assertFalse(any(s in name for name in names for s in ("success", "teacher", "actual_after", "label")))


if __name__ == "__main__":
    unittest.main()

from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np

from wam_reranking import initial_belief
from wam_reranking.belief import update_belief
from wam_reranking.contracts import (AttributionOutput, BeliefFact, CandidateDecision,
    CandidateEffect, CoarseCause, ConsistencyFactor, EvidenceQuality, Stage, TriValue)
from wam_reranking.candidate_utility import CandidateUtilityModel
from wam_reranking.evidence_residual import EvidenceResidualModel
from wam_reranking.evidence_arbitration import select_evidence_arbitration
from wam_reranking.persistent_conditioned_utility import (FEATURE_NAMES, ConditionalUtility,
    conditioned_features, comparable_ids, fit_once, select_conditioned)


def attr(cause=CoarseCause.OBJECT_SHIFT, visual=True, execution=True):
    p = {f.value: 1. for f in ConsistencyFactor}
    if cause is CoarseCause.OBJECT_SHIFT: p['world_state_consistent'] = 0.
    if cause is CoarseCause.EXECUTION_CONTACT_DEVIATION: p['execution_contact_consistent'] = 0.
    if not visual: p['observation_reliable'] = 0.
    return AttributionOutput(p, {c.value: float(c is cause) for c in CoarseCause}, cause, .9, 0.,
        EvidenceQuality(visual, visual, execution), 'source',
        {f.value: TriValue.TRUE if p[f.value] else TriValue.FALSE for f in ConsistencyFactor})


def effect():
    return CandidateEffect(0, Stage.LIFT, {'grasped': .6, 'execution_consistent': .6},
        {'lifted': (TriValue.TRUE, .9)}, .9,
        {'cross_view_agreement': .9, 'relation_confidence': .9, 'contact_confidence': .9,
         'grasp_support_after': .9, 'target_displacement': .4}, ())


class PersistentUtilityTests(unittest.TestCase):
    def feature(self, prior=None, attribution=None, **kwargs):
        plan = np.zeros((16, 7)); plan[:, 2] = .1; plan[:, 6] = 1.
        return conditioned_features(prior=initial_belief(0) if prior is None else prior,
            attribution=attr() if attribution is None else attribution, effect=effect(), actions=plan, **kwargs)

    def test_history_survives_normal_without_becoming_unknown_false(self):
        b = initial_belief(0); update_belief(b, attr(), 3)
        before = b.snapshot()
        x, trace = self.feature(b, attr(CoarseCause.NORMAL), block_index=4)
        self.assertEqual(before, b.snapshot())
        self.assertEqual(trace['deficits']['target_pose_current']['state'], 'false')
        self.assertEqual(trace['deficits']['grasped']['state'], 'unknown')
        self.assertTrue(trace['deficits']['grasped']['historical'])
        self.assertTrue(np.any(x))

    def test_current_true_does_not_resurrect_history(self):
        b = initial_belief(0); update_belief(b, attr(), 3)
        b.facts['target_pose_current'] = BeliefFact(TriValue.TRUE, .95, 'observation', 4, ('new_observation',))
        _, trace = self.feature(b, attr(CoarseCause.NORMAL), block_index=4)
        self.assertNotIn('target_pose_current', trace['deficits'])

    def test_candidate_prediction_cannot_certify_current_truth(self):
        b = initial_belief(0); update_belief(b, attr(), 3)
        b.facts['grasped'] = BeliefFact(TriValue.TRUE, .99, 'candidate_prediction', 3, ('forecast',))
        before = deepcopy(b.snapshot()); self.feature(b, attr(CoarseCause.NORMAL))
        self.assertEqual(before, b.snapshot())

    def test_no_dag_removes_only_propagated_deficits(self):
        b = initial_belief(0); update_belief(b, attr(), 3)
        _, t = self.feature(b, attr(CoarseCause.NORMAL), no_dag=True)
        self.assertIn('target_pose_current', t['deficits']); self.assertNotIn('grasped', t['deficits'])

    def test_reliable_execution_survives_bad_vision_without_contact_certificate(self):
        x, trace = self.feature(attribution=attr(CoarseCause.EXECUTION_CONTACT_DEVIATION, visual=False))
        self.assertTrue(np.any(x)); self.assertFalse(trace['visual_usable'])
        self.assertTrue(trace['execution_usable'])
        self.assertFalse(any(v for n, v in zip(FEATURE_NAMES, x) if '.grasped.' in n))
        self.assertTrue(any(v for n, v in zip(FEATURE_NAMES, x) if '.execution_consistent.' in n))

    def test_unreliable_execution_is_masked(self):
        x, _ = self.feature(attribution=attr(CoarseCause.EXECUTION_CONTACT_DEVIATION, visual=False, execution=False))
        self.assertFalse(np.any(x))

    def test_coarse_disagreement_not_fabricated_physical_deficit(self):
        a = attr(CoarseCause.EXECUTION_CONTACT_DEVIATION)
        a = replace(a, factor_probs={f.value: 1. for f in ConsistencyFactor},
                    factor_states={f.value: TriValue.TRUE for f in ConsistencyFactor})
        x, trace = self.feature(attribution=a)
        self.assertFalse(np.any(x)); self.assertEqual(trace['deficits'], {})

    def fixture(self):
        backbone = CandidateUtilityModel(np.zeros(27), np.ones(27), np.zeros(27), 0., switch_margin=.01)
        residual = EvidenceResidualModel(np.ones(10), np.zeros(10), np.ones(10), 0., .1)
        ds = [CandidateDecision(i, True, .6 - i * .01, None, (), {}) for i in range(3)]
        xs = {i: np.zeros(27) for i in range(3)}
        for i in range(3): xs[i][0] = ds[i].official_value; xs[i][16] = 1.; xs[i][5] = xs[i][8] = .9
        _, t = self.feature()
        return dict(decisions=ds, backbone_features=xs, cause_features={i: np.zeros(10) for i in range(3)},
            backbone=backbone, frozen_residual=residual, attribution=attr(), traces=[deepcopy(t) for _ in range(3)])

    def test_zero_exact_v8_scores_choice(self):
        k = self.fixture(); original = {n: v for n, v in k.items() if n not in {'frozen_residual', 'traces'}}
        ref, scores = select_evidence_arbitration(**original, residual=k['frozen_residual'])
        chosen, new, _ = select_conditioned(**k, deltas=[0., 0., 0.])
        self.assertEqual(ref, chosen); self.assertEqual(scores, new)

    def test_explainable_score_difference_changes_choice(self):
        k = self.fixture(); chosen, scores, trace = select_conditioned(**k, deltas=[-.04, .04, 0.])
        self.assertEqual(chosen.candidate_id, 1); self.assertTrue(trace['conditional_switch'])
        self.assertGreater(scores[1] - scores[0], .01)

    def test_hard_rejection_and_margin_unchanged(self):
        k = self.fixture(); k['decisions'][1] = replace(k['decisions'][1], accepted=False, rejection_reasons=('physical_false',))
        chosen, _, _ = select_conditioned(**k, deltas=[-.04, .04, 0.])
        self.assertNotEqual(chosen.candidate_id, 1)
        chosen, _, _ = select_conditioned(**self.fixture(), deltas=[0., .005, 0.])
        self.assertEqual(chosen.candidate_id, 0)

    def test_one_unseen_feature_does_not_close_supported_execution_channel(self):
        n = len(FEATURE_NAMES); support = np.zeros(n); support[0] = 1.
        weights = np.zeros(n); weights[0] = 1.
        model = ConditionalUtility(np.ones(n), weights, support, 1.)
        x = np.zeros((2, n)); x[1, 0] = 1.; x[0, 1] = 100.
        self.assertGreater(model.deltas(x)[1], model.deltas(x)[0])
        x[:, 0] = 0.; self.assertFalse(np.any(model.deltas(x)))

    def test_train_only_and_single_fit_has_score_scale(self):
        k = self.fixture(); n = len(FEATURE_NAMES); x = np.zeros((3, n)); x[1, 0] = 1.
        p = dict(split='train', x=x, baseline_id=0, comparable={0, 1}, kw=k,
                 base_scores=np.array([.1, .095, .08]),
                 y=[dict(success=False, steps=20, calls=4), dict(success=True, steps=10, calls=2), dict(success=False, steps=20, calls=4)])
        model, report = fit_once([p], steps=300)
        self.assertEqual(report['fit_count'], 1)
        self.assertGreater(model.__class__(model.scale, model.weights, model.support, 1.).deltas(x)[1] -
                           model.__class__(model.scale, model.weights, model.support, 1.).deltas(x)[0], .015)
        with self.assertRaises(ValueError): fit_once([{**p, 'split': 'val'}])


if __name__ == '__main__': unittest.main()

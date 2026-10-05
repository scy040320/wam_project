import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from wam_reranking import BeliefFact, CoarseCause, ConsistencyFactor, EvidenceQuality, TriValue, initial_belief
from wam_reranking.source_scoped_policy import prepare_source_scoped_decisions, scoped_attribution
from wam_reranking.shared_query_cache import SharedQueryCache
from test_evidence_residual import attr, effect, model


class SourceScopedTests(unittest.TestCase):
    def test_unknown_world_cannot_become_physical_false(self):
        a = attr(CoarseCause.UNKNOWN)
        a = replace(a, factor_probs={**a.factor_probs, 'world_state_consistent': .49})
        self.assertIs(scoped_attribution(a).factor_state(ConsistencyFactor.WORLD_STATE_CONSISTENT), TriValue.UNKNOWN)

    def test_obscured_world_negative_is_unknown(self):
        a = attr(CoarseCause.UNKNOWN, True)
        a = replace(a, factor_probs={**a.factor_probs, 'world_state_consistent': .01})
        self.assertIs(scoped_attribution(a).factor_state(ConsistencyFactor.WORLD_STATE_CONSISTENT), TriValue.UNKNOWN)

    def test_resolved_visible_world_negative_is_preserved(self):
        self.assertIs(scoped_attribution(attr(CoarseCause.OBJECT_SHIFT)).factor_state(ConsistencyFactor.WORLD_STATE_CONSISTENT), TriValue.FALSE)

    def test_command_evidence_not_masked_by_occlusion(self):
        a = attr(CoarseCause.UNKNOWN, True)
        a = replace(a, factor_probs={**a.factor_probs, 'execution_contact_consistent': .01})
        self.assertIs(scoped_attribution(a).factor_state(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT), TriValue.FALSE)

    def test_reliable_unknown_state_mismatch_preserves_recovery_backbone(self):
        a = attr(CoarseCause.UNKNOWN)
        a = replace(a, factor_probs={**a.factor_probs, 'world_state_consistent': .49})
        ds = prepare_source_scoped_decisions(belief=initial_belief(0), attribution=a,
            block_index=3, effects=[effect()], values=[.9], relation='inside', backbone=model())
        self.assertFalse(ds[0].components.get('preserve_official_value'))

    def test_unresolved_but_consistent_physical_factors_keep_value_route(self):
        ds = prepare_source_scoped_decisions(belief=initial_belief(0), attribution=attr(CoarseCause.UNKNOWN),
            block_index=3, effects=[effect()], values=[.9], relation='inside', backbone=model())
        self.assertTrue(ds[0].components.get('preserve_official_value'))
        self.assertTrue(ds[0].components.get('unresolved_without_supported_physical_recovery'))

    def test_unreliable_view_still_preserves_official_route(self):
        a = replace(attr(CoarseCause.UNKNOWN), factor_probs={**attr(CoarseCause.UNKNOWN).factor_probs, 'observation_reliable': .01})
        ds = prepare_source_scoped_decisions(belief=initial_belief(0), attribution=a,
            block_index=3, effects=[effect()], values=[.9], relation='inside', backbone=model())
        self.assertTrue(ds[0].components.get('preserve_official_value'))

    def test_witnessed_physical_false_remains_hard_rejection(self):
        b = initial_belief(0)
        b.facts['lifted'] = BeliefFact(TriValue.FALSE, .99, 'observation')
        ds = prepare_source_scoped_decisions(belief=b, attribution=attr(CoarseCause.UNKNOWN),
            block_index=3, effects=[effect()], values=[.9], relation='inside', backbone=model())
        self.assertFalse(ds[0].accepted)
        self.assertTrue(any('lifted=false' in s for s in ds[0].rejection_reasons))

    def test_contact_violation_cannot_be_overridden(self):
        e = replace(effect(), hard_violations={'predicted_target_gripper_contact_missing': .9})
        ds = prepare_source_scoped_decisions(belief=initial_belief(0), attribution=attr(CoarseCause.UNKNOWN),
            block_index=3, effects=[e], values=[.9], relation='inside', backbone=model())
        self.assertFalse(ds[0].accepted)

    def test_unknown_class_confidence_does_not_amplify_world_factor(self):
        a = attr(CoarseCause.UNKNOWN)
        a = replace(a, confidence=.9999, factor_probs={**a.factor_probs, 'world_state_consistent': .49},
                    factor_confidences={f.value: (.51 if f is ConsistencyFactor.WORLD_STATE_CONSISTENT else .99) for f in ConsistencyFactor})
        b = initial_belief(0)
        prepare_source_scoped_decisions(belief=b, attribution=a, block_index=3,
                                       effects=[effect()], values=[.9], relation='inside', backbone=model())
        history = [c for c in b.history if c.reason == 'world_state_consistent']
        self.assertTrue(history)
        self.assertTrue(all(c.new_confidence <= .51 for c in history))


class QueryCacheTests(unittest.TestCase):
    def result(self):
        return {'actions': np.zeros((16, 7), np.float32), 'future_image_predictions': {
            'future_image': np.zeros((8, 8, 3), np.uint8), 'future_wrist_image': np.ones((8, 8, 3), np.uint8)},
            'value_prediction': .6}, 2.

    def test_exact_request_is_generated_once(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = SharedQueryCache(folder, 'frozen-model')
            first = cache.get_or_generate('obs', 12, 'task-language', self.result)
            def forbidden(): raise AssertionError('same query must not be regenerated')
            second = cache.get_or_generate('obs', 12, 'task-language', forbidden)
            self.assertFalse(first[3]); self.assertTrue(second[3])
            np.testing.assert_array_equal(first[0]['actions'], second[0]['actions'])
            self.assertEqual(first[1:3], second[1:3])

    def test_seed_observation_and_language_are_separate_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = SharedQueryCache(folder, 'model')
            keys = {cache.request(*request)[0] for request in [('obs', 1, 'a'), ('obs', 2, 'a'), ('other', 1, 'a'), ('obs', 1, 'b')]}
            self.assertEqual(len(keys), 4)

    def test_payload_tampering_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = SharedQueryCache(folder, 'model')
            _, _, key, _ = cache.get_or_generate('obs', 1, 'a', self.result)
            np.save(Path(folder) / key / 'actions.npy', np.ones((16, 7)))
            with self.assertRaises(RuntimeError): cache.get_or_generate('obs', 1, 'a', self.result)


if __name__ == '__main__': unittest.main()

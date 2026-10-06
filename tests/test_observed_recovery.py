import inspect
import unittest

import numpy as np

from wam_reranking.contracts import PREDICATES
from wam_reranking.observed_recovery import labels_from_actual, map_endpoint


def visible_map():
    array=np.zeros((8,8),dtype=np.float32);array[2:4,2:4]=1.
    return array


def evidence():
    maps=np.stack([visible_map() for _ in range(4)])
    return dict(identity=dict(dataset='aux_v8',suite='libero90',task=46,state=10,
                             candidate_id=0,observed_block_id='block_2'),
                subject_maps=maps.copy(),anchor_maps=maps.copy(),
                requested=np.zeros((16,7),dtype=np.float32),
                applied=np.zeros((16,7),dtype=np.float32))


class ObservedRecoveryTests(unittest.TestCase):
    def test_hidden_before_visible_after_only_after_supervised(self):
        args=evidence();args['subject_maps'][:2]=0.
        result=labels_from_actual(**args)
        label=result['predicates']['target_visible']
        self.assertEqual((label['before'],label['after']),('unknown','true'))
        self.assertFalse(label['before_mask']);self.assertTrue(label['after_mask'])
        self.assertFalse(label['transition_mask'])
        available=result['observability']['target_evidence_available']
        self.assertEqual((available['before'],available['after']),('false','true'))
        self.assertTrue(available['transition_mask'])

    def test_commands_consistent_do_not_certify_physical_grasp_or_contact(self):
        labels=labels_from_actual(**evidence())
        self.assertEqual(labels['predicates']['command_consistent']['after'],'true')
        for name in set(PREDICATES)-{'target_visible','receptacle_visible'}:
            item=labels['predicates'][name]
            self.assertEqual((item['before'],item['after']),('unknown','unknown'))
            self.assertFalse(item['before_mask'] or item['after_mask'] or item['transition_mask'])

    def test_applied_requested_difference_stays_command_channel(self):
        args=evidence();args['applied'][3,0]=.2
        labels=labels_from_actual(**args)
        item=labels['predicates']['command_consistent']
        self.assertEqual(item['after'],'false');self.assertTrue(item['after_mask'])
        self.assertEqual(labels['predicates']['grasped']['after'],'unknown')
        self.assertFalse(labels['predicates']['execution_consistent']['after_mask'])

    def test_unusable_localizer_is_not_object_absence(self):
        args=evidence();args['subject_maps'][:]=0.
        labels=labels_from_actual(**args)
        item=labels['predicates']['target_visible']
        self.assertEqual(item['after'],'unknown');self.assertFalse(item['after_mask'])
        self.assertEqual(labels['observability']['target_evidence_available']['after'],'false')
        proxy=labels['relation_support_2d']
        self.assertIsNone(proxy['after']);self.assertFalse(proxy['after_mask'])

    def test_missing_view_endpoint_rejected(self):
        args=evidence();args['subject_maps']=args['subject_maps'][:3]
        with self.assertRaises(ValueError):labels_from_actual(**args)

    def test_nonfinite_actual_map_rejected(self):
        args=evidence();args['anchor_maps'][3,0,0]=np.nan
        with self.assertRaises(ValueError):labels_from_actual(**args)

    def test_nonfinite_actual_commands_rejected(self):
        args=evidence();args['applied'][0,0]=np.inf
        with self.assertRaises(ValueError):labels_from_actual(**args)

    def test_empty_map_is_not_valid_actual_evidence(self):
        with self.assertRaises(ValueError):map_endpoint(np.zeros((0,0),dtype=np.float32))

    def test_final_success_or_attribution_are_not_accepted_inputs(self):
        parameters=set(inspect.signature(labels_from_actual).parameters)
        self.assertEqual(parameters,{'identity','subject_maps','anchor_maps','requested','applied'})
        for forbidden in ('success','attribution','intervention','predicted_frames','sim_state'):
            args=evidence();args[forbidden]=True
            with self.subTest(field=forbidden),self.assertRaises(TypeError):labels_from_actual(**args)

    def test_observer_quality_uses_frozen_threshold_not_human_certainty(self):
        weak=map_endpoint(np.zeros((8,8),dtype=np.float32))
        strong=map_endpoint(visible_map())
        self.assertFalse(weak['usable']);self.assertTrue(strong['usable'])
        self.assertGreaterEqual(strong['quality'],.5)
        self.assertLessEqual(strong['quality'],1.)
        proxy=labels_from_actual(**evidence())['relation_support_2d']
        self.assertEqual(proxy['units'],'image_space_proxy')
        self.assertNotIn('physical_contact',proxy)


if __name__=='__main__':unittest.main()

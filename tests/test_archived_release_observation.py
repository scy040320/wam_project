"""Source-time/cache-lineage tests. No simulator, query, pickle or training."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

scripts=Path(__file__).resolve().parents[1]/'scripts'
spec=importlib.util.spec_from_file_location('replay_archived_release_tail',scripts/'replay_archived_release_tail.py')
tail=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tail;spec.loader.exec_module(tail)
spec=importlib.util.spec_from_file_location('_observable_archived_tail_test',scripts/'replay_archived_release_observation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixtures():
    row=dict(task=46,state=10,tail=dict(block=7,candidate_id=1,directory='source/block_7'))
    prefix=[];caches=[]
    for block in range(3,7):
        frames=[dict(step=s,physics={}) for s in range(17)]
        b=dict(block=dict(block=block,candidate_id=3,directory=f'source/block_{block}'),frames=frames,
            proprio=np.zeros((17,9)),qpos=np.zeros((17,2)),qvel=np.zeros((17,2)),checks=[dict(step=s) for s in range(17)])
        prefix.append(b);caches.extend([dict(block=block,step=s) for s in range(17)])
    short=dict(frames=[dict(step=s,physics={}) for s in range(4)],proprio=np.zeros((4,9)),qpos=np.zeros((4,2)),qvel=np.zeros((4,2)),
        checks=[dict(step=s) for s in range(1,4)])
    caches.extend([dict(block=7,step=s) for s in range(1,4)])
    return row,prefix,short,caches


class ObservationTailTests(unittest.TestCase):
    def test_cache_mapping_removes_only_duplicate_boundary_frames(self):
        h=m.aligned_history(*fixtures())
        self.assertEqual(len(h['frames']),68);self.assertEqual(len(h['caches']),68)
        self.assertEqual(h['caches'][16],dict(block=3,step=16))
        self.assertEqual(h['caches'][17],dict(block=4,step=1))
        self.assertEqual(h['caches'][65],dict(block=7,step=1))

    def test_tail_candidate_identity_not_prior_candidate(self):
        h=m.aligned_history(*fixtures())
        self.assertEqual(h['lineage'][64]['candidate_id'],3)
        self.assertEqual(h['lineage'][65]['candidate_id'],1)

    def test_source_step_and_chain_index_are_separate(self):
        h=m.aligned_history(*fixtures());i=m.source_identity(h['lineage'][67])
        self.assertEqual(i['source_step'],3);self.assertEqual(h['lineage'][67]['chain_step'],67)
        self.assertNotIn('local_step',i)

    def test_source_cache_length_mismatch_hard_stop(self):
        row,prefix,short,caches=fixtures()
        with self.assertRaises(ValueError):m.aligned_history(row,prefix,short,caches[:-1])

    def test_missing_original_short_proprio_not_zero(self):
        h=m.aligned_history(*fixtures());end=h['checks'][-1]
        self.assertIsNone(end['endpoint_proprio_max_abs']);self.assertFalse(end['original_endpoint_proprio_available'])

    def test_prefix_original_endpoint_checks_are_retained(self):
        row,prefix,short,caches=fixtures();prefix[-1]['checks'][-1]['endpoint_proprio_max_abs']=.0001
        h=m.aligned_history(row,prefix,short,caches)
        self.assertEqual(h['checks'][64]['endpoint_proprio_max_abs'],.0001)

    def test_causal_motion_starts_exclude_cache_frame_zero_and_future(self):
        for step in range(68):
            source=dict(source_step=step if step<65 else step-64)
            self.assertTrue(all(0<s<step for s in m.causal_starts(step,source)))
        self.assertEqual(m.causal_starts(0,dict(source_step=0)),[])

    def test_release_window_can_reference_past_distinct_candidate_without_transplant(self):
        starts=m.causal_starts(67,dict(source_step=3))
        self.assertIn(65,starts);self.assertIn(66,starts);self.assertIn(51,starts)

    def test_negative_or_boolean_chain_index_rejected(self):
        for step in (-1,True):
            with self.assertRaises(ValueError):m.causal_starts(step,dict(source_step=0))

    def test_original_rgb_lineage_hash_and_true_index_required(self):
        row=dict(task=46);line=dict(observed_block_id='source/block_7',local_step=3,cached_reference_only=False)
        result=m.original_images(row,line,{'source/block_7/trajectory.npz':'sha'})
        self.assertEqual(result['primary']['array_index'],2)
        self.assertEqual(result['primary']['sha256'],'sha')
        with self.assertRaises(ValueError):m.original_images(row,line,{})

    def test_initial_cached_source_not_replaced_with_fresh_rgb(self):
        row={};line=dict(observed_block_id='source/block_3',local_step=0,cached_reference_only=True)
        hashes={f'source/block_3/evidence/block/O_t_{v}.npy':v for v in ('primary','wrist')}
        result=m.original_images(row,line,hashes)
        self.assertIsNone(result['primary']['array_index']);self.assertTrue(result['primary']['cached_reference_only'])

    def test_original_three_step_tail_not_padded_by_cache_mapping(self):
        h=m.aligned_history(*fixtures())
        self.assertEqual(sum(x['observed_block_id']=='source/block_7' for x in h['lineage']),3)
        self.assertEqual(h['proprio'].shape,(68,9))


if __name__=='__main__':unittest.main()

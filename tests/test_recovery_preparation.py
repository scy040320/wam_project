import unittest
import numpy as np
from scripts.prepare_recovery_training import before_observations, shuffled_index
from wam_reranking.contracts import TriValue

class PreparationTests(unittest.TestCase):
    def test_current_observation_cannot_depend_on_after_maps(self):
        before=np.zeros((2,8,8));before[:,2:4,2:4]=1
        a=np.concatenate([before,np.zeros_like(before)])
        b=np.concatenate([before,np.ones_like(before)])
        self.assertEqual(before_observations(a,a),before_observations(b,b))

    def test_empty_or_weak_current_maps_do_not_certify_false(self):
        obs=before_observations(np.zeros((2,8,8)),np.zeros((2,8,8)))
        self.assertEqual(obs,[])

    def test_all_unmasked_current_observations_are_positive_visibility(self):
        maps=np.zeros((2,8,8));maps[:,2:4,2:4]=1
        obs=before_observations(maps,maps)
        self.assertTrue(obs)
        self.assertTrue(all(o.value is TriValue.TRUE for o in obs))
        self.assertEqual({o.predicate for o in obs},{'target_visible','receptacle_visible'})

    def test_shuffle_cannot_move_across_task_or_split(self):
        rows=[dict(id=i,task=t,split=s) for i,(t,s) in enumerate([(0,'train')]*4+[(9,'val')]*4)]
        permutation=shuffled_index(rows,lambda r:r['id'],lambda r:(r['task'],r['split']))
        lookup={r['id']:r for r in rows}
        self.assertEqual(set(permutation),set(lookup))
        self.assertEqual(set(permutation.values()),set(lookup))
        for src,dst in permutation.items():
            self.assertEqual((lookup[src]['task'],lookup[src]['split']),(lookup[dst]['task'],lookup[dst]['split']))

    def test_shuffle_is_fixed_without_reading_outcomes(self):
        rows=[dict(id=i,task=0,split='train',success=bool(i%2)) for i in range(8)]
        one=shuffled_index(rows,lambda r:r['id'],lambda r:(r['task'],r['split']))
        for r in rows:r['success']=not r['success']
        self.assertEqual(one,shuffled_index(rows,lambda r:r['id'],lambda r:(r['task'],r['split'])))

if __name__=='__main__':unittest.main()

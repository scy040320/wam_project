import unittest
from types import SimpleNamespace
import numpy as np
try:
    from replay_temporal_recovery_evidence import decode_segmentation_rgb,image_check,body_descendants
except ModuleNotFoundError:
    from scripts.replay_temporal_recovery_evidence import decode_segmentation_rgb,image_check,body_descendants

class TemporalReplayTest(unittest.TestCase):
    def scene(self,n):
        return SimpleNamespace(ngeom=n,geoms=[SimpleNamespace(segid=i,objtype=5,objid=i+100) for i in range(n)])
    def test_background(self):
        self.assertTrue((decode_segmentation_rgb(np.zeros((2,2,3),np.uint8),self.scene(4))==-1).all())
    def test_red_id(self):
        self.assertEqual(decode_segmentation_rgb(np.array([[[1,0,0]]],np.uint8),self.scene(4)).tolist(),[[[5,100]]])
    def test_green_no_uint8_overflow(self):
        self.assertEqual(decode_segmentation_rgb(np.array([[[4,1,0]]],np.uint8),self.scene(300)).tolist(),[[[5,359]]])
    def test_blue_outside_scene(self):
        self.assertEqual(decode_segmentation_rgb(np.array([[[0,0,1]]],np.uint8),self.scene(300)).tolist(),[[[-1,-1]]])
    def test_invalid_dtype(self):
        with self.assertRaises(ValueError):decode_segmentation_rgb(np.zeros((2,2,3)),self.scene(4))
    def test_invalid_shape(self):
        with self.assertRaises(ValueError):decode_segmentation_rgb(np.zeros((2,2),np.uint8),self.scene(4))
    def test_unused_scene_geom(self):
        s=self.scene(4);s.geoms[1].segid=-1
        self.assertEqual(decode_segmentation_rgb(np.array([[[2,0,0]]],np.uint8),s).tolist(),[[[-1,-1]]])
    def test_exact_render(self):
        a=np.zeros((4,4,3),np.uint8);self.assertTrue(image_check(a,a)['passed'])
    def test_gate_not_loosened(self):
        a=np.zeros((4,4,3),np.uint8);b=np.full_like(a,3);self.assertFalse(image_check(a,b)['passed'])
    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):image_check(np.zeros((2,2,3)),np.zeros((3,3,3)))
    def test_descendant_binding(self):
        m=SimpleNamespace(nbody=5,body_parentid=[0,0,1,1,0]);self.assertEqual(body_descendants(m,1),{1,2,3})
    def test_joint_body_not_all_fixture(self):
        m=SimpleNamespace(nbody=5,body_parentid=[0,0,1,1,0]);self.assertEqual(body_descendants(m,2),{2})

if __name__=='__main__':unittest.main()

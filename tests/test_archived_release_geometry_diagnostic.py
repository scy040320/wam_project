"""Finite CPU checks of diagnostic math, no simulator/policy/labels."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from wam_reranking.observation_geometry import project_world_points

spec=importlib.util.spec_from_file_location("_release_geom_diag_test",Path(__file__).resolve().parents[1]/"scripts/diagnose_archived_release_geometry.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def caches():
    camera=dict(camera_position_m=np.zeros(3),camera_rotation_world_from_camera=np.eye(3),fovy_degrees=90.,image_shape=(8,8),image_convention=1)
    a=dict(camera=camera,depth=np.full((8,8),.5),geom=np.ones((8,8),int),geom_body_ids=np.array([0,1]),
        body_pos=np.zeros((2,3)),body_rot=np.tile(np.eye(3),(2,1,1)),roles={"target":np.ones((8,8),bool)})
    import copy
    b=copy.deepcopy(a)
    return {"primary":a},{"primary":b}


class GeometryDiagnosticTests(unittest.TestCase):
    def trace(self,a,b,p):return m.trace_material_points(a,b,"primary","target",p,project_world_points)
    def test_stationary_center_roundtrip_is_exact(self):
        ep,t=self.trace(*caches(),[[4,4]])
        self.assertTrue(ep["valid"][0]);self.assertAlmostEqual(t["start_roundtrip_error_px"][0],0)
    def test_body_translation_changes_material_not_camera_only(self):
        a,b=caches();b["primary"]["body_pos"][1,0]=.1
        ep,t=self.trace(a,b,[[4,4]])
        self.assertAlmostEqual(ep["end_pixel_xy"][0,0],4.8);self.assertEqual(ep["camera_only_end_pixel_xy"][0,0],4)
    def test_depth_threshold_not_relaxed(self):
        a,b=caches();b["primary"]["depth"][:]=.502
        ep,t=self.trace(a,b,[[4,4]])
        self.assertFalse(ep["valid"][0]);self.assertIn("material_endpoint_surface_depth_error_gt_1mm",t["rejection_reasons"][0])
    def test_half_mm_surface_noise_passes_original_measurement_limit(self):
        a,b=caches();b["primary"]["depth"][:]=.5005
        self.assertTrue(self.trace(a,b,[[4,4]])[0]["valid"][0])
    def test_invalid_start_depth_separately_reported(self):
        a,b=caches();a["primary"]["depth"][4,4]=0
        ep,t=self.trace(a,b,[[4,4]])
        self.assertFalse(ep["valid"][0]);self.assertIn("start_depth_or_geom_invalid",t["rejection_reasons"][0])
    def test_offscreen_never_clips_to_visible_edge(self):
        a,b=caches();b["primary"]["body_pos"][1,0]=2
        ep,t=self.trace(a,b,[[4,4]])
        self.assertFalse(ep["valid"][0]);self.assertGreater(ep["end_pixel_xy"][0,0],7)
    def test_projected_role_failure_is_not_called_absent_object(self):
        a,b=caches();b["primary"]["roles"]["target"][4,4]=False
        ep,t=self.trace(a,b,[[4,4]])
        self.assertTrue(ep["valid"][0]);self.assertIn("material_endpoint_not_in_own_rendered_role",t["rejection_reasons"][0])
    def test_saved_bottom_left_roundtrip_is_exact(self):
        a,b=caches();a["primary"]["camera"]["image_convention"]=-1;b["primary"]["camera"]["image_convention"]=-1
        ep,t=self.trace(a,b,[[4,2]])
        np.testing.assert_allclose(ep["end_pixel_xy"],[[4,2]],atol=1e-9)
    def test_nonfinite_diagnostic_is_null_not_fabricated_zero(self):
        self.assertEqual(m.json_ready(dict(x=np.array([np.nan,np.inf]),y=.1)),dict(x=[None,None],y=.1))
    def test_exact_windows_and_prior_carry_are_frozen(self):
        self.assertEqual(m.WINDOWS,((65,66),(65,67),(66,67)));self.assertIn(57,m.CAPTURE_INDICES)
        self.assertEqual(m.DEPTH_TOLERANCE_M,.001);self.assertEqual(m.LK_TOLERANCE_PX,1.)
    def test_role_membership_uses_independent_side_not_union(self):
        mask=np.zeros((8,8),bool);mask[3,3]=True
        keep,_=m.role_membership([[3,3],[4,4]],mask);np.testing.assert_array_equal(keep,[True,False])
    def test_no_world_body_binding_guessed_from_other_role(self):
        a,b=caches();a["primary"]["geom"][4,4]=-1
        ep,t=self.trace(a,b,[[4,4]])
        self.assertFalse(ep["valid"][0]);self.assertEqual(t["start_body_id"][0],-1)


if __name__=="__main__":unittest.main()

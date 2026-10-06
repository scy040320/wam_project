"""Synthetic CPU contract checks; no empirical recovery labels are asserted."""
import unittest
from unittest.mock import patch
import sys

import numpy as np

from wam_reranking.observation_geometry import (
    contact_interface_witness, depth_visibility, local_rgb_registration,
    project_world_points, saved_image_from_raw, track_rgb_role)


def project(points, convention=1, rotation=None):
    return project_world_points(points, camera_position_m=[0., 0., 0.],
        camera_rotation_world_from_camera=np.eye(3) if rotation is None else rotation,
        fovy_degrees=90., image_shape=(20, 20), image_convention=convention)


def interface_fixture():
    ys, xs = np.indices((20, 20))
    rgb = np.stack((xs * 10, ys * 10, (xs + ys) * 5), axis=-1).astype(np.uint8)
    return dict(projected=project([0., 0., -1.]), depth_m=np.ones((20, 20)),
        target_mask=(xs < 10), finger_mask=(xs >= 10), actual_rgb=rgb,
        rendered_rgb=rgb.copy())


class ObservationGeometryTests(unittest.TestCase):
    def test_camera_projection_front_behind_and_offscreen_are_not_clipped(self):
        result = project([[0., 0., -1.], [0., 0., 1.], [4., 0., -1.]])
        np.testing.assert_allclose(result["pixel_xy"][0], [10., 10.])
        self.assertEqual(result["valid"].tolist(), [True, False, False])
        self.assertTrue(np.isnan(result["pixel_xy"][1]).all())
        self.assertGreater(result["pixel_xy"][2, 0], 19.)

    def test_camera_world_rotation_and_opengl_y(self):
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        np.testing.assert_allclose(project([-1., .2, 0.], rotation=rotation)["pixel_xy"], [[10., 8.]])

    def test_saved_convention_matches_flip_of_rgb_depth_and_masks(self):
        top_down = np.arange(400).reshape(20, 20)
        raw = np.flipud(top_down)
        np.testing.assert_array_equal(saved_image_from_raw(raw, image_convention=1), top_down)
        np.testing.assert_array_equal(saved_image_from_raw(raw, image_convention=-1), raw)
        np.testing.assert_allclose(project([0., .2, -1.], convention=1)["pixel_xy"], [[10., 8.]])
        np.testing.assert_allclose(project([0., .2, -1.], convention=-1)["pixel_xy"], [[10., 11.]])

    def test_invalid_projection_points_never_sample_border(self):
        p = project([[0., 0., 0.], [float("nan"), 0., -1.], [-4., 0., -1.]])
        result = depth_visibility(p, np.ones((20, 20)))
        self.assertFalse(result["valid"].any())
        self.assertTrue(np.isnan(result["sampled_depth_m"]).all())
        np.testing.assert_array_equal(result["sampled_pixel_xy"], -np.ones((3, 2)))

    def test_outside_and_rounding_past_last_pixel_never_clip(self):
        p = project([[.98, 0., -1.], [-1.01, 0., -1.]])
        result = depth_visibility(p, np.ones((20, 20)))
        self.assertFalse(result["valid"].any())
        self.assertTrue(np.isnan(result["sampled_depth_m"]).all())

    def test_occluder_depth_and_surface_matching(self):
        p = project([[0., 0., -1.], [0., 0., -.5]])
        result = depth_visibility(p, np.full((20, 20), .5))
        self.assertEqual(result["visible"].tolist(), [False, True])
        self.assertEqual(result["occluded"].tolist(), [True, False])
        self.assertEqual(result["depth_match"].tolist(), [False, True])

    def test_negative_and_nan_surface_depth_abstain(self):
        for value in (-1., 0., float("nan"), float("inf")):
            result = depth_visibility(project([0., 0., -1.]), np.full((20, 20), value))
            self.assertFalse(result["valid"][0])
            self.assertFalse(result["visible"][0])
            self.assertFalse(result["depth_match"][0])

    def test_local_rgb_detects_small_roi_unreliable_despite_global_match(self):
        f = interface_fixture(); actual = f["actual_rgb"].copy()
        roi = np.zeros((20, 20), bool); roi[9:12, 9:12] = True
        actual[roi] = 255 - actual[roi]
        self.assertFalse(local_rgb_registration(actual, f["rendered_rgb"], roi)["passed"])

    def test_frozen_local_threshold_metrics_and_constant_texture_abstain(self):
        f = interface_fixture(); roi = np.ones((20, 20), bool)
        good = local_rgb_registration(f["actual_rgb"], f["rendered_rgb"], roi)
        self.assertTrue(good["passed"]); self.assertEqual(good["psnr"], 99.)
        blank = np.zeros((20, 20, 3), np.uint8)
        absent = local_rgb_registration(blank, blank, roi)
        self.assertTrue(absent["pixel_metrics_passed"])
        self.assertFalse(absent["texture_sufficient"])
        self.assertFalse(absent["passed"])
        self.assertEqual(absent["reason"], "insufficient_local_texture")

    def test_nan_rgb_and_empty_roi_abstain(self):
        f = interface_fixture(); invalid = f["actual_rgb"].astype(float); invalid[10, 10, 0] = np.nan
        self.assertFalse(local_rgb_registration(invalid, f["rendered_rgb"], np.ones((20, 20), bool))["measurement_valid"])
        self.assertFalse(local_rgb_registration(f["actual_rgb"], f["rendered_rgb"], np.zeros((20, 20), bool))["passed"])

    def test_visible_interface_is_not_physical_contact_certificate(self):
        f = interface_fixture(); result = contact_interface_witness(**f)
        self.assertTrue(result["passed"])
        self.assertFalse(result["physical_contact_certified"])
        self.assertFalse(result["physical_force_closure_certified"])
        self.assertTrue(all(v > 0 for v in result["role_depth_matched_patch_pixels"].values()))

    def test_roles_elsewhere_cooccur_but_do_not_witness_interface(self):
        f = interface_fixture(); ys, xs = np.indices((20, 20))
        f["target_mask"] = xs < 3; f["finger_mask"] = xs > 16
        result = contact_interface_witness(**f)
        self.assertTrue(result["role_reference_floor_passed"])
        self.assertFalse(result["passed"])
        self.assertIn("roles_not_both_at_contact_patch", result["reasons"])

    def test_role_depth_occlusion_prevents_contact_witness(self):
        f = interface_fixture(); f["depth_m"][f["finger_mask"]] = .9
        result = contact_interface_witness(**f)
        self.assertFalse(result["passed"])
        self.assertEqual(result["role_depth_matched_patch_pixels"]["finger"], 0)

    def test_one_role_local_rgb_mismatch_prevents_witness(self):
        f = interface_fixture(); f["actual_rgb"][f["finger_mask"]] = 0
        result = contact_interface_witness(**f)
        self.assertFalse(result["finger_registration"]["passed"])
        self.assertFalse(result["passed"])

    def test_all_black_contact_patch_abstains(self):
        f = interface_fixture(); f["actual_rgb"][:] = 0; f["rendered_rgb"][:] = 0
        result = contact_interface_witness(**f)
        self.assertFalse(result["passed"])
        self.assertFalse(result["shared_patch_registration"]["texture_sufficient"])

    def test_textured_background_cannot_replace_role_local_texture(self):
        f = interface_fixture()
        f["actual_rgb"][:] = 50
        f["rendered_rgb"][:] = 50
        # One white background pixel gives the shared patch texture while both
        # actual role surfaces remain constant and cannot certify registration.
        f["target_mask"][8, 10] = False; f["finger_mask"][8, 10] = False
        f["actual_rgb"][8, 10] = 255; f["rendered_rgb"][8, 10] = 255
        result = contact_interface_witness(**f)
        self.assertTrue(result["shared_patch_registration"]["passed"])
        self.assertFalse(result["target_registration"]["texture_sufficient"])
        self.assertFalse(result["finger_registration"]["texture_sufficient"])
        self.assertFalse(result["supports_interface_observability"])

    def test_role_floor_is_reference_support_not_contact_patch_count(self):
        result = contact_interface_witness(**interface_fixture())
        self.assertTrue(result["passed"])
        self.assertLess(result["role_patch_pixels"]["target"], 16)
        self.assertGreaterEqual(result["role_visible_pixels"]["target"], 16)

    def test_contact_outside_partial_patch_or_behind_abstains(self):
        for point in ([4., 0., -1.], [.8, 0., -1.], [0., 0., 1.], [np.nan, 0., -1.]):
            f = interface_fixture(); f["projected"] = project(point)
            result = contact_interface_witness(**f)
            self.assertFalse(result["passed"])
            self.assertFalse(result["full_patch_in_bounds"])

    def test_invalid_calibration_or_binary_masks_rejected(self):
        with self.assertRaises(ValueError): project([0., 0., -1.], convention=0)
        with self.assertRaises(ValueError): project([0., 0., -1.], rotation=np.zeros((3, 3)))
        f = interface_fixture(); f["target_mask"] = np.full((20, 20), .5)
        with self.assertRaises(ValueError): contact_interface_witness(**f)


class ControlledOpticalFlow:
    """Backend test double only; these tests do not validate OpenCV's solver."""
    COLOR_RGB2GRAY = 7
    TERM_CRITERIA_EPS = 2
    TERM_CRITERIA_COUNT = 1

    def __init__(self, back_error=0., corners=3, failed_status=False):
        self.calls = 0; self.back_error = back_error
        self.corner_count = corners; self.failed_status = failed_status

    def cvtColor(self, array, _):
        return array[:, :, 0]

    def goodFeaturesToTrack(self, image, **kwargs):
        return np.array([[6., 6.], [8., 6.], [6., 8.]], np.float32)[:self.corner_count, None, :]

    def calcOpticalFlowPyrLK(self, start, end, points, _, **kwargs):
        self.calls += 1
        delta = [1., 0.] if self.calls == 1 else [-1. + self.back_error, 0.]
        status = np.ones((len(points), 1), np.uint8)
        if self.failed_status: status[-1] = 0
        return points + np.array(delta, np.float32), status, np.zeros((len(points), 1))


class RGBTrackingContractTests(unittest.TestCase):
    def inputs(self):
        f = interface_fixture()
        return f["actual_rgb"], f["rendered_rgb"], np.ones((20, 20), bool), np.ones((20, 20), bool)

    def track(self, backend, endpoints=None, inputs=None):
        with patch.dict(sys.modules, {"cv2": backend}):
            return track_rgb_role(*(self.inputs() if inputs is None else inputs), projected_endpoints=endpoints)

    def test_rgb_tracks_alone_are_only_proxy(self):
        result = self.track(ControlledOpticalFlow())
        self.assertTrue(result["verified"])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["median_delta_px"], [1., 0.])
        self.assertEqual(result["evidence_kind"], "RGB_tracking_proxy")
        self.assertFalse(result["geometry_motion_support"])
        self.assertFalse(result["metric_world_motion_certified"])

    def test_geometry_callback_ties_actual_tracks_to_surface_endpoints(self):
        def geometry(corners):
            return dict(start_pixel_xy=corners, end_pixel_xy=corners + [1., 0.])
        result = self.track(ControlledOpticalFlow(), geometry)
        self.assertTrue(result["geometry_motion_support"])
        self.assertEqual(result["geometry_matched_count"], 3)
        self.assertEqual(result["median_endpoint_error_px"], 0.)
        self.assertFalse(result["metric_world_motion_certified"])

    def test_private_camera_only_endpoint_info_preserves_acceptance_and_correspondence(self):
        def geometry(corners):
            return dict(start_pixel_xy=corners, end_pixel_xy=corners + [1., 0.],
                        camera_only_end_pixel_xy=corners + [.25, .5])
        result = self.track(ControlledOpticalFlow(), geometry)
        self.assertTrue(result["geometry_motion_support"])
        self.assertEqual(result["count"], 3)
        for track in result["tracks"]:
            np.testing.assert_allclose(track["projected_end_pixel_xy"], np.array(track["start_pixel_xy"]) + [1., 0.])
            np.testing.assert_allclose(track["camera_only_end_pixel_xy"], np.array(track["start_pixel_xy"]) + [.25, .5])
        self.assertFalse(result["metric_world_motion_certified"])

    def test_camera_only_info_is_matched_by_surface_pair_not_track_array_index(self):
        p = np.array([[6., 6.], [8., 6.], [6., 8.]])[::-1]
        endpoints = dict(start_pixel_xy=p, end_pixel_xy=p + [1., 0.],
                         camera_only_end_pixel_xy=p + [.25, .5])
        result = self.track(ControlledOpticalFlow(), endpoints)
        for track in result["tracks"]:
            np.testing.assert_allclose(track["camera_only_end_pixel_xy"], np.array(track["start_pixel_xy"]) + [.25, .5])

    def test_missing_nonfinite_or_malformed_camera_only_info_is_explicit(self):
        p = np.array([[6., 6.], [8., 6.], [6., 8.]])
        endpoints = dict(start_pixel_xy=p, end_pixel_xy=p + [1., 0.])
        baseline = self.track(ControlledOpticalFlow(), endpoints)
        self.assertTrue(all(t["camera_only_end_pixel_xy"] is None for t in baseline["tracks"]))
        endpoints["camera_only_end_pixel_xy"] = p.copy()
        endpoints["camera_only_end_pixel_xy"][0, 0] = np.nan
        result = self.track(ControlledOpticalFlow(), endpoints)
        self.assertEqual(result["count"], baseline["count"])
        self.assertIsNone(result["tracks"][0]["camera_only_end_pixel_xy"])
        endpoints["camera_only_end_pixel_xy"] = np.zeros((2, 2))
        with self.assertRaises(ValueError): self.track(ControlledOpticalFlow(), endpoints)

    def test_forward_backward_above_one_pixel_and_status_fail_abstain(self):
        result = self.track(ControlledOpticalFlow(back_error=1.1))
        self.assertFalse(result["verified"])
        self.assertEqual(result["forward_backward_count"], 0)
        self.assertFalse(self.track(ControlledOpticalFlow(failed_status=True))["verified"])

    def test_end_track_must_stay_in_same_actual_rendered_role(self):
        data = list(self.inputs()); data[3][8, 7] = False
        result = self.track(ControlledOpticalFlow(), inputs=data)
        self.assertEqual(result["role_endpoint_count"], 2)
        self.assertFalse(result["verified"])

    def test_wrong_geometry_or_one_point_cannot_certify_three_tracks(self):
        for endpoints in (dict(start_pixel_xy=[[6., 6.]], end_pixel_xy=[[7., 6.]]),
                          dict(start_pixel_xy=[[6., 6.], [8., 6.], [6., 8.]],
                               end_pixel_xy=[[9., 6.], [11., 6.], [9., 8.]])):
            result = self.track(ControlledOpticalFlow(), endpoints)
            self.assertFalse(result["verified"])
            self.assertFalse(result["geometry_motion_support"])

    def test_nan_or_invalid_geometry_is_not_motion_support(self):
        p = np.array([[6., 6.], [8., 6.], [6., 8.]])
        endpoints = dict(start_pixel_xy=p, end_pixel_xy=p + [1., 0.], valid=np.array([True, True, False]))
        valid_copy = endpoints["valid"].copy()
        result = self.track(ControlledOpticalFlow(), endpoints)
        np.testing.assert_array_equal(endpoints["valid"], valid_copy)
        self.assertFalse(result["verified"])
        endpoints["end_pixel_xy"][0, 0] = np.nan
        self.assertFalse(self.track(ControlledOpticalFlow(), endpoints)["verified"])

    def test_too_few_corners_uniform_texture_or_absent_cv2_abstains(self):
        self.assertFalse(self.track(ControlledOpticalFlow(corners=2))["verified"])
        data = list(self.inputs()); data[0] = np.zeros_like(data[0])
        result = self.track(ControlledOpticalFlow(), inputs=data)
        self.assertEqual(result["reason"], "insufficient_actual_role_texture")
        result = self.track(None)
        self.assertEqual(result["reason"], "opencv_backend_unavailable")

    def test_nonfinite_actual_rgb_never_enters_backend(self):
        data = list(self.inputs()); data[0] = data[0].astype(float); data[0][0, 0, 0] = np.nan
        result = self.track(ControlledOpticalFlow(), inputs=data)
        self.assertEqual(result["reason"], "invalid_actual_rgb")


if __name__ == "__main__":
    unittest.main()

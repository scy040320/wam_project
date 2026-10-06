from copy import deepcopy
import unittest
import numpy as np
from wam_reranking.observable_separation import negative_contact_witness


def fixture():
    rgb = np.zeros((40, 40, 3), np.uint8)
    target = np.zeros((40, 40), bool); target[4:12, 4:12] = True
    finger = np.zeros((40, 40), bool); finger[4:12, 22:30] = True
    yy, xx = np.indices(target.shape)
    rgb[target] = (60 + 90*((xx+yy) % 2))[target, None]
    rgb[finger] = (70 + 90*((xx+yy) % 2))[finger, None]
    good = dict(passed=True, mean_abs=0., p95=0., fraction_gt5=0., psnr=99.)
    return dict(actual_rgb=rgb, target_mask=target, finger_mask=finger,
        target_registration=deepcopy(good), finger_registration=deepcopy(good),
        target_identity_certified=True, finger_identity_certified=True,
        target_repeat_noise_px=.1, finger_repeat_noise_px=.1,
        target_repeat_rgb_mean_abs=.1, finger_repeat_rgb_mean_abs=.1,
        fresh_geometry_registered=True, physical_contact=False, side="left")


class SeparationTests(unittest.TestCase):
    def test_registered_independent_surfaces_with_real_gap_are_contact_negative_only(self):
        x = fixture(); before = deepcopy(x); result = negative_contact_witness(**x)
        self.assertIsNotNone(result); self.assertFalse(result["physics_contact"])
        self.assertFalse(result["certifies_negative_grasp"])
        self.assertGreater(result["actual_surface_gap_px"], 2)
        np.testing.assert_array_equal(x["actual_rgb"], before["actual_rgb"])

    def test_contact_true_unknown_or_number_is_never_negative(self):
        for value in (True, None, 0):
            x = fixture(); x["physical_contact"] = value
            self.assertIsNone(negative_contact_witness(**x))

    def test_hidden_single_finger_cannot_borrow_eef_or_other_finger_identity(self):
        x = fixture(); x["finger_identity_certified"] = False
        self.assertIsNone(negative_contact_witness(**x))

    def test_no_registration_or_role_registration_failure_stays_unknown(self):
        for field in ("fresh_geometry_registered", "target_identity_certified"):
            x = fixture(); x[field] = False
            self.assertIsNone(negative_contact_witness(**x))
        x = fixture(); x["finger_registration"]["fraction_gt5"] = .061
        self.assertIsNone(negative_contact_witness(**x))

    def test_noisy_separation_does_not_pass_same_candidate_noise(self):
        x = fixture(); x["finger_repeat_noise_px"] = 4.
        self.assertIsNone(negative_contact_witness(**x))

    def test_missing_noise_is_not_silently_zero(self):
        x = fixture(); x["target_repeat_noise_px"] = None
        self.assertIsNone(negative_contact_witness(**x))

    def test_role_constant_but_background_textured_is_not_certified(self):
        x = fixture(); x["actual_rgb"][x["target_mask"]] = 60
        x["actual_rgb"][35, 35] = 255
        self.assertIsNone(negative_contact_witness(**x))

    def test_rgb_structure_without_role_silhouette_contrast_is_not_boundary(self):
        x = fixture(); x["actual_rgb"][:] = 60
        x["actual_rgb"][6, 6] = 150; x["actual_rgb"][6, 24] = 150
        self.assertIsNone(negative_contact_witness(**x))

    def test_overlap_or_tiny_role_is_unknown(self):
        x = fixture(); x["finger_mask"] = x["target_mask"].copy()
        self.assertIsNone(negative_contact_witness(**x))
        x = fixture(); x["finger_mask"][:] = False; x["finger_mask"][5, 24] = True
        self.assertIsNone(negative_contact_witness(**x))

    def test_close_untextured_tip_cannot_be_ignored_for_convenient_gap(self):
        x = fixture(); x["finger_mask"][6, 12] = True
        x["actual_rgb"][6, 12] = 0
        self.assertIsNone(negative_contact_witness(**x))

    def test_support_side_not_invented(self):
        x = fixture(); x["side"] = "support"
        with self.assertRaises(ValueError): negative_contact_witness(**x)


if __name__ == "__main__": unittest.main()

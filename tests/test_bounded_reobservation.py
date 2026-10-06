import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from wam_reranking.reobservation_supervision import derive_reobservation_availability

spec = importlib.util.spec_from_file_location("bounded_reobservation_test_script",
    Path(__file__).resolve().parents[1] / "scripts/collect_bounded_reobservation.py")
script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
helper_spec = importlib.util.spec_from_file_location("bounded_observation_helpers",
    Path(__file__).resolve().parents[1] / "scripts/build_observable_recovery_labels.py")
helpers = importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(helpers)


def rgb_metrics():
    return dict(passed=True, mean_abs=0., p95=0., fraction_gt5=0., psnr=99.,
                texture_sufficient=True, actual_texture_range=30.)


def actual_track(available):
    points = [dict(start_pixel_xy=[float(i), 5.], end_pixel_xy=[float(i), 5.],
                   projected_end_pixel_xy=[float(i), 5.], camera_only_end_pixel_xy=[float(i), 5.],
                   forward_backward_error_px=0., projected_endpoint_error_px=0.) for i in range(3)]
    return dict(verified=available, count=3 if available else 0,
                geometry_motion_support=available, evidence_kind="actual_RGB_geometry_correspondence",
                tracks=points if available else [])


def frames(available=False):
    result = []
    for step in range(17):
        view = dict(global_rgb=rgb_metrics(), fresh_geometry_registered_to_actual_input=True,
                    repeat_noise=dict(target=dict(centroid_noise_px=0., rgb_repeat=rgb_metrics())),
                    roles=dict(target=dict(rendered_pixels=24, local_rgb=rgb_metrics())))
        intervals = []
        if step >= 2:
            for name in ("primary", "wrist"):
                intervals.append(dict(start_step=1, end_step=step, view=name,
                    tracks=dict(target=actual_track(available)),
                    same_candidate_repeat_noise=dict(target=0.)))
        result.append(dict(step=step, observation_evidence=dict(
            same_candidate_repeat_qc=dict(physics_passed=True, all_contact_atoms_match=True,
                                         qpos_qvel_max_abs=0., qpos_qvel_p95=0.),
            views={name: copy.deepcopy(view) for name in ("primary", "wrist")},
            actual_rgb_temporal_intervals=intervals)))
    return result


class ProbePlanTests(unittest.TestCase):
    def test_all_arms_share_gripper_and_initial_motion(self):
        original = np.zeros((16, 7)); original[:, 6] = .7
        for arm in script.ARMS:
            planned = script.planned_probe(arm, original)
            np.testing.assert_array_equal(planned[:, 6], np.full(16, np.float32(.7)))
            np.testing.assert_array_equal(planned[:2, :6], np.zeros((2, 6)))

    def test_hold_moves_no_end_effector(self):
        np.testing.assert_array_equal(script.planned_probe("hold", np.zeros((16, 7))), np.zeros((16, 7)))

    def test_distinct_predeclared_movements_only(self):
        plans = [script.planned_probe(arm, np.zeros((16, 7))) for arm in script.ARMS]
        self.assertEqual(len({p.tobytes() for p in plans}), 4)
        for plan in plans: self.assertLessEqual(np.max(np.abs(plan[:, :6])), .2 + 1e-7)

    def test_reject_unknown_arm(self):
        with self.assertRaises(ValueError): script.planned_probe("best_of_outcomes", np.zeros((16, 7)))

    def test_reject_short_plan(self):
        with self.assertRaises(ValueError): script.planned_probe("hold", np.zeros((7, 7)))

    def test_reject_nonfinite_source(self):
        for value in (float("nan"), float("inf")):
            plan = np.zeros((16, 7)); plan[0, 6] = value
            with self.assertRaises(ValueError): script.planned_probe("hold", plan)

    def test_real_cosmos_overshoot_clips_only_new_auxiliary_gripper(self):
        plan = np.zeros((16, 7)); plan[0, 6] = -1.0096359252929688
        before = plan.copy()
        for arm in script.ARMS:
            actions = script.planned_probe(arm, plan)
            np.testing.assert_array_equal(actions[:, 6], np.full(16, -1.))
        np.testing.assert_array_equal(plan, before)


class ObservationAvailabilityTests(unittest.TestCase):
    def derive(self, records, arm="hold"):
        return derive_reobservation_availability(records, identity=dict(dataset="aux_test"), arm=arm,
                                                certificate_helpers=helpers)

    def test_empty_actual_tracks_are_observer_insufficient_not_absent_target(self):
        value = self.derive(frames())
        self.assertIs(value["frames"][16]["value"], False)
        self.assertIs(value["frames"][16]["target_physical_absence_inferred"], False)
        self.assertFalse(value["observed_availability_recovered"])
        self.assertFalse(value["ranking_training_ready"])

    def test_persistent_primary_conflict_does_not_disappear(self):
        records = frames(True)
        for record in records: record["observation_evidence"]["views"]["primary"]["global_rgb"]["passed"] = False
        value = self.derive(records, "peek_up")
        self.assertFalse(value["frames"][16]["views"]["primary"])
        self.assertTrue(value["frames"][16]["views"]["wrist"])
        self.assertFalse(value["cross_view_conflict_resolved"]["mask"])

    def test_actually_recovered_tracks_have_before_and_after(self):
        records = frames(False)
        records[16] = frames(True)[16]
        value = self.derive(records, "peek_left")
        self.assertTrue(value["observed_availability_recovered"])
        self.assertFalse(value["task_predicate_mapping_admitted"])

    def test_already_available_is_not_recovery(self):
        self.assertFalse(self.derive(frames(True))["observed_availability_recovered"])

    def test_frame_zero_and_one_remain_masked(self):
        value = self.derive(frames(True))
        for index in (0, 1):
            self.assertFalse(value["frames"][index]["mask"])
            self.assertIsNone(value["frames"][index]["value"])

    def test_no_same_arm_qc_cannot_label(self):
        records = frames(); records[5]["observation_evidence"]["same_candidate_repeat_qc"]["physics_passed"] = False
        with self.assertRaises(ValueError): self.derive(records)

    def test_insufficient_pixels_not_admitted(self):
        records = frames(True)
        for record in records:
            for view in record["observation_evidence"]["views"].values(): view["roles"]["target"]["rendered_pixels"] = 15
        self.assertFalse(self.derive(records)["frames"][16]["value"])

    def test_registration_failure_not_admitted(self):
        records = frames(True)
        for record in records:
            for view in record["observation_evidence"]["views"].values(): view["roles"]["target"]["local_rgb"]["passed"] = False
        self.assertFalse(self.derive(records)["frames"][16]["value"])

    def test_missing_repeat_noise_is_not_assumed_zero(self):
        records = frames(True)
        for record in records:
            for interval in record["observation_evidence"]["actual_rgb_temporal_intervals"]: interval["same_candidate_repeat_noise"]["target"] = None
        self.assertFalse(self.derive(records)["frames"][16]["value"])

    def test_flag_alone_does_not_relax_numeric_repeat_gate(self):
        records = frames(True)
        records[8]["observation_evidence"]["same_candidate_repeat_qc"]["qpos_qvel_max_abs"] = .002
        with self.assertRaises(ValueError): self.derive(records)

    def test_duplicate_actual_corner_cannot_enable_observability(self):
        records = frames(True)
        for record in records:
            for interval in record["observation_evidence"]["actual_rgb_temporal_intervals"]:
                point = interval["tracks"]["target"]["tracks"][0]
                interval["tracks"]["target"]["tracks"] = [point] * 3
        self.assertFalse(self.derive(records)["frames"][16]["value"])

    def test_cached_geometry_flag_blocks_availability(self):
        records = frames(True)
        for record in records:
            for view in record["observation_evidence"]["views"].values():
                view["fresh_geometry_registered_to_actual_input"] = False
        self.assertFalse(self.derive(records)["frames"][16]["value"])

    def test_blank_own_role_texture_blocks_availability(self):
        records = frames(True)
        for record in records:
            for view in record["observation_evidence"]["views"].values():
                view["roles"]["target"]["local_rgb"]["texture_sufficient"] = False
        self.assertFalse(self.derive(records)["frames"][16]["value"])

    def test_short_tail_cannot_be_padded_to_probe(self):
        with self.assertRaises(ValueError): self.derive(frames()[:8])

    def test_future_interval_rejected(self):
        records = frames(True)
        records[5]["observation_evidence"]["actual_rgb_temporal_intervals"][0]["start_step"] = 6
        with self.assertRaises(ValueError): self.derive(records)

    def test_truth_and_terminal_success_do_not_enable_labels(self):
        records = frames()
        for record in records: record["physics"] = dict(grasped=True, success=True)
        self.assertFalse(self.derive(records)["observed_availability_recovered"])


if __name__ == "__main__": unittest.main()

"""Pure CPU contract tests; synthetic physics is NOT an experimental result."""
from copy import deepcopy
from dataclasses import replace
import unittest

try:
    from temporal_recovery_teacher import derive_temporal_recovery, FROZEN_CONTRACT
except ModuleNotFoundError:
    from wam_reranking.temporal_recovery_teacher import derive_temporal_recovery, FROZEN_CONTRACT


def frames(moving=False, contact=False):
    result = []
    for step in range(17):
        x = step * .004 if moving else 0.
        result.append(dict(step=step, physics=dict(
            eef_pos_m=[x, 0., .8], target_pos_m=[x, 0., .79],
            gripper_aperture_m=.02, left_finger_target_contact=contact,
            right_finger_target_contact=contact, target_support_contact=not contact),
            observability=dict(target_visible=True, eef_visible=True,
                primary_reliable=True, wrist_reliable=True,
                identity_consistent=True, cross_view_consistent=True,
                contact_observable=True, target_motion_observable=True,
                joint_state_observable=True)))
    return result


def atom(result, step, name):
    return result["frames"][step]["physics"][name]


class TemporalTeacherTests(unittest.TestCase):
    def test_single_contact_is_not_grasp(self):
        data = frames(moving=True)
        for key in ("left_finger_target_contact", "right_finger_target_contact"):
            data[5]["physics"][key] = True
        result = derive_temporal_recovery(data)
        self.assertTrue(atom(result, 5, "left_finger_target_contact")["value"])
        self.assertNotIn("carried_sufficient_evidence", result["first_measured_events"])

    def test_stationary_double_contact_does_not_establish_carrying(self):
        result = derive_temporal_recovery(frames(contact=True))
        self.assertIsNone(atom(result, 16, "carried_sufficient_evidence")["value"])

    def test_three_frame_common_motion_is_positive_support(self):
        result = derive_temporal_recovery(frames(moving=True, contact=True))
        self.assertEqual(result["first_measured_events"]["carried_sufficient_evidence"], 2)
        self.assertTrue(atom(result, 2, "carried_sufficient_evidence")["supervision_mask"])

    def test_endpoint_event_does_not_fill_earlier_frames(self):
        data = frames()
        data[16]["physics"]["target_pos_m"][2] += .02
        result = derive_temporal_recovery(data)
        self.assertEqual(result["first_measured_events"]["target_vertical_rise"], 16)
        self.assertFalse(atom(result, 15, "target_vertical_rise")["value"])
        self.assertIsNone(atom(result, 16, "lifted_sufficient_evidence")["value"])

    def test_cabinet_height_change_is_not_lift(self):
        data = frames(moving=True, contact=True)
        for f in data:
            f["physics"]["target_joint_qpos"] = f["step"] * .002
            f["physics"]["target_pos_m"][2] += f["step"] * .02
        result = derive_temporal_recovery(data, entity_kind="articulated", joint_unit="m")
        self.assertTrue(atom(result, 1, "joint_state_changed")["value"])
        self.assertIsNone(atom(result, 16, "target_vertical_rise")["value"])
        self.assertIsNone(atom(result, 16, "carried_sufficient_evidence")["value"])

    def test_unknown_view_quality_does_not_guess_physical_cause(self):
        data = frames(moving=True, contact=True)
        for f in data:
            f["observability"]["cross_view_consistent"] = None
        result = derive_temporal_recovery(data)
        self.assertTrue(atom(result, 2, "carried_sufficient_evidence")["measurement_valid"])
        self.assertFalse(atom(result, 2, "carried_sufficient_evidence")["supervision_mask"])
        self.assertNotIn("cause", result)

    def test_invisible_evidence_remains_sim_truth_but_is_masked(self):
        data = frames(moving=True, contact=True)
        data[1]["observability"]["target_visible"] = False
        result = derive_temporal_recovery(data)
        self.assertTrue(atom(result, 2, "carried_sufficient_evidence")["value"])
        self.assertFalse(atom(result, 2, "carried_sufficient_evidence")["supervision_mask"])

    def test_missing_contact_is_not_negative(self):
        data = frames(moving=True, contact=True)
        for f in data:
            del f["physics"]["left_finger_target_contact"]
        result = derive_temporal_recovery(data)
        self.assertFalse(atom(result, 16, "left_finger_target_contact")["measurement_valid"])
        self.assertIsNone(atom(result, 16, "carried_sufficient_evidence")["value"])

    def test_command_close_cannot_generate_physical_grasp(self):
        data = frames(moving=True)
        for f in data:
            f["physics"]["gripper_command"] = 1.
        result = derive_temporal_recovery(data)
        self.assertNotIn("carried_sufficient_evidence", result["first_measured_events"])

    def test_release_requires_prior_carry_and_persistent_separation(self):
        data = frames(moving=True, contact=True)
        for f in data[5:]:
            f["physics"]["left_finger_target_contact"] = False
            f["physics"]["right_finger_target_contact"] = False
            f["physics"]["gripper_aperture_m"] = .03
            f["physics"]["target_pos_m"][0] = .016
        result = derive_temporal_recovery(data)
        self.assertEqual(result["first_measured_events"]["released_sufficient_evidence"], 6)
        self.assertIsNone(atom(result, 5, "released_sufficient_evidence")["value"])
        self.assertFalse(atom(result, 6, "holding_after_measured_release")["value"])

    def test_open_without_prior_carry_is_not_release(self):
        data = frames(moving=True)
        for f in data[4:]:
            f["physics"]["gripper_aperture_m"] = .04
        result = derive_temporal_recovery(data)
        self.assertNotIn("released_sufficient_evidence", result["first_measured_events"])

    def test_nonshared_motion_is_not_carry(self):
        data = frames(moving=True, contact=True)
        for f in data:
            f["physics"]["target_pos_m"][0] = 0.
        result = derive_temporal_recovery(data)
        self.assertNotIn("carried_sufficient_evidence", result["first_measured_events"])

    def test_large_relative_drift_is_not_carry(self):
        data = frames(moving=True, contact=True)
        for f in data:
            f["physics"]["target_pos_m"][0] *= 2
        result = derive_temporal_recovery(data)
        self.assertNotIn("carried_sufficient_evidence", result["first_measured_events"])

    def test_lift_needs_object_not_only_eef_height(self):
        data = frames(moving=True, contact=True)
        for f in data:
            f["physics"]["eef_pos_m"][2] += f["step"] * .004
        result = derive_temporal_recovery(data)
        self.assertNotIn("lifted_sufficient_evidence", result["first_measured_events"])

    def test_lift_needs_recorded_support_loss_and_carry(self):
        data = frames(moving=True, contact=True)
        for f in data:
            f["physics"]["eef_pos_m"][2] += f["step"] * .004
            f["physics"]["target_pos_m"][2] += f["step"] * .004
            f["physics"]["target_support_contact"] = f["step"] < 3
        result = derive_temporal_recovery(data)
        self.assertEqual(result["first_measured_events"]["lifted_sufficient_evidence"], 5)

    def test_table_push_with_common_motion_is_not_carry(self):
        data = frames(moving=True, contact=True)
        for f in data:
            f["physics"]["target_support_contact"] = True
        result = derive_temporal_recovery(data)
        self.assertNotIn("carried_sufficient_evidence", result["first_measured_events"])

    def test_release_not_propagated_through_unverified_recontact(self):
        data = frames(moving=True, contact=True)
        for f in data[5:10]:
            f["physics"]["left_finger_target_contact"] = False
            f["physics"]["right_finger_target_contact"] = False
            f["physics"]["gripper_aperture_m"] = .03
            f["physics"]["target_pos_m"][0] = .016
        result = derive_temporal_recovery(data)
        self.assertFalse(atom(result, 6, "holding_after_measured_release")["value"])
        self.assertIsNone(atom(result, 10, "holding_after_measured_release")["value"])

    def test_invisible_before_masks_relative_transition_not_endpoint_position(self):
        data = frames(moving=True)
        data[0]["observability"]["target_visible"] = False
        result = derive_temporal_recovery(data)
        self.assertTrue(atom(result, 16, "target_displacement_m")["measurement_valid"])
        self.assertFalse(atom(result, 16, "target_displacement_m")["supervision_mask"])
        self.assertTrue(atom(result, 16, "target_relative_to_eef_m")["supervision_mask"])

    def test_release_mask_requires_observable_prior_carry(self):
        data = frames(moving=True, contact=True)
        for f in data[:5]:
            f["observability"]["target_visible"] = False
        for f in data[5:]:
            f["physics"]["left_finger_target_contact"] = False
            f["physics"]["right_finger_target_contact"] = False
            f["physics"]["gripper_aperture_m"] = .03
            f["physics"]["target_pos_m"][0] = .016
        result = derive_temporal_recovery(data)
        self.assertTrue(atom(result, 6, "released_sufficient_evidence")["value"])
        self.assertFalse(atom(result, 6, "released_sufficient_evidence")["supervision_mask"])

    def test_visibility_recovery_and_degradation_are_distinct(self):
        data = frames()
        for f in data[:4]:
            f["observability"]["target_visible"] = False
        data[9]["observability"]["target_visible"] = False
        result = derive_temporal_recovery(data)
        events = [(x["step"], x["event"]) for x in result["temporal_transitions"] if x["name"] == "target_visible"]
        self.assertEqual(events, [(4, "recovered"), (9, "degraded"), (10, "recovered")])

    def test_unknown_before_is_establishment_not_recovery(self):
        data = frames()
        data[0]["observability"]["target_visible"] = None
        result = derive_temporal_recovery(data)
        events = [x for x in result["temporal_transitions"] if x["name"] == "target_visible"]
        self.assertEqual(events[0]["event"], "established")

    def test_contact_is_masked_without_visual_contact_certification(self):
        data = frames(contact=True)
        data[0]["observability"]["contact_observable"] = None
        result = derive_temporal_recovery(data)
        self.assertTrue(atom(result, 0, "left_finger_target_contact")["value"])
        self.assertFalse(atom(result, 0, "left_finger_target_contact")["supervision_mask"])

    def test_missing_camera_flags_do_not_default_reliable(self):
        data = frames(moving=True, contact=True)
        for f in data:
            f["observability"] = {}
        result = derive_temporal_recovery(data)
        self.assertFalse(atom(result, 16, "carried_sufficient_evidence")["supervision_mask"])

    def test_nans_rejected(self):
        data = frames()
        data[3]["physics"]["target_pos_m"][0] = float("nan")
        with self.assertRaises(ValueError):
            derive_temporal_recovery(data)

    def test_bad_indices_or_short_blocks_rejected(self):
        with self.assertRaises(ValueError):
            derive_temporal_recovery(frames()[:-1])
        data = frames()
        data[4]["step"] = 3
        with self.assertRaises(ValueError):
            derive_temporal_recovery(data)

    def test_gt_outcome_or_prediction_cannot_enter(self):
        for key in ("success", "attribution", "predicted_cause", "condition"):
            data = frames()
            data[5]["physics"][key] = True
            with self.assertRaises(ValueError):
                derive_temporal_recovery(data)

    def test_joint_units_are_required(self):
        with self.assertRaises(ValueError):
            derive_temporal_recovery(frames(), entity_kind="articulated")

    def test_future_changes_cannot_change_earlier_labels(self):
        data = frames(moving=True, contact=True)
        a = derive_temporal_recovery(data)
        changed = deepcopy(data)
        for f in changed[9:]:
            f["physics"]["target_pos_m"] = [4., 3., 2.]
            f["physics"]["left_finger_target_contact"] = False
        b = derive_temporal_recovery(changed)
        self.assertEqual(a["frames"][:9], b["frames"][:9])

    def test_measurement_contract_is_frozen(self):
        with self.assertRaises(AttributeError):
            FROZEN_CONTRACT.vertical_rise_m = 0.

    def test_mutated_threshold_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            derive_temporal_recovery(frames(), contract=replace(FROZEN_CONTRACT, vertical_rise_m=0.))

    def test_visible_but_unobservable_joint_keeps_gt_and_masks_learning(self):
        data = frames()
        for f in data:
            f["physics"]["target_joint_qpos"] = f["step"] * .02
            f["observability"]["joint_state_observable"] = None
        result = derive_temporal_recovery(data, entity_kind="articulated", joint_unit="rad")
        self.assertTrue(atom(result, 16, "joint_state_changed")["value"])
        self.assertFalse(atom(result, 16, "joint_state_changed")["supervision_mask"])

    def test_carry_mask_cannot_bypass_contact_observability(self):
        data = frames(moving=True, contact=True)
        for f in data:
            f["observability"]["contact_observable"] = None
        result = derive_temporal_recovery(data)
        self.assertTrue(atom(result, 2, "carried_sufficient_evidence")["value"])
        self.assertFalse(atom(result, 2, "carried_sufficient_evidence")["supervision_mask"])

    def test_visible_static_projection_does_not_certify_world_motion(self):
        data = frames(moving=True)
        for f in data:
            f["observability"]["target_motion_observable"] = None
        result = derive_temporal_recovery(data)
        self.assertTrue(atom(result, 16, "target_displacement_m")["measurement_valid"])
        self.assertFalse(atom(result, 16, "target_displacement_m")["supervision_mask"])

    def test_single_reliable_view_does_not_require_cross_view_certificate(self):
        data = frames(moving=True, contact=True)
        for f in data:
            f["observability"]["wrist_reliable"] = False
            f["observability"]["cross_view_consistent"] = None
        result = derive_temporal_recovery(data)
        self.assertTrue(atom(result, 2, "carried_sufficient_evidence")["supervision_mask"])


    def test_lift_mask_requires_observable_initial_support_and_vertical_reference(self):
        for missing in ("target_visible", "contact_observable"):
            data = frames(moving=True, contact=True)
            for f in data:
                f["physics"]["eef_pos_m"][2] += f["step"] * .004
                f["physics"]["target_pos_m"][2] += f["step"] * .004
                f["physics"]["target_support_contact"] = f["step"] < 3
            data[0]["observability"][missing] = False
            result = derive_temporal_recovery(data)
            self.assertTrue(atom(result, 5, "lifted_sufficient_evidence")["value"])
            self.assertTrue(atom(result, 5, "carried_sufficient_evidence")["supervision_mask"])
            self.assertFalse(atom(result, 5, "lifted_sufficient_evidence")["supervision_mask"])

if __name__ == "__main__":
    unittest.main()

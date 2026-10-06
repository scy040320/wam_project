"""Synthetic CPU proofs of contract logic, not experimental supervision yield."""
from copy import deepcopy
from dataclasses import replace
import unittest

from wam_reranking.recovery_observability_contract import (
    certify_frame,certify_temporal_motion,certify_supervision,FROZEN_CONTRACT)


ID=dict(dataset="paired160",suite="libero90",task=9,state=10,candidate_id=1,observed_block_id="pool/first_block")
GOOD=dict(mean_abs=0.,p95=0.,fraction_gt5=0.,psnr=99.)
HASH={"actual_rgb":"a"*64,"private_projection":"b"*64,"physical_measurement":"c"*64}


def frame(step,side="left",contact=True):
    role=dict(rendered_pixels=100,local_rgb=GOOD,identity_rgb_unique=True,identity_quality=.8,
        identity_basis="unique_scene_instance_with_actual_rgb_track",actual_rgb_structure_nonconstant=True)
    interface=dict(physics_contact=contact,other_surface_role="eef",projection_in_frame=True,
        depth_error_m=0.,rgb_boundary_pixels=3,local_rgb=GOOD,surfaces_rgb_visible=True,
        actual_surface_gap_px=4.,separation_rgb_verified=True)
    return dict(identity=ID,step=step,proprio_max_abs=0.,source_hashes_verified=True,evidence_sha256=HASH,
        views={"primary":dict(global_rgb=GOOD,roles={k:deepcopy(role) for k in ("target","eef","anchor","support")},
                              interfaces={side:deepcopy(interface)})})


def motion(start=1,end=3,relative=False):
    return dict(identity=ID,same_candidate_repeat_identity=ID,start_step=start,end_step=end,view="primary",
        source_hashes_verified=True,evidence_sha256=HASH,repeat_noise_verified=True,repeat_noise_px=.1,
        camera_motion_compensated=True,actual_rgb_tracking_verified=True,
        actual_rgb_tracking_basis="actual_rgb_geometric_correspondence",actual_rgb_tracking_points=3,
        actual_subject_delta_px=[6.,0.],actual_reference_delta_px=[0. if relative else 6.,0.],
        projected_subject_delta_px=[6.,0.],projected_reference_delta_px=[0. if relative else 6.,0.])


def atom(name="carried_sufficient_evidence",step=3):
    return dict(name=name,identity=ID,step=step,value=True,measurement_valid=True,supervision_mask=False,
        source="offline_simulator_measurement",source_hashes_verified=True,evidence_sha256=HASH,entity_kind="rigid")


class ObservationContractTests(unittest.TestCase):
    def test_global_registration_and_co_presence_are_not_contact(self):
        raw=frame(1);raw["views"]["primary"]["interfaces"]={}
        c=certify_frame(raw)
        self.assertTrue(c["views"]["primary"]["roles"]["target"]["available"])
        self.assertIsNone(c["views"]["primary"]["interfaces"]["left"]["value"])

    def test_contact_point_and_actual_rgb_interface_can_enable_one_head(self):
        c=certify_frame(frame(1));a=atom("left_finger_target_contact",1)
        self.assertTrue(certify_supervision(a,[c])["new_mask"])
        self.assertFalse(certify_supervision(atom("right_finger_target_contact",1),[c])["new_mask"])

    def test_hidden_depth_masks_contact(self):
        raw=frame(1);raw["views"]["primary"]["interfaces"]["left"]["depth_error_m"] = .01
        self.assertFalse(certify_supervision(atom("left_finger_target_contact",1),[certify_frame(raw)])["new_mask"])

    def test_bad_local_roi_masks_even_if_global_passes(self):
        raw=frame(1);raw["views"]["primary"]["roles"]["target"]["local_rgb"]={**GOOD,"mean_abs":3.}
        self.assertFalse(certify_supervision(atom("left_finger_target_contact",1),[certify_frame(raw)])["new_mask"])

    def test_no_sim_id_as_visual_identity_certificate(self):
        raw=frame(1);raw["views"]["primary"]["roles"]["target"]["identity_basis"]="sim_instance_id"
        self.assertFalse(certify_supervision(atom("left_finger_target_contact",1),[certify_frame(raw)])["new_mask"])

    def test_ambiguous_identical_instances_masked(self):
        raw=frame(1);raw["views"]["primary"]["roles"]["target"]["identity_rgb_unique"]=False
        self.assertIsNotNone(certify_frame(raw)["views"]["primary"]["roles"]["target"])
        self.assertFalse(certify_supervision(atom("left_finger_target_contact",1),[certify_frame(raw)])["new_mask"])

    def test_frame0_remains_diagnostic(self):
        c=certify_frame(frame(0))
        self.assertFalse(c["reference_certified"])
        self.assertFalse(certify_supervision(atom("left_finger_target_contact",0),[c])["new_mask"])

    def test_carrying_from_real_step1_can_be_certified(self):
        cs=[certify_frame(frame(s)) for s in range(1,4)]
        m=certify_temporal_motion(cs,motion())
        result=certify_supervision(atom(),cs,motion_cert=m)
        self.assertTrue(result["new_mask"])
        self.assertFalse(result["old_mask"])
        self.assertFalse(result["transition_mask"])

    def test_single_frame_contact_never_certifies_carrying(self):
        cs=[certify_frame(frame(s)) for s in (1,2)]
        self.assertFalse(certify_supervision(atom(step=2),cs,motion_cert=certify_temporal_motion(cs,motion(1,2)))["new_mask"])

    def test_no_physical_event_is_not_a_negative_grasp(self):
        a=atom();a.update(value=None,measurement_valid=False)
        result=certify_supervision(a,[certify_frame(frame(3))])
        self.assertIsNone(result["physical_value"])
        self.assertFalse(result["new_mask"])

    def test_repeat_from_candidate0_cannot_certify_candidate1(self):
        cs=[certify_frame(frame(s)) for s in range(1,4)];raw=motion()
        raw["same_candidate_repeat_identity"]={**ID,"candidate_id":0}
        self.assertIsNone(certify_temporal_motion(cs,raw)["value"])

    def test_signal_must_exceed_repeat_noise(self):
        cs=[certify_frame(frame(s)) for s in range(1,4)];raw=motion();raw["repeat_noise_px"]=2.
        c=certify_temporal_motion(cs,raw)
        self.assertFalse(c["common_motion"])

    def test_projection_alone_does_not_certify_motion(self):
        cs=[certify_frame(frame(s)) for s in range(1,4)];raw=motion();raw["actual_rgb_tracking_verified"]=None
        self.assertIsNone(certify_temporal_motion(cs,raw)["value"])

    def test_camera_motion_not_removed_is_unusable(self):
        cs=[certify_frame(frame(s)) for s in range(1,4)];raw=motion();raw["camera_motion_compensated"]=False
        self.assertIsNone(certify_temporal_motion(cs,raw)["value"])

    def test_projection_disagrees_with_actual_motion(self):
        cs=[certify_frame(frame(s)) for s in range(1,4)];raw=motion();raw["actual_subject_delta_px"]=[20.,0.]
        self.assertFalse(certify_temporal_motion(cs,raw)["value"])

    def test_hidden_joint_qpos_not_force_supervised(self):
        a=atom("joint_state_changed");a.update(entity_kind="articulated",joint_unit="m",joint_delta_from_reference=.02)
        self.assertFalse(certify_supervision(a,[certify_frame(frame(3))])["new_mask"])

    def test_visible_target_anchor_relative_joint_change(self):
        cs=[certify_frame(frame(s)) for s in range(1,4)]
        a=atom("joint_state_changed");a.update(entity_kind="articulated",joint_unit="m",joint_delta_from_reference=.02)
        m=certify_temporal_motion(cs,motion(relative=True),reference_role="anchor")
        self.assertTrue(certify_supervision(a,cs,motion_cert=m)["new_mask"])

    def test_release_requires_prior_certified_carry_and_real_opening(self):
        cs=[certify_frame(frame(s,contact=s<=3)) for s in range(1,7)]
        prior=certify_supervision(atom(),cs,motion_cert=certify_temporal_motion(cs,motion()))
        a=atom("released_sufficient_evidence",6);a.update(actual_proprio_opening_m=.005,opening_source="actual_proprio",
            actual_proprio_opening_start_step=3,actual_proprio_opening_end_step=6)
        m=certify_temporal_motion(cs,motion(4,6,relative=True))
        self.assertTrue(certify_supervision(a,cs,motion_cert=m,prior_event_cert=prior)["new_mask"])
        self.assertFalse(certify_supervision(a,cs,motion_cert=m)["new_mask"])

    def test_gripper_command_cannot_be_release_certificate(self):
        a=atom("released_sufficient_evidence",6);a["gripper_command"]=1.
        with self.assertRaises(ValueError):certify_supervision(a,[])

    def test_future_frame_cannot_backfill_current_certificate(self):
        raw=frame(2);raw["used_steps"]=[2,16]
        with self.assertRaises(ValueError):certify_frame(raw)

    def test_outcome_and_attribution_rejected(self):
        for name in ("success","attribution","predicted_cause"):
            raw=frame(2);raw[name]=True
            with self.assertRaises(ValueError):certify_frame(raw)

    def test_nonfinite_evidence_rejected(self):
        raw=frame(1);raw["proprio_max_abs"]=float("nan")
        with self.assertRaises(ValueError):certify_frame(raw)

    def test_masks_are_label_only_and_inputs_unchanged(self):
        raw=frame(1);before=deepcopy(raw);c=certify_frame(raw)
        self.assertEqual(raw,before)
        self.assertFalse(c["deployment_allowed"])
        self.assertFalse(certify_supervision(atom("left_finger_target_contact",1),[c])["deployment_allowed"])

    def test_hash_missing_keeps_geometry_evidence_unsupervised(self):
        raw=frame(1);raw["source_hashes_verified"]=False
        self.assertFalse(certify_supervision(atom("left_finger_target_contact",1),[certify_frame(raw)])["new_mask"])

    def test_contract_is_not_relaxed(self):
        with self.assertRaises(ValueError):certify_frame(frame(1),contract=replace(FROZEN_CONTRACT,image_mean_abs=3.))

    def test_motion_needs_three_actual_rgb_correspondences(self):
        cs=[certify_frame(frame(s)) for s in range(1,4)]
        for points,basis in ((2,"actual_rgb_geometric_correspondence"),(3,"sim_id_track")):
            raw=motion();raw.update(actual_rgb_tracking_points=points,actual_rgb_tracking_basis=basis)
            self.assertIsNone(certify_temporal_motion(cs,raw)["value"])

    def test_duplicate_frame_does_not_hide_scope_conflict(self):
        c=certify_frame(frame(1));other=deepcopy(c);other["identity"]["candidate_id"]=2
        with self.assertRaises(ValueError):certify_temporal_motion([c,other],motion())

    def test_measurement_hash_required_to_link_positive_label(self):
        a=atom("left_finger_target_contact",1);a["evidence_sha256"]={"actual_rgb":"a"*64}
        self.assertFalse(certify_supervision(a,[certify_frame(frame(1))])["new_mask"])

    def test_carry_motion_cannot_come_from_unrelated_anchor(self):
        cs=[certify_frame(frame(s)) for s in range(1,4)]
        m=certify_temporal_motion(cs,motion(),reference_role="anchor")
        self.assertFalse(certify_supervision(atom(),cs,motion_cert=m)["new_mask"])

    def test_motion_prefix_cannot_certify_later_sustained_window(self):
        cs=[certify_frame(frame(s)) for s in range(1,7)]
        m=certify_temporal_motion(cs,motion(1,6))
        self.assertFalse(certify_supervision(atom(step=6),cs,motion_cert=m)["new_mask"])

    def test_articulated_body_never_carried(self):
        cs=[certify_frame(frame(s)) for s in range(1,4)];a=atom();a["entity_kind"]="articulated"
        self.assertFalse(certify_supervision(a,cs,motion_cert=certify_temporal_motion(cs,motion()))["new_mask"])

    def test_foreign_predicated_before_cannot_enable_transition(self):
        cs=[certify_frame(frame(s)) for s in range(1,4)]
        before=certify_supervision(atom("left_finger_target_contact",1),cs)
        after=certify_supervision(atom(),cs,motion_cert=certify_temporal_motion(cs,motion()),before_certificate=before)
        self.assertTrue(after["after_mask"]);self.assertFalse(after["before_mask"]);self.assertFalse(after["transition_mask"])

    def test_contact_before_false_to_after_true_uses_both_actual_endpoints(self):
        cs=[certify_frame(frame(1,contact=False)),certify_frame(frame(2,contact=True))]
        a=atom("left_finger_target_contact",1);a["value"]=False
        before=certify_supervision(a,cs)
        result=certify_supervision(atom("left_finger_target_contact",2),cs,before_certificate=before)
        self.assertTrue(result["before_mask"]);self.assertTrue(result["after_mask"]);self.assertTrue(result["transition_mask"])

    def test_foreign_prior_event_does_not_certify_release(self):
        cs=[certify_frame(frame(s,contact=s<=3)) for s in range(1,7)]
        prior=certify_supervision(atom(),cs,motion_cert=certify_temporal_motion(cs,motion()));prior["physical_value"]=False
        a=atom("released_sufficient_evidence",6);a.update(actual_proprio_opening_m=.005,opening_source="actual_proprio",
            actual_proprio_opening_start_step=3,actual_proprio_opening_end_step=6)
        m=certify_temporal_motion(cs,motion(4,6,relative=True))
        self.assertFalse(certify_supervision(a,cs,motion_cert=m,prior_event_cert=prior)["new_mask"])

    def test_lift_requires_certified_current_carry_not_boolean_claim(self):
        raw=[frame(s) for s in range(1,5)]
        for s,f in enumerate(raw,1):
            interface=deepcopy(f["views"]["primary"]["interfaces"]["left"])
            interface.update(other_surface_role="support",physics_contact=s==1)
            f["views"]["primary"]["interfaces"]["support"]=interface
        cs=[certify_frame(f) for f in raw]
        support=certify_supervision(atom("target_support_contact",1),cs)
        carry=certify_supervision(atom(step=4),cs,motion_cert=certify_temporal_motion(cs,motion(2,4)))
        a=atom("lifted_sufficient_evidence",4);a.update(vertical_rise_from_reference_m=.02,
            projected_vertical_direction_px=[1.,0.],vertical_component_identifiable=True,certified_carried_at_step=True)
        m=certify_temporal_motion(cs,motion(1,4,relative=True),reference_role="support")
        self.assertFalse(certify_supervision(a,cs,motion_cert=m,support_reference_cert=support)["new_mask"])
        self.assertTrue(certify_supervision(a,cs,motion_cert=m,support_reference_cert=support,carried_event_cert=carry)["new_mask"])

    def test_missing_window_frames_is_mask_not_keyerror(self):
        cs=[certify_frame(frame(s)) for s in range(1,4)];m=certify_temporal_motion(cs,motion())
        self.assertFalse(certify_supervision(atom(),[cs[-1]],motion_cert=m)["new_mask"])


if __name__=="__main__":unittest.main()
